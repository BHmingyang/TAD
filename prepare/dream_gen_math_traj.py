"""

Generate trajectories using the Dream diffusion language model on GSM8k / competition_math datasets.

Core logic:
  - Each diffusion step decodes only 1 token (steps = max_new_tokens)
  - If the model's final answer is correct, collect the complete trajectory (token sequence at each step)
  - Trajectories are written in JSONL format, consistent with trajectory_generation.py

Usage example:
  python dream_gen_math_traj.py \
      --model_name Dream-org/Dream-v0-Instruct-7B \
      --output_path outputs/gsm8k_dream_traj.jsonl \
      --dataset_name gsm8k \
      --dataset_config main \
      --dataset_split train \
      --max_new_tokens 256 \
      --num_samples 1 \
      --temperature 0.2 \
      --top_p 0.95 \
      --limit 100
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# -- Reuse Parser / is_equiv --
# Prefer importing from sibling eval directory; fallback to inline minimal version
_DLLM_EVAL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "TAD",
    "eval",
)
if os.path.isdir(_DLLM_EVAL_PATH):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from dllm_inference_acceleration_main.eval.parsers import Parser, is_equiv  # type: ignore
else:
    import re as _re

    def _remove_boxed(s):
        if "\\boxed " in s:
            return s[len("\\boxed "):]
        left = "\\boxed{"
        try:
            assert s[: len(left)] == left and s[-1] == "}"
            return s[len(left): -1]
        except Exception:
            return s

    def _last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return string
        i, right_brace_idx, num_open = idx, None, 0
        while i < len(string):
            if string[i] == "{":
                num_open += 1
            if string[i] == "}":
                num_open -= 1
                if num_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        return string[idx: right_brace_idx + 1] if right_brace_idx is not None else None

    class Parser:
        @classmethod
        def extract_answer_gsm8k(cls, text):
            try:
                m = _re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", text)
                if m:
                    return float(m.group(1).replace(",", ""))
            except Exception:
                pass
            return None

        @classmethod
        def extract_answer_boxed(cls, text):
            try:
                return _remove_boxed(_last_boxed_only_string(text))
            except Exception:
                return text

    def _strip_string(s):
        s = s.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
        s = s.replace("tfrac", "frac").replace("dfrac", "frac")
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("^{\\circ}", "").replace("^\\circ", "")
        s = s.replace("\\$", "").replace("\\%", "").replace("\%", "")
        s = s.replace(" .", " 0.").replace("{.", "{0.")
        if not s:
            return s
        if s[0] == ".":
            s = "0" + s
        if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
            s = s.split("=")[1]
        s = s.replace(" ", "")
        return s

    def is_equiv(str1, str2, verbose=False):
        if isinstance(str1, float) or isinstance(str2, float):
            try:
                return abs(float(str1) - float(str2)) < 1e-6
            except Exception:
                return False
        if str1 is None and str2 is None:
            return True
        if str1 is None or str2 is None:
            return False
        try:
            return _strip_string(str(str1)) == _strip_string(str(str2))
        except Exception:
            return str1 == str2


# -- Helper functions --

def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def extract_last_number(text: str):
    nums = re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_pred(text: str):
    pred = Parser.extract_answer_boxed(text)
    if pred is None:
        pred = Parser.extract_answer_gsm8k(text)
    if pred is None:
        return extract_last_number(text)
    pred_num = extract_last_number(pred)
    return pred_num if pred_num is not None else pred


def extract_example(example, question_key: str, answer_key: str):
    """Extract question, answer, and dataset type from a dataset sample."""
    if question_key and answer_key:
        return example[question_key], example[answer_key], "custom"
    if "question" in example and "answer" in example:
        return example["question"], example["answer"], "gsm8k"
    if "problem" in example and "solution" in example:
        return example["problem"], example["solution"], "math"
    raise ValueError(
        "Cannot infer question/answer fields, please specify via --question_key and --answer_key."
    )


# -- Trajectory generation (core) --

@torch.no_grad()
def generate_dream_trajectory(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    block_length: int=32,
):
    """
    Call Dream model's diffusion_generate interface with steps = max_new_tokens,
    i.e., each diffusion step decodes only 1 token.

    Returns:
        output_text  : str, final generated text (truncated after eos)
        traj_dict    : dict, keys are "step0", "step1", ..., values are token id lists for the generation region
                       (consistent with trajectory_generation.py format)
    """
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=True,          # Collect intermediate states
        return_dict_in_generate=True,
        steps=max_new_tokens,         # Each step generates only 1 token
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        block_length=block_length,
    )

    # Decode final output
    prompt_len = input_ids.shape[1]
    final_ids = output.sequences[0]          # shape: (prompt_len + gen_len,)
    gen_ids = final_ids[prompt_len:].tolist()
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    # Truncate at first eos
    eos_token = tokenizer.eos_token
    if eos_token and eos_token in output_text:
        output_text = output_text.split(eos_token)[0]

    # Organize trajectory: output.history is a list, each element shape=(batch, full_seq_len)
    # Only keep the generation region (after prompt), consistent with trajectory_generation.py
    traj_dict = {}
    history = output.history  # List[Tensor(batch, seq_len)]
    for step_idx, h in enumerate(history):
        # h: (batch_size, prompt_len + gen_len), take sample 0's generation region
        gen_part = h[0, prompt_len:].tolist()
        traj_dict[f"step{step_idx}"] = gen_part

    return output_text, traj_dict


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="Generate math reasoning trajectories using Dream diffusion language model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Dream model path or HuggingFace Hub ID",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens (also the number of diffusion steps)")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--alg", type=str, default="entropy", help="Dream decoding algorithm (entropy / origin etc.)")
    parser.add_argument("--alg_temp", type=float, default=0.0, help="Dream decoding algorithm temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Max samples to process, 0 means all")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Dataset name",
    )
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset config")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--question_key", type=str, default="", help="Custom question field name")
    parser.add_argument("--answer_key", type=str, default="", help="Custom answer field name")
    args = parser.parse_args()

    # -- Random seed --
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -- Device and model --
    device = select_device()
    dtype_map = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}
    dtype = dtype_map[device]
    print(f"[INFO] Device: {device}  dtype={dtype}")

    print(f"[INFO] Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="left"
    )

    # -- Dataset --
    print(f"[INFO] Loading dataset: {args.dataset_name}/{args.dataset_config} split={args.dataset_split}")
    if args.dataset_config is not None:
      ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
      ds = load_dataset(args.dataset_name, split=args.dataset_split)

    # -- Output file --
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    solved = 0

    for idx in tqdm(range(total), desc=f"dataset={args.dataset_name}"):
        raw_q, raw_a, kind = extract_example(ds[idx], args.question_key, args.answer_key)

        # Extract ground-truth answer
        if kind == "gsm8k":
            answer_gt = Parser.extract_answer_gsm8k(raw_a)
        else:
            # math / custom: prefer boxed, otherwise raw text
            answer_gt = Parser.extract_answer_boxed(raw_a)

        # Build question text: provide reference answer to the model, let it re-solve in its own way
        question = (
            f"Question: {raw_q}\n"
            f"Reference Answer: {raw_a}\n"
            f"After understanding the reference answer, please try to solve this problem "
            f"using your own approach below and output a detailed solution process:"
        )

        # Build prompt and tokenize (batch_size=1, Dream does not support batch trajectory collection)
        prompt_str = build_prompt(tokenizer, question)
        encoded = tokenizer(
            [prompt_str],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        counted = False  # Only count solved once per problem

        for sample_id in range(args.num_samples):
            output_text, traj_dict = generate_dream_trajectory(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.alg,
                alg_temp=args.alg_temp,
                block_length=args.block_length,
            )
            print(f"output_text:{output_text}")

            pred = extract_pred(output_text)
            print(f"[idx={idx} sample={sample_id}] pred={pred}  gt={answer_gt}")

            if is_equiv(pred, answer_gt):
                if not counted:
                    solved += 1
                    counted = True

                # Collect trajectory record (format consistent with trajectory_generation.py)
                record = {
                    "dataset_name": args.dataset_name,
                    "dataset_config": args.dataset_config,
                    "dataset_split": args.dataset_split,
                    "index": idx,
                    "sample_id": sample_id,
                    "prompt": prompt_str,
                    "question": question,
                    "answer_gt": answer_gt,
                    "pred": pred,
                    "prompt_len": int(input_ids.shape[1]),
                    "trajectory": traj_dict,
                    "output_text": output_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    # -- Summary --
    accuracy = solved / total if total > 0 else 0.0
    print(f"Accuracy_any_success: {solved}/{total} = {accuracy * 100:.2f}%")
    out_f.write(
        json.dumps(
            {
                "summary": "any_success",
                "solved": solved,
                "total": total,
                "accuracy": accuracy,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    out_f.close()


if __name__ == "__main__":
    main()
