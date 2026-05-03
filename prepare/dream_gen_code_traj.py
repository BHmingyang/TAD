"""
Dream_code_traj.py

Generate trajectories using the Dream diffusion language model on the KodCode code dataset.

Core logic:
  - Provide the problem + reference answer together to the Dream model, letting it re-solve in its own way
  - Each diffusion step decodes only 1 token (steps = max_new_tokens)
  - Only keep trajectories that pass all test cases
  - Output JSONL, with an extra "prompt" field (question with chat_template applied)

Usage example:
  python Dream_code_traj.py \
      --model_name Dream-org/Dream-v0-Instruct-7B \
      --parquet_path /path/to/KodCode-V1-SFT-R1/data/train-00000-of-00011.parquet \
      --output_path outputs/kodcode_dream_traj.jsonl \
      --max_new_tokens 256 \
      --num_samples 1 \
      --temperature 0.2 \
      --top_p 0.95 \
      --target_count 10000
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer



def select_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"




def build_prompt(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )




def extract_code_from_text(text: str) -> str:
    """Extract ```python ... ``` code block from model output; return original text if none found."""
    parts = text.split("```")
    if len(parts) >= 3:
        body = parts[1]
        if body.strip().lower().startswith("python"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        return body
    return text




def run_tests(solution_code: str, test_code: str, timeout: int = 20) -> bool:
    """
    Write solution_code and test_code to a temp directory and run all test_* functions.
    Return True if all pass, False otherwise.
    """
    tmp = tempfile.mkdtemp(prefix="kodcode_dream_")
    try:
        sol_path = os.path.join(tmp, "solution.py")
        tst_path = os.path.join(tmp, "test_content.py")
        runner_path = os.path.join(tmp, "test_runner.py")

        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_code)
        with open(tst_path, "w", encoding="utf-8") as f:
            f.write(test_code)

        runner_src = (
            "import importlib.util, sys\n"
            "sys.path.insert(0, '.')\n"
            "spec = importlib.util.spec_from_file_location('solution', 'solution.py')\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "g = {'__name__': '__main__'}\n"
            "exec(open('test_content.py', 'r', encoding='utf-8').read(), g)\n"
            "fails = 0\n"
            "for k, v in list(g.items()):\n"
            "    if callable(v) and k.startswith('test_'):\n"
            "        try:\n"
            "            v()\n"
            "        except Exception:\n"
            "            fails += 1\n"
            "import sys\n"
            "sys.exit(1 if fails > 0 else 0)\n"
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_src)

        p = subprocess.run(
            [sys.executable, runner_path],
            cwd=tmp,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode == 0
    except Exception:
        return False
    finally:
        try:
            for fn in os.listdir(tmp):
                fp = os.path.join(tmp, fn)
                if os.path.isfile(fp):
                    os.remove(fp)
            os.rmdir(tmp)
        except Exception:
            pass



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
        output_text : str, final generated text (truncated at first eos)
        traj_dict   : dict, keys are "step0", "step1", ..., values are token id lists for the generation region at that step
    """
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=True,           # Collect intermediate states
        return_dict_in_generate=True,
        steps=max_new_tokens,          # Each step generates only 1 token
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
    eos_token = tokenizer.eos_token
    if eos_token and eos_token in output_text:
        output_text = output_text.split(eos_token)[0]

    # Organize trajectory: output.history is a list, each element shape=(batch, full_seq_len)
    # Only keep the generation region (after prompt)
    traj_dict = {}
    for step_idx, h in enumerate(output.history):
        gen_part = h[0, prompt_len:].tolist()
        traj_dict[f"step{step_idx}"] = gen_part

    return output_text, traj_dict


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="Generate code trajectories using Dream diffusion language model (only keep samples passing all tests)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Dream model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="",
        help="KodCode Parquet file path (single file)",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process, 0 means all")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generated tokens (also the number of diffusion steps)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    parser.add_argument("--alg", type=str, default="entropy", help="Dream decoding algorithm (entropy / origin etc.)")
    parser.add_argument("--alg_temp", type=float, default=0.0, help="Dream decoding algorithm temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target_count", type=int, default=10000, help="Target trajectory count, exit early when reached")
    parser.add_argument("--test_timeout", type=int, default=30, help="Timeout in seconds for each test execution")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # -- Output file (supports resume) --
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    existing = set()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = (int(obj.get("row_id", -1)), int(obj.get("sample_id", -1)))
                    existing.add(key)
                except Exception:
                    continue
        print(f"[INFO] Found {len(existing)} existing records, will skip duplicates (resume mode)")

    out_f = open(args.output_path, "a", encoding="utf-8")

    # -- Read Parquet --
    pf = pq.ParquetFile(args.parquet_path)
    total_rows = sum(pf.read_row_group(i).num_rows for i in range(pf.num_row_groups))
    display_total = total_rows if args.limit <= 0 else min(args.limit, total_rows)

    collected = 0
    total_attempts = 0   
    processed = 0
    progress = tqdm(total=display_total, desc="processing")

    for i in range(pf.num_row_groups):
        if args.limit > 0 and processed >= args.limit:
            break
        if collected >= args.target_count:
            break

        rg = pf.read_row_group(i)
        tbl = rg.to_pydict()
        n = rg.num_rows

        for r in range(n):
            if args.limit > 0 and processed >= args.limit:
                break
            if collected >= args.target_count:
                break

            processed += 1
            progress.update(1)

            row = {k: tbl[k][r] for k in tbl.keys()}
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict) and "row_id" in metadata:
                row_id = int(metadata["row_id"])
            else:
                row_id = processed

            question = str(row.get("question", ""))
            solution_ref = str(row.get("solution", ""))
            test_code = str(row.get("test", ""))

            teacher_text = (
                question
                + "\nReference Answer: "
                + solution_ref
                + "\nAfter understanding the reference solution, please try to solve this problem "
                "using your own approach below and just OUTPUT YOUR SOLUTION CODE, DON'T EXPLAIN IT:"
            )

            prompt_str = build_prompt(tokenizer, question)

            teacher_prompt_str = build_prompt(tokenizer, teacher_text)

            encoded = tokenizer(
                [teacher_prompt_str],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            for sample_id in range(args.num_samples):
                if (row_id, sample_id) in existing:
                    continue

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
                total_attempts += 1

                code_text = extract_code_from_text(output_text)

                # Only keep trajectories that pass all test cases
                passed = run_tests(code_text, test_code, timeout=args.test_timeout)
                success_rate = collected / total_attempts * 100 if total_attempts > 0 else 0.0
                if not passed:
                    print(f"[idx={row_id} sample={sample_id}] x Test failed, skipped  "
                          f"(success rate: {collected}/{total_attempts} = {success_rate:.2f}%)")
                    continue

                print(f"[idx={row_id} sample={sample_id}] v Test passed, collecting trajectory  "
                      f"(success rate: {collected + 1}/{total_attempts} = {(collected + 1) / total_attempts * 100:.2f}%)")

                rec = {
                    "row_id": int(row_id),
                    "sample_id": int(sample_id),
                    "prompt": prompt_str,          # Added: question with chat_template applied
                    "question": question,
                    "solution_ref": solution_ref,
                    "teacher_text": teacher_text,
                    # "trajectory": traj_dict,
                    "output_text": output_text,
                    "solution_code": code_text,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                collected += 1

                if collected >= args.target_count:
                    break

    progress.close()
    out_f.close()

    # -- Summary statistics --
    final_rate = collected / total_attempts * 100 if total_attempts > 0 else 0.0
    summary = {
        "summary": "collection_stats",
        "collected": collected,
        "total_attempts": total_attempts,
        "success_rate": round(final_rate, 4),
        "processed_rows": processed,
        "target_count": args.target_count,
    }
    # Append to JSONL
    with open(args.output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[Done] Collected {collected} trajectories / Total attempts {total_attempts} / "
          f"Success rate {final_rate:.2f}% / Processed rows {processed}")
    print(f"[Done] Output: {args.output_path}")


if __name__ == "__main__":
    main()
