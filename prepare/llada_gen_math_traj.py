import argparse
import json
import math
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random

import re
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.parsers import Parser, is_equiv


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_trajectory(
    model,
    prompt,
    attention_mask=None,
    steps=16,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
):
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )

    prompt_index = x != mask_id
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    trajectory = {}
    step_idx = 0
    trajectory[f"step{step_idx}"] = x[:, prompt.shape[1]:].clone()
    step_idx += 1

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            trajectory[f"step{step_idx}"] = x[:, prompt.shape[1]:].clone()
            step_idx += 1

    return x, trajectory


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def extract_last_number(text):
    nums = re.findall(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def extract_pred(text):
    pred = Parser.extract_answer_boxed(text)
    if pred is None:
        pred = Parser.extract_answer_gsm8k(text)
    if pred is None:
        return extract_last_number(text)
    pred_num = extract_last_number(pred)
    return pred_num if pred_num is not None else pred


def extract_example(example, question_key, answer_key):
    if question_key and answer_key:
        return example[question_key], example[answer_key], "custom"
    if "question" in example and "answer" in example:
        return example["question"], example["answer"], "gsm8k"
    if "problem" in example and "solution" in example:
        return example["problem"], example["solution"], "math"
    raise ValueError("Cannot infer question/answer fields, please set --question_key and --answer_key.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Model Path")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--question_key", type=str, default="")
    parser.add_argument("--answer_key", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    assert tokenizer.pad_token_id != 126336

    if args.dataset_config is not None:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset_name, split=args.dataset_split)


    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_f = open(args.output_path, "w", encoding="utf-8")

    total = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    solved = 0
    for idx in tqdm(range(total), desc=f"dataset={args.dataset_name}"):
        example = ds[idx]
        raw_q, raw_a, kind = extract_example(example, args.question_key, args.answer_key)
        question = f"Question: {raw_q}\nAnswer:"
        answer = example.get("answer", None)
        if answer is None:
            answer = example.get("solution", None)
        
        if kind == "gsm8k":
            answer_gt = Parser.extract_answer_gsm8k(raw_a)
        elif kind == "math":
            answer_gt = Parser.extract_answer_boxed(raw_a)
        else:
            answer_gt = Parser.extract_answer_boxed(raw_a)

        question = (
            f"Question: {raw_q}\n"
            f"Reference Answer: {raw_a}\n"
            f"After understanding the reference answer, please try to solve this problem "
            f"using your own approach below and output a detailed solution process:"
        )

        prompt = build_prompt(tokenizer, question)
        encoded = tokenizer([prompt], add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        counted = False

        for sample_id in range(args.num_samples):
            steps = args.steps
            gen_length = args.max_new_tokens
            block_length = args.block_length

            final_x, trajectory = generate_trajectory(
                model,
                input_ids,
                attention_mask=attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=args.temperature,
                cfg_scale=0.0,
                remasking="low_confidence",
            )

            output_text = tokenizer.batch_decode(final_x[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
            pred = extract_pred(output_text)
            print(f"Pred: {pred}")
            if is_equiv(pred, answer_gt):
                if not counted:
                    solved += 1
                    counted = True
                traj_dict = {k: v.squeeze(0).tolist() for k, v in trajectory.items()}
                record = {
                    "dataset_name": args.dataset_name,
                    "index": idx,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "question": question,
                    "answer_gt": answer_gt,
                    "pred": pred,
                    "prompt_len": int(input_ids.shape[1]),
                    "trajectory": traj_dict,
                    "output_text": output_text,
                    "answer": answer,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    print(f"Accuracy_any_success: {solved}/{total} = {solved/total*100:.2f}%")
    out_f.write(json.dumps({"summary": "any_success", "solved": solved, "total": total, "accuracy": solved/total}, ensure_ascii=False) + "\n")
    out_f.close()


if __name__ == "__main__":
    main()
