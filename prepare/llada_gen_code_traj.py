import argparse
import json
import os
import random
import sys
import tempfile
import subprocess
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

MASK_ID = 126336

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
def generate_trajectory(model, prompt_ids, attention_mask=None, steps=16, gen_length=256, block_length=32, temperature=0.0, remasking="low_confidence"):
    x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length), MASK_ID, dtype=torch.long).to(model.device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((prompt_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)],
            dim=-1,
        )
    prompt_index = x != MASK_ID
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    trajectory = {}
    step_idx = 0
    trajectory[f"step{step_idx}"] = x[:, prompt_ids.shape[1]:].clone()
    step_idx += 1
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt_ids.shape[1] + num_block * block_length: prompt_ids.shape[1] + (num_block + 1) * block_length] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == MASK_ID
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length :] = -torch.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            trajectory[f"step{step_idx}"] = x[:, prompt_ids.shape[1]:].clone()
            step_idx += 1
    return x, trajectory

def build_prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def extract_code_from_text(text):
    l = text.split("```")
    if len(l) >= 3:
        body = l[1]
        if body.strip().lower().startswith("python"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        return body
    return text

def run_tests(solution_code, test_code, timeout=20):
    tmp = tempfile.mkdtemp(prefix="kodcode_")
    try:
        sol_path = os.path.join(tmp, "solution.py")
        tst_path = os.path.join(tmp, "test_content.py")
        runner_path = os.path.join(tmp, "test_runner.py")
        with open(sol_path, "w", encoding="utf-8") as f:
            f.write(solution_code)
        with open(tst_path, "w", encoding="utf-8") as f:
            f.write(test_code)
        runner_src = (
            "import importlib.util,sys\n"
            "sys.path.insert(0,'.')\n"
            "spec=importlib.util.spec_from_file_location('solution','solution.py')\n"
            "m=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "g={'__name__':'__main__'}\n"
            "exec(open('test_content.py','r',encoding='utf-8').read(),g)\n"
            "fails=0\n"
            "for k,v in list(g.items()):\n"
            "    if callable(v) and k.startswith('test_'):\n"
            "        try:\n"
            "            v()\n"
            "        except Exception:\n"
            "            fails+=1\n"
            "import sys\n"
            "sys.exit(1 if fails>0 else 0)\n"
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_src)
        p = subprocess.run([sys.executable, runner_path], cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Model Path")
    parser.add_argument("--parquet_path", type=str, default="Model Path")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_count", type=int, default=500)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
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
    out_f = open(args.output_path, "a", encoding="utf-8")
    pf = pq.ParquetFile(args.parquet_path)
    total_rows = 0
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        total_rows += rg.num_rows
    target = args.target_count
    collected = 0
    total_attempts = 0   # Total generation attempts (including passed and failed)
    processed = 0
    progress = tqdm(total=total_rows if args.limit <= 0 else min(args.limit, total_rows), desc="processing")
    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        tbl = rg.to_pydict()
        n = rg.num_rows
        for r in range(n):
            if args.limit > 0 and processed >= args.limit:
                break
            processed += 1
            progress.update(1)
            row = {k: tbl[k][r] for k in tbl.keys()}
            row_id = int(row.get("metadata", {}).get("row_id")) if isinstance(row.get("metadata"), dict) and "row_id" in row["metadata"] else processed
            for sample_id in range(args.num_samples):
                if (row_id, sample_id) in existing:
                    continue
                question = str(row.get("question", ""))
                solution_ref = str(row.get("solution", ""))
                test_code = str(row.get("test", ""))
                teacher_text = question + "\nReference Answer: " + solution_ref + "\nAfter understanding the reference solution, please try to solve this problem using your own approach below and just OUTPUT YOUR SOLUTION CODE, DON'T EXPLAIN IT:"
                prompt_text = build_prompt(tokenizer, teacher_text)
                encoded = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)
                final_x, trajectory = generate_trajectory(
                    model,
                    input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    gen_length=args.max_new_tokens,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking="low_confidence",
                )
                output_text = tokenizer.batch_decode(final_x[:, input_ids.shape[1] :], skip_special_tokens=True)[0]
                code_text = extract_code_from_text(output_text)
                passed = run_tests(code_text, test_code, timeout=30)
                total_attempts += 1
                success_rate = collected / total_attempts * 100 if total_attempts > 0 else 0.0
                if not passed:
                    print(f"[idx={row_id} sample={sample_id}] x Test failed, skipped  "
                          f"(success rate: {collected}/{total_attempts} = {success_rate:.2f}%)")
                    continue
                print(f"[idx={row_id} sample={sample_id}] v Test passed, collecting trajectory  "
                      f"(success rate: {collected + 1}/{total_attempts} = {(collected + 1) / total_attempts * 100:.2f}%)")
                traj_dict = {k: v.squeeze(0).tolist() for k, v in trajectory.items()}
                rec = {
                    "row_id": int(row_id),
                    "sample_id": int(sample_id),
                    "question": question,
                    "solution_ref": solution_ref,
                    "teacher_text": teacher_text,
                    "trajectory": traj_dict,
                    "output_text": output_text,
                    "solution_code": code_text,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                collected += 1
                if collected >= target:
                    break
            if collected >= target:
                break
        if collected >= target:
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
        "target_count": target,
    }
    with open(args.output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[Done] Collected {collected} trajectories / Total attempts {total_attempts} / "
          f"Success rate {final_rate:.2f}% / Processed rows {processed}")
    print(f"[Done] Output: {args.output_path}")

if __name__ == "__main__":
    main()
