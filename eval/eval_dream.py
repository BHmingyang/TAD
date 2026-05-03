import accelerate
import torch
import random
import numpy as np
import types
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoConfig, AutoModel
import json
import time
from pathlib import Path
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datetime import timedelta
from model.generation_utils_dream import DreamGenerationMixin


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("dream_dist")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path="Dream-org/Dream-v0-Instruct-7B",
        max_length=2048,
        batch_size=1,
        mc_num=128,
        diffusion_steps=128,
        max_new_tokens=128,
        temperature=0.0,
        top_p=None,
        top_k=None,
        alg="entropy",
        alg_temp=0.0,
        device="cuda",
        remasking="low_confidence",
        block_length=None,
        threshold=0.9,
        save_dir=None,
        stats_dir=None,
        show_speed=False,
        multi_block=False,
        block_add_threshold=0.5,
        decoded_token_threshold=0.5,
        early_stop=False,
        task="null",
        nll_type="mc",
        log_type="nll",
        classifier_free_guidance=0.0,
        sampling_eps=0.0,
        add_bos_token=False,
        escape_until=False,
        **kwargs,
    ):
        super().__init__()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            config=config,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.model.to(self.accelerator.device)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Dream uses mask_token_id from tokenizer or config
        if hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
            self.mask_id = self.tokenizer.mask_token_id
        elif hasattr(config, "mask_token_id") and config.mask_token_id is not None:
            self.mask_id = config.mask_token_id
        else:
            # Default Dream mask token id
            self.mask_id = 151666

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = float(sampling_eps)
        self.max_length = int(max_length)
        self.diffusion_steps = int(diffusion_steps)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p) if top_p is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.alg = str(alg)
        self.alg_temp = float(alg_temp) if alg_temp is not None else None
        self.remasking = remasking
        self.classifier_free_guidance = float(classifier_free_guidance)
        self.add_bos_token = add_bos_token if isinstance(add_bos_token, bool) else str(add_bos_token).lower() == "true"
        self.escape_until = escape_until if isinstance(escape_until, bool) else str(escape_until).lower() == "true"

        self.is_instruct = True if ("instruct" in model_path.lower()) else False
        self.save_dir = save_dir
        self.stats_dir = stats_dir if stats_dir is not None else save_dir
        self.show_speed = show_speed if isinstance(show_speed, bool) else str(show_speed).lower() == "true"
        self.multi_block = multi_block if isinstance(multi_block, bool) else str(multi_block).lower() == "true"
        self.block_add_threshold = float(block_add_threshold)
        self.decoded_token_threshold = float(decoded_token_threshold)
        self.early_stop = early_stop if isinstance(early_stop, bool) else str(early_stop).lower() == "true"
        self.task = task
        self.nll_type = nll_type
        self.log_type = log_type
        self.block_length = int(block_length) if block_length is not None else None
        self.threshold = float(threshold)

        # Monkey-patch CDLM's diffusion_generate and _sample onto the model
        # This replaces the HuggingFace official methods with CDLM's versions
        # that support confidence_threshold dynamic decoding and block-wise generation
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model
        )
        self.model._sample = types.MethodType(
            DreamGenerationMixin._sample, self.model
        )
        self.model.generate_multi_block = types.MethodType(
            DreamGenerationMixin.generate_multi_block, self.model
        )
        self.model._sample_multi_block = types.MethodType(
            DreamGenerationMixin._sample_multi_block, self.model
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    # ── loglikelihood helpers ──────────────────────────────────────────────

    def _forward_process(self, batch, prompt_index):
        """
        Apply forward diffusion noise to the target portion of the batch.
        Returns noisy_batch and the noise level p_mask for each position.
        """
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()

        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask),
            dim=1,
        )
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        """
        Get logits from the Dream model.
        Dream uses a shifted logits convention:
            logits = cat([logits[:, :1], logits[:, :-1]], dim=1)
        so that logits[i] predicts token[i] (not token[i+1] as in AR models).
        """
        if self.classifier_free_guidance > 0.0:
            assert len(prompt_index) == batch.shape[1]
            pi = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[pi] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        # Dream logits shift: logits = cat([logits[:,:1], logits[:,:-1]], dim=1)
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        if self.classifier_free_guidance > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.classifier_free_guidance + 1) * (logits - un_logits)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        """
        Monte Carlo estimation of log-likelihood for the target given the prefix.
        """
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(
                logits[mask_indices], seq[mask_indices], reduction="none"
            ) / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        """
        Check if greedy decoding from the model matches the target.
        """
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for _ in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()

        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= self.max_length, (
            f"Max prompt length {max(prompt_len)} exceeds max_length {self.max_length}"
        )

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))

        torch.cuda.empty_cache()
        return out

    # ── generation ────────────────────────────────────────────────────────

    def _generate_batch(self, prompts):
        """
        Call Dream model's diffusion_generate method (CDLM version) for a batch of prompts.
        Accepts a list of prompt strings, tokenizes internally, generates, and decodes.
        Returns (responses: List[str], nfe: int).
        """
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id).to(device=self.device)

        generation_result = self.model.diffusion_generate(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            threshold=self.threshold,
            block_length=self.block_length,
            early_stop=self.early_stop,
        )

        nfe = int(getattr(generation_result, 'nfe', self.diffusion_steps))

        # decode only the generated part (after prompt)
        responses = [
            self.tokenizer.decode(g[len(p):].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_result.sequences)
        ]

        return responses, nfe

    def _generate_batch_multi_block(self, prompts):
        """
        Call Dream model's generate_multi_block method for a batch of prompts.
        Uses pipelined parallel decoding with multi-block strategy.
        Accepts a list of prompt strings, tokenizes internally, generates, and decodes.
        Returns (responses: List[str], nfe: int).
        """
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id).to(device=self.device)

        generation_result, nfe = self.model.generate_multi_block(
            prompt_ids,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            threshold=self.threshold,
            block_size=self.block_length if self.block_length is not None else 32,
            block_add_threshold=self.block_add_threshold,
            decoded_token_threshold=self.decoded_token_threshold,
            early_stop=self.early_stop,
        )

        nfe = int(nfe)

        # decode only the generated part (after prompt)
        responses = [
            self.tokenizer.decode(g[len(p):].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, generation_result.sequences)
        ]

        return responses, nfe

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        start_time = time.time()
        log_fh = None

        # ── Checkpoint resume ─────────────────────────────────────────────
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f"rank_{rank}.jsonl")
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, "r", encoding="utf-8") as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")

        if self.stats_dir is not None:
            os.makedirs(self.stats_dir, exist_ok=True)
            stats_samples_path = os.path.join(self.stats_dir, f"rank_{self.rank}_samples.jsonl")
            log_fh = open(stats_samples_path, "a", encoding="utf-8")

        # ── Main generation loop (batch-based, aligned with dParallel) ────
        pbar = tqdm(total=len(requests), desc="Generating...")

        for batch_idx in range(0, len(requests), self.batch_size):
            if batch_idx < processed_count:
                pbar.update(min(self.batch_size, len(requests) - batch_idx))
                continue

            sample_start_time = time.time()

            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])

            # Generate — contexts are already chat-templated by the framework
            if self.multi_block:
                responses, nfe = self._generate_batch_multi_block(contexts)
            else:
                responses, nfe = self._generate_batch(contexts)
            num_nfe += nfe

            sample_end_time = time.time()

            for i, r in enumerate(responses):
                # Count tokens before truncation
                generated_answer_ids = self.tokenizer.encode(r)
                num_tokens += len(generated_answer_ids)

                # Truncate at stop tokens
                for s in gen_args[i]['until']:
                    r = r.split(s)[0]
                responses[i] = r

            output.extend(responses)

            if self.rank == 0:
                print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")
                print("nfe step:", nfe)

            # Save checkpoint
            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    for r in responses:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Log per-sample stats
            if log_fh is not None:
                for i, r in enumerate(responses):
                    record = {
                        "sample_index": int(batch_idx + i),
                        "generated_tokens": int(len(self.tokenizer.encode(r))),
                        "steps": int(self.diffusion_steps),
                        "nfe": int(nfe),
                        "latency_seconds": float(sample_end_time - sample_start_time),
                        "timestamp": float(sample_end_time),
                    }
                    log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            pbar.update(len(batch_requests))

        pbar.close()

        if log_fh is not None:
            log_fh.close()

        total_time = time.time() - start_time

        # ── Per-rank final stats ──────────────────────────────────────────
        if self.stats_dir is not None:
            processed_samples = int(len(output))
            avg_iters_per_sample = (
                (float(num_nfe) / float(processed_samples)) if processed_samples > 0 else 0.0
            )
            avg_gen_len = (
                (float(num_tokens) / float(processed_samples)) if processed_samples > 0 else 0.0
            )
            avg_latency_per_sample = (
                (float(total_time) / float(processed_samples)) if processed_samples > 0 else 0.0
            )

            final_stats = {
                "processed_samples": processed_samples,
                "total_samples": int(len(requests)),
                "total_tokens": int(num_tokens),
                "total_nfe": int(num_nfe),
                "total_time": float(total_time),
                "tokens_per_second": (
                    (float(num_tokens) / float(total_time)) if total_time > 0 else 0.0
                ),
                "nfe_per_token": (
                    (float(num_nfe) / float(num_tokens)) if num_tokens > 0 else 0.0
                ),
                "tokens_per_forward": (
                    (float(num_tokens) / float(num_nfe)) if num_nfe > 0 else 0.0
                ),
                "avg_iters_per_sample": avg_iters_per_sample,
                "avg_gen_len": avg_gen_len,
                "avg_latency_per_sample": avg_latency_per_sample,
                "diffusion_steps": int(self.diffusion_steps),
                "max_new_tokens": int(self.max_new_tokens),
                "timestamp": time.time(),
            }
            stats_path = os.path.join(self.stats_dir, f"rank_{self.rank}_final_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)

        if self.show_speed:
            print(f"Total time taken: {total_time} seconds")
            print(f"Total NFE is {num_nfe}")
            if num_tokens > 0:
                print(f"Tokens per second (TPS): {num_tokens / total_time:.2f}")
                print(f"Tokens per forward (TPF): {num_tokens / num_nfe:.4f}")
                print(f"NFE per token: {num_nfe / num_tokens:.4f}")
            if len(output) > 0:
                print(f"Avg latency per sample: {total_time / len(output):.2f}s")

        # ── Multi-GPU aggregation ─────────────────────────────────────────
        if self.accelerator is not None and getattr(self, "world_size", 1) > 1:
            local_stats = torch.tensor(
                [
                    float(len(output)),
                    float(num_tokens),
                    float(total_time),
                    float(num_nfe),
                ],
                dtype=torch.float64,
                device=self.device,
            )
            gathered_stats = self.accelerator.gather(local_stats)

            if self.accelerator.is_local_main_process:
                gathered_stats = gathered_stats.view(self.world_size, -1)
                total_samples_all = int(gathered_stats[:, 0].sum().item())
                total_tokens_all = int(gathered_stats[:, 1].sum().item())
                sum_time_all = gathered_stats[:, 2].sum().item()
                total_nfe_all = int(gathered_stats[:, 3].sum().item())

                overall_tps = (
                    (total_tokens_all / sum_time_all) if sum_time_all > 0 else 0.0
                )
                avg_iters_per_sample_all = (
                    (float(total_nfe_all) / float(total_samples_all))
                    if total_samples_all > 0
                    else 0.0
                )
                avg_gen_len_all = (
                    (float(total_tokens_all) / float(total_samples_all))
                    if total_samples_all > 0
                    else 0.0
                )
                overall_avg_latency = (
                    (sum_time_all / float(total_samples_all))
                    if total_samples_all > 0
                    else 0.0
                )

                if self.stats_dir is not None:
                    aggregated_stats = {
                        "total_processed_samples": total_samples_all,
                        "total_generated_tokens": total_tokens_all,
                        "total_wall_time": sum_time_all,
                        "overall_tokens_per_second": overall_tps,
                        "overall_nfe": total_nfe_all,
                        "overall_nfe_per_token": (
                            (float(total_nfe_all) / float(total_tokens_all))
                            if total_tokens_all > 0
                            else 0.0
                        ),
                        "overall_tokens_per_forward": (
                            (float(total_tokens_all) / float(total_nfe_all))
                            if total_nfe_all > 0
                            else 0.0
                        ),
                        "avg_iters_per_sample": avg_iters_per_sample_all,
                        "avg_gen_len": avg_gen_len_all,
                        "overall_avg_latency_per_sample": overall_avg_latency,
                        "diffusion_steps": int(self.diffusion_steps),
                        "max_new_tokens": int(self.max_new_tokens),
                        "timestamp": time.time(),
                    }
                    all_ranks_stats_path = os.path.join(
                        self.stats_dir, "all_ranks_final_stats.json"
                    )
                    with open(all_ranks_stats_path, "w", encoding="utf-8") as f:
                        json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)

        return output


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
