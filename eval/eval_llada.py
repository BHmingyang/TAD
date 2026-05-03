import os
# os.environ["ALL_PROXY"] = "socks5h://127.0.0.1:13659"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from model.modeling_llada import LLaDAModelLM
import json
import time
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datetime import timedelta


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def get_transfer_index_entropy(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    entropy_threshold=None,
    num_transfer_tokens=None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    p = F.softmax(logits.to(torch.float64), dim=-1)
    if remasking == "low_confidence":
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
    elif remasking == "random":
        entropy = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    x0 = torch.where(mask_index, x0, x)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if entropy_threshold is not None:
        transfer_index = entropy_for_selection < entropy_threshold
        for j in range(entropy_for_selection.shape[0]):
            if mask_index[j].sum() > 0 and transfer_index[j].sum() == 0:
                min_index = torch.argmin(entropy_for_selection[j])
                transfer_index[j, min_index] = True
    else:
        for j in range(entropy_for_selection.shape[0]):
            _, select_index = torch.topk(entropy_for_selection[j], k=num_transfer_tokens[j], largest=False)
            transfer_index[j, select_index] = True
    return x0, transfer_index


def _check_early_stop(x, prompt_length, eos_token_id, mask_id):
    """
    Check for EOS token in the generation region.
    If found, set all tokens after the first EOS to EOS (not mask) and
    return (True, first_eos_absolute_position).
    Otherwise return (False, None).
    """
    if eos_token_id is None:
        return False, None
    gen_region = x[:, prompt_length:]
    eos_mask = (gen_region == eos_token_id) & (gen_region != mask_id)
    if not eos_mask.any():
        return False, None
    pos = torch.arange(gen_region.shape[1], device=x.device).unsqueeze(0)
    first_eos_rel = torch.where(eos_mask, pos, gen_region.shape[1]).amin(dim=1)
    first_eos_abs = prompt_length + first_eos_rel[0].item()
    # Set everything after first EOS to EOS (not mask)
    x[:, first_eos_abs + 1:] = eos_token_id
    return True, first_eos_abs


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    eos_token_id=None,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    nfe = 0
    for num_block in range(num_blocks):
        # Early stop: skip remaining blocks if EOS already found
        if eos_token_id is not None:
            has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
            if has_eos:
                break
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0
            x0, transfer_index = get_transfer_index_entropy(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                entropy_threshold=threshold,
                num_transfer_tokens=num_transfer_tokens[:, i] if threshold is None else None,
            )
            x[transfer_index] = x0[transfer_index]
            i += 1
            # Early stop: check after each decode step
            if eos_token_id is not None:
                has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
                if has_eos:
                    break
            if (
                x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length] == mask_id
            ).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    eos_token_id=None,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    nfe = 0
    for num_block in range(num_blocks):
        # Early stop: skip remaining blocks if EOS already found
        if eos_token_id is not None:
            has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
            if has_eos:
                break
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = x == mask_id
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index_entropy(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            entropy_threshold=threshold,
            num_transfer_tokens=num_transfer_tokens[:, 0] if threshold is None else None,
        )
        x[transfer_index] = x0[transfer_index]
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        past_key_values = new_past_key_values
        nfe += 1
        i = 1
        while True:
            nfe += 1
            mask_index = x[:, current_block_start:] == mask_id
            mask_index[:, block_length:] = 0
            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
            x0, transfer_index = get_transfer_index_entropy(
                logits,
                temperature,
                remasking,
                mask_index,
                x[:, current_block_start:],
                entropy_threshold=threshold,
                num_transfer_tokens=num_transfer_tokens[:, i] if threshold is None else None,
            )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            # Early stop: check after each decode step
            if eos_token_id is not None:
                has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
                if has_eos:
                    break
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1
    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    eos_token_id=None,
):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    nfe = 0
    for num_block in range(num_blocks):
        # Early stop: skip remaining blocks if EOS already found
        if eos_token_id is not None:
            has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
            if has_eos:
                break
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        block_mask_index = x[:, current_block_start:current_block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = x == mask_id
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index_entropy(
            output.logits,
            temperature,
            remasking,
            mask_index,
            x,
            entropy_threshold=threshold,
            num_transfer_tokens=num_transfer_tokens[:, 0] if threshold is None else None,
        )
        x[transfer_index] = x0[transfer_index]
        nfe += 1
        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            nfe += 1
            mask_index = x[:, current_block_start:current_block_end] == mask_id
            logits = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            ).logits
            x0, transfer_index = get_transfer_index_entropy(
                logits,
                temperature,
                remasking,
                mask_index,
                x[:, current_block_start:current_block_end],
                entropy_threshold=threshold,
                num_transfer_tokens=num_transfer_tokens[:, i] if threshold is None else None,
            )
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            # Early stop: check after each decode step
            if eos_token_id is not None:
                has_eos, _ = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
                if has_eos:
                    break
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1
    return x, nfe


@torch.no_grad()
def generate_multi_block(
    model,
    prompt,
    steps=128,
    max_new_tokens=512,
    block_size=32,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=0.5,
    block_add_threshold=0.5,
    decoded_token_threshold=0.5,
    eos_token_id=None,
):
    """
    Pipelined parallel decoding without cache.

    Args:
        block_add_threshold: Add new block when last block progress >= this threshold.
                            Set to 1.0 for fully sequential processing (like generate()).
        decoded_token_threshold: Block becomes fully activated when previous block progress >= this threshold.
                                Set to 1.0 for fully sequential processing (like generate()).
        threshold: Entropy threshold for decoding (lower entropy = higher confidence).
                   Tokens with entropy > threshold are skipped.
                   Same semantics as in generate() method. Typical value: 0.5

    When block_add_threshold=1.0 and decoded_token_threshold=1.0, this method behaves
    identically to generate() with sequential block processing.
    """
    x = torch.full((1, prompt.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]

    # Track block states: {block_id: {start, end, mask_count, total_masks, is_complete}}
    # Initialize with prompt block
    block_states = {
        0: {
            "start": 0,
            "end": prompt.shape[1],
            "mask_count": 0,
            "total_masks": prompt.shape[1],
            "is_complete": True,
        }
    }

    # Create first generation block
    num_blocks = max_new_tokens // block_size
    next_block_id = 1
    if next_block_id <= num_blocks:
        block_start = prompt.shape[1] + (next_block_id - 1) * block_size
        block_end = min(block_start + block_size, prompt.shape[1] + max_new_tokens)
        # First block should be immediately activated since prompt (block 0) is already complete
        should_activate = 1.0 >= decoded_token_threshold  # prompt progress is always 1.0
        block_states[next_block_id] = {
            "start": block_start,
            "end": block_end,
            "mask_count": block_end - block_start,
            "total_masks": block_end - block_start,
            "is_complete": should_activate,
        }
        next_block_id += 1

    nfe = 0

    while True:
        # Check if all blocks are exhausted AND no more blocks to create
        mask_index = x == mask_id
        total_masks = mask_index[:, prompt_length:].sum()

        if total_masks == 0 and next_block_id > num_blocks:
            break

        nfe += 1

        # Early stop: check for EOS token
        if eos_token_id is not None:
            has_eos, first_eos_abs = _check_early_stop(x, prompt_length, eos_token_id, mask_id)
            if has_eos:
                # Create all remaining blocks after EOS and mark them as complete
                while next_block_id <= num_blocks:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_start > first_eos_abs:
                        block_states[next_block_id] = {
                            "start": block_start, "end": block_end, "mask_count": 0,
                            "total_masks": block_end - block_start, "is_complete": True,
                        }
                        next_block_id += 1
                    else:
                        break
                # Recalculate: if no masks remain, we're done
                if (x == mask_id)[:, prompt_length:].sum() == 0:
                    break

        # Update block completion states
        def update_block_activation_states():
            """Update which blocks should be fully activated based on previous block progress."""
            for bid in sorted(block_states.keys()):
                if bid > 0 and not block_states[bid]["is_complete"]:
                    prev_progress = (
                        1
                        - block_states[bid - 1]["mask_count"]
                        / block_states[bid - 1]["total_masks"]
                    )
                    if prev_progress >= decoded_token_threshold:
                        block_states[bid]["is_complete"] = True

        update_block_activation_states()

        # Add new block dynamically based on last block's progress
        if next_block_id <= num_blocks:
            last_bid = max(block_states.keys())
            if last_bid > 0:  # Not just prompt
                last_progress = (
                    1
                    - block_states[last_bid]["mask_count"]
                    / block_states[last_bid]["total_masks"]
                )
                # Create next block when:
                # 1. Last block progress >= block_add_threshold (for parallel processing), OR
                # 2. Last block is complete (mask_count == 0) for sequential processing
                should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)

                if should_add_block:
                    # Add next block
                    block_start = prompt.shape[1] + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt.shape[1] + max_new_tokens)
                    if block_end > block_start:
                        # Check how many positions in this block are actually masked
                        actual_mask_count = (x[:, block_start:block_end] == mask_id).sum().item()

                        # Determine if this block should be immediately activated
                        # Check if previous block is complete enough
                        prev_bid = next_block_id - 1
                        prev_progress = (
                            1 - block_states[prev_bid]["mask_count"] / block_states[prev_bid]["total_masks"]
                        )
                        should_activate = prev_progress >= decoded_token_threshold

                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": actual_mask_count,
                            "total_masks": block_end - block_start,
                            "is_complete": should_activate,
                        }
                        next_block_id += 1

        # Forward pass: only process up to the last complete or semi-activated block
        # Find the rightmost block that should be processed
        rightmost_active_bid = 0
        for bid in sorted(block_states.keys()):
            if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                rightmost_active_bid = bid

        if rightmost_active_bid == 0:
            break

        active_end = block_states[rightmost_active_bid]["end"]

        # Always do forward pass on entire sequence (like generate() does)
        logits = model(x).logits

        # Mask out future blocks (positions after active_end) to prevent them from being decoded
        mask_index_for_decode = mask_index.clone()
        mask_index_for_decode[:, active_end:] = 0

        # Decode using full-sequence approach (like generate())
        x0, transfer_index = get_transfer_index_entropy(
            logits,
            temperature,
            remasking,
            mask_index_for_decode,
            x,
            entropy_threshold=threshold if threshold is not None else 999.0,
            num_transfer_tokens=None,
        )

        # For fully activated blocks, ensure at least one token is decoded (guaranteed progress)
        # Find the first fully activated block with masks
        first_fully_activated_bid = None
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                first_fully_activated_bid = bid
                break

        if first_fully_activated_bid is not None:
            # Check if any token was decoded in this block
            start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
            block_transfer = transfer_index[:, start:end]

            if not block_transfer.any():
                # Force decode the lowest entropy token in this fully activated block
                p = F.softmax(logits[:, start:end].to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                block_mask = mask_index_for_decode[:, start:end]
                entropy = torch.where(block_mask, entropy, torch.inf)
                best_idx = entropy[0].argmin()
                transfer_index[0, start + best_idx] = True
                x0_resample = torch.argmax(logits[0, start + best_idx], dim=-1)
                x0[0, start + best_idx] = x0_resample

        # Apply the decoded tokens
        x[transfer_index] = x0[transfer_index]

        # Update block states based on which positions were decoded
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["mask_count"] > 0:
                start, end = block_states[bid]["start"], block_states[bid]["end"]
                block_decoded = transfer_index[:, start:end].sum().item()
                if block_decoded > 0:
                    block_states[bid]["mask_count"] -= block_decoded

        if nfe > 10000:
            break

    return x, nfe


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path="/home/u2025104115/LLada_sft/SAFT-LLaDA/checkpoint/save_checkpoints/SAFT-LLaDA-5k_data",
        mask_id=126336,
        max_length=1024,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        use_cache=False,
        threshold=None,
        save_dir=None,
        stats_dir=None,
        show_speed=False,
        dual_cache=False,
        multi_block=False,
        block_add_threshold=0.5,
        decoded_token_threshold=0.5,
        early_stop=False,
        task="null",
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
        if hasattr(config, "flash_attention"):
            config.flash_attention = True
        self.model = LLaDAModelLM.from_pretrained(
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
        self.mask_id = mask_id
        if hasattr(config, "mask_token_id") and config.mask_token_id is not None:
            self.mask_id = config.mask_token_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        # self.is_instruct = True if ("instruct" in model_path.lower() or "1.5" in model_path.lower()) else False
        self.is_instruct = True
        self.save_dir = save_dir
        self.stats_dir = stats_dir if stats_dir is not None else save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.multi_block = multi_block
        self.block_add_threshold = block_add_threshold
        self.decoded_token_threshold = decoded_token_threshold
        self.early_stop = early_stop
        self.task = task
        self.cfg = 0

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
        logits = self.model(batch).logits
        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none") / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix
        for _ in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
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
        assert max(prompt_len) <= 4096
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

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = []
        num_tokens = 0
        num_nfe = 0
        processed_count = 0
        start_time = time.time()
        log_fh = None
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
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            sample_start_time = time.time()
            if i < processed_count:
                continue
            question = req.args[0]
            if self.is_instruct:
                tail = r" Please reason step by step, and put your final answer within \boxed{}."
                if self.task == "gsm8k":
                    m = [{"role": "user", "content": question + tail}]
                elif self.task == "humaneval" or self.task == "humaneval_plus":
                    start = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{{ prompt }}\n```\n "
                    question = start.replace("{{ prompt }}", question)
                    m = [{"role": "user", "content": question}]
                else:
                    m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)["input_ids"]
            else:
                user_input = question
                input_ids = self.tokenizer(user_input)["input_ids"]
            stop_tokens = list(req.args[1]["until"])
            if "<|eot_id|>" not in stop_tokens:
                stop_tokens.append("<|eot_id|>")
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            # Determine eos_token_id for early stopping
            eos_token_id = self.tokenizer.eos_token_id if self.early_stop else None
            if self.multi_block:
                generated_answer, nfe = generate_multi_block(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    max_new_tokens=self.gen_length,
                    block_size=self.block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    block_add_threshold=self.block_add_threshold,
                    decoded_token_threshold=self.decoded_token_threshold,
                    eos_token_id=eos_token_id,
                )
            elif self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                        eos_token_id=eos_token_id,
                    )
                else:
                    generated_answer, nfe = generate_with_prefix_cache(
                        self.model,
                        input_ids,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=0,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        threshold=self.threshold,
                        eos_token_id=eos_token_id,
                    )
            else:
                generated_answer, nfe = generate(
                    self.model,
                    input_ids,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    threshold=self.threshold,
                    eos_token_id=eos_token_id,
                )
            if self.is_instruct and "task_id" in req.doc and str(req.doc["task_id"]).lower().startswith("humaneval"):
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1] :], skip_special_tokens=True)
                generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            else:
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1] :], skip_special_tokens=False)
                print(f"stop_tokens: {stop_tokens}")
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]
                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            num_tokens += len(generated_answer_ids)
            num_nfe += nfe
            output.append(generated_answer)
            if self.save_dir is not None:
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(generated_answer, ensure_ascii=False) + "\n")
            sample_end_time = time.time()
            if log_fh is not None:
                record = {
                    "sample_index": int(i),
                    "prompt_tokens": int(input_ids.shape[1]),
                    "generated_tokens": int(len(generated_answer_ids)),
                    "steps": int(self.steps),
                    "nfe": int(nfe),
                    "latency_seconds": float(sample_end_time - sample_start_time),
                    "timestamp": float(sample_end_time),
                }
                log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("=" * 20)
            print("question: ", question)
            print("answer: ", generated_answer)
            print("=" * 20, end="\n\n")
        if log_fh is not None:
            log_fh.close()
        total_time = time.time() - start_time
        # Compute blocks per sample for logging
        blocks_per_sample = int(self.gen_length // self.block_length) if getattr(self, "block_length", 0) > 0 else 0
        if self.stats_dir is not None:
            processed_samples = int(len(output))
            avg_iters_per_sample = (float(num_nfe) / float(processed_samples)) if processed_samples > 0 else 0.0
            avg_gen_len = (float(num_tokens) / float(processed_samples)) if processed_samples > 0 else 0.0
            avg_gen_len_blocks_persample = (float(self.gen_length) / float(self.block_length)) if getattr(self, "block_length", 0) > 0 else 0.0
            final_stats = {
                "processed_samples": processed_samples,
                "total_samples": int(len(requests)),
                "total_tokens": int(num_tokens),
                "total_nfe": int(num_nfe),
                "total_time": float(total_time),
                "tokens_per_second": (float(num_tokens) / float(total_time)) if total_time > 0 else 0.0,
                "nfe_per_token": (float(num_nfe) / float(num_tokens)) if num_tokens > 0 else 0.0,
                "tokens_per_forward": (float(num_tokens)) / (float(num_nfe)),
                "blocks_per_sample": blocks_per_sample,
                "gen_length": int(self.gen_length),
                "block_length": int(self.block_length),
                "avg_iters_per_sample": avg_iters_per_sample,
                "avg_gen_len": avg_gen_len,
                "avg_gen_len_blocks_persample": avg_gen_len_blocks_persample,
                "timestamp": time.time(),
            }
            stats_path = os.path.join(self.stats_dir, f"rank_{self.rank}_final_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(final_stats, f, ensure_ascii=False, indent=2)
        if self.show_speed:
            print(f"Total time taken: {total_time} seconds")
            print(f"Total NFE is {num_nfe}")
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
                overall_tps = (total_tokens_all / sum_time_all) if sum_time_all > 0 else 0.0
                avg_iters_per_sample_all = (float(total_nfe_all) / float(total_samples_all)) if total_samples_all > 0 else 0.0
                avg_gen_len_all = (float(total_tokens_all) / float(total_samples_all)) if total_samples_all > 0 else 0.0
                if self.stats_dir is not None:
                    aggregated_stats = {
                        "total_processed_samples": total_samples_all,
                        "total_generated_tokens": total_tokens_all,
                        "total_wall_time": sum_time_all,
                        "overall_tokens_per_second": overall_tps,
                        "overall_nfe": total_nfe_all,
                        "overall_nfe_per_token": (float(total_nfe_all) / float(total_tokens_all)) if total_tokens_all > 0 else 0.0,
                        "overall_tokens_per_forward": (float(total_tokens_all)) / (float(total_nfe_all)),
                        "avg_iters_per_sample": avg_iters_per_sample_all,
                        "avg_gen_len": avg_gen_len_all,
                        "blocks_per_sample": blocks_per_sample,
                        "gen_length": int(self.gen_length),
                        "block_length": int(self.block_length),
                        "timestamp": time.time(),
                    }
                    all_ranks_stats_path = os.path.join(self.stats_dir, "all_ranks_final_stats.json")
                    with open(all_ranks_stats_path, "w", encoding="utf-8") as f:
                        json.dump(aggregated_stats, f, ensure_ascii=False, indent=2)
        return output


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
