import json
import os
import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import LoraConfig, get_peft_model


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Creating a DeepSpeed Configuration"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }

def prepare_models(config: Dict[str, Any]):
    """Prepare Student and Frozen Teacher models"""
    torch_dtype = config['model']['torch_dtype']
    model_name = config['model']['name']
    trust_remote_code = config['model']['trust_remote_code']
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load base model as Student
    student_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    
    # 2. Deep copy base model as Teacher and freeze parameters
    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # 3. Inject LoRA into Student model
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    return student_model, teacher_model, tokenizer

class TrajectoryDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, delta: int = 4, mask_token_id: int = 151666, sample_ratio: float = 1.0, shuffle: bool = True, seed: int = 42):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.delta = delta  
        self.sample_ratio = max(0.0, min(1.0, float(sample_ratio)))
        self.shuffle = shuffle
        self.seed = seed
        
        self.data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        self.step_keys = [None] * len(self.data)
        self.index_map = []
        for record_idx, record in enumerate(self.data):
            trajectory = record.get("trajectory", {})
            # step0 (all Mask) -> stepN (Final answer)
            step_keys = sorted(trajectory.keys(), key=self._step_key_to_idx)
            if len(step_keys) < 2:
                continue
            self.step_keys[record_idx] = step_keys
            # Exclude steps that cannot perform delta prediction
            for step_idx in range(len(step_keys) - 1):
                self.index_map.append((record_idx, step_idx))
        if self.sample_ratio < 1.0:
            target_size = int(len(self.index_map) * self.sample_ratio)
            if target_size < len(self.index_map):
                self.index_map = random.sample(self.index_map, target_size)

        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    @staticmethod
    def _step_key_to_idx(key: str) -> int:
        digits = "".join([c for c in key if c.isdigit()])
        return int(digits) if digits else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record_idx, step_idx = self.index_map[idx]
        record = self.data[record_idx]
        prompt = record["prompt"]
        trajectory = record["trajectory"]
        answer_gt = record.get("groundtruth", "") 
        
        step_keys = self.step_keys[record_idx]
        
        # Determine skip-step target (later steps have fewer masks)
        target_idx = min(step_idx + self.delta, len(step_keys) - 1)
        
        x_t = trajectory[step_keys[step_idx]]
        x_target = trajectory[step_keys[target_idx]]
        
        # 1. Build Student (blind guess) input
        student_prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        student_input_ids = student_prompt_ids + x_t
        
        # 2. Teacher Inputs
        # Prepend Reference Answer after the first system message
        # Find the end of the first system message (<|im_end|>) and insert Reference Answer before it
        if prompt.startswith("<|im_start|>system") and "<|im_end|>" in prompt:
            first_end = prompt.find("<|im_end|>")
            teacher_prompt_text = (
                prompt[:first_end] +
                f"\n\nReference Answer: {answer_gt}" +
                prompt[first_end:]
            )
        else:
            # If prompt format is unexpected, prepend a system message with Reference Answer
            teacher_prompt_text = f"<|im_start|>system\n\nReference Answer: {answer_gt}<|im_end|>\n" + prompt
            
        teacher_prompt_ids = self.tokenizer(teacher_prompt_text, add_special_tokens=False).input_ids
        teacher_input_ids = teacher_prompt_ids + x_t
        
        # 3. Build hard labels (CE Target) and separate masks
        # Only provide labels for the answer part; prompt part uses -100 to ignore

        labels_tail = []
        ce_mask_tail = []   # CE region: currently MASK and Target is not MASK (has hard label)
        kl_mask_tail = []   # KL region: currently MASK and Target is still MASK (no hard label, needs Teacher guidance)
        
        for tk_t, tk_target in zip(x_t, x_target):
            if tk_t == self.mask_token_id and tk_target != self.mask_token_id:
                # CE region: currently MASK, Target revealed -> train with hard label
                labels_tail.append(tk_target)
                ce_mask_tail.append(1.0)
                kl_mask_tail.append(0.0)
            elif tk_t == self.mask_token_id and tk_target == self.mask_token_id:
                # KL region: currently MASK, Target still MASK -> use Teacher soft label
                labels_tail.append(-100)
                ce_mask_tail.append(0.0)
                kl_mask_tail.append(1.0)
            else:
                # Non-MASK position: not involved in any loss computation
                labels_tail.append(-100)
                ce_mask_tail.append(0.0)
                kl_mask_tail.append(0.0)
                
        # Concatenate prompt part (prompt part does not contribute to loss)
        labels = [-100] * len(student_prompt_ids) + labels_tail
        student_ce_mask = [0.0] * len(student_prompt_ids) + ce_mask_tail
        student_kl_mask = [0.0] * len(student_prompt_ids) + kl_mask_tail
        teacher_kl_mask = [0.0] * len(teacher_prompt_ids) + kl_mask_tail

        return {
            "student_input_ids": torch.tensor(student_input_ids, dtype=torch.long),
            "teacher_input_ids": torch.tensor(teacher_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "student_ce_mask": torch.tensor(student_ce_mask, dtype=torch.float),
            "student_kl_mask": torch.tensor(student_kl_mask, dtype=torch.float),
            "teacher_kl_mask": torch.tensor(teacher_kl_mask, dtype=torch.float),
        }

def build_collator(tokenizer: AutoTokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        student_input_ids = [item["student_input_ids"] for item in batch]
        teacher_input_ids = [item["teacher_input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        student_ce_mask = [item["student_ce_mask"] for item in batch]
        student_kl_mask = [item["student_kl_mask"] for item in batch]
        teacher_kl_mask = [item["teacher_kl_mask"] for item in batch]

        from torch.nn.utils.rnn import pad_sequence
        
        # Right Padding
        student_batch = pad_sequence(student_input_ids, batch_first=True, padding_value=pad_id)
        teacher_batch = pad_sequence(teacher_input_ids, batch_first=True, padding_value=pad_id)
        labels_batch = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        student_ce_mask_batch = pad_sequence(student_ce_mask, batch_first=True, padding_value=0.0)
        student_kl_mask_batch = pad_sequence(student_kl_mask, batch_first=True, padding_value=0.0)
        teacher_kl_mask_batch = pad_sequence(teacher_kl_mask, batch_first=True, padding_value=0.0)
        
        # Dream model uses bidirectional attention (is_causal=False), all tokens attend to each other without attention_mask
        # Padding positions do not affect loss (labels and loss_mask are correctly handled)

        return {
            "student_input_ids": student_batch,
            "teacher_input_ids": teacher_batch,
            "labels": labels_batch,
            "student_ce_mask": student_ce_mask_batch,
            "student_kl_mask": student_kl_mask_batch,
            "teacher_kl_mask": teacher_kl_mask_batch,
        }

    return collate_fn


class DLMTrainer(Trainer):
    def __init__(self, teacher_model, lambda_ce=1.0, lambda_kl=1.0, tau=1.0, mask_token_id=151666, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.lambda_ce = lambda_ce
        self.lambda_kl = lambda_kl
        self.tau = tau
        self.mask_token_id = mask_token_id

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        Override create_scheduler to fix LR scheduler param group mismatch with DeepSpeed + LoRA.
        """
        import torch
        from torch.optim.lr_scheduler import LambdaLR
        
        optimizer = self.optimizer if optimizer is None else optimizer
        
        # Get number of param groups
        num_param_groups = len(optimizer.param_groups)
        
        # Create a constant LR scheduler
        def lr_lambda(current_step: int):
            return 1.0
        
        # Create the same LR schedule for each param group
        lr_scheduler = LambdaLR(optimizer, [lr_lambda] * num_param_groups)
        
        self.lr_scheduler = lr_scheduler
        return lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if next(self.teacher_model.parameters()).device != model.device:
            self.teacher_model = self.teacher_model.to(model.device)
            
        # ================== 1. Student Forward ==================
        student_outputs = model(
            input_ids=inputs["student_input_ids"],
        )
        student_logits = student_outputs.logits
        # Dream model requires logits shift: logits[i] predicts token[i]
        # Official approach: logits = cat([logits[:,:1], logits[:,:-1]], dim=1)
        student_logits = torch.cat([student_logits[:, :1, :], student_logits[:, :-1, :]], dim=1)
        
        # ================== 2. Teacher Forward ==================
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["teacher_input_ids"],
            )
            teacher_logits = teacher_outputs.logits
            # Teacher also requires logits shift
            teacher_logits = torch.cat([teacher_logits[:, :1, :], teacher_logits[:, :-1, :]], dim=1)

        # ================== 3. Extract CE and KL region Logits ==================
        # CE region: currently MASK and Target revealed (has hard label)
        s_ce_bool = inputs["student_ce_mask"].bool()
        ce_student_logits = student_logits[s_ce_bool]
        ce_labels = inputs["labels"][s_ce_bool]

        # KL region: currently MASK and Target still MASK (no hard label, needs Teacher guidance)
        s_kl_bool = inputs["student_kl_mask"].bool()
        t_kl_bool = inputs["teacher_kl_mask"].bool()
        kl_student_logits = student_logits[s_kl_bool]
        kl_teacher_logits = teacher_logits[t_kl_bool]

        assert kl_student_logits.shape[0] == kl_teacher_logits.shape[0], \
            f"KL mask mismatch: student {kl_student_logits.shape[0]} vs teacher {kl_teacher_logits.shape[0]}"

        # ================== 4. Dual Loss (spatially exclusive) ==================
        # a. Trajectory CE Loss (only on mask tokens with hard labels)
        if ce_student_logits.numel() > 0 and (ce_labels != -100).any():
            loss_ce = F.cross_entropy(ce_student_logits, ce_labels, ignore_index=-100)
        else:
            loss_ce = torch.tensor(0.0, device=model.device)

        # b. Privileged KL Loss (only on mask tokens without hard labels, still masked)
        if kl_student_logits.numel() > 0:
            p_teacher = F.softmax(kl_teacher_logits / self.tau, dim=-1)
            log_p_student = F.log_softmax(kl_student_logits / self.tau, dim=-1)
            loss_kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (self.tau ** 2)
        else:
            loss_kl = torch.tensor(0.0, device=model.device)

        # total Loss
        if ce_student_logits.numel() == 0 and kl_student_logits.numel() == 0:
            loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        else:
            loss = self.lambda_ce * loss_ce + self.lambda_kl * loss_kl

        self.log({
            "loss_ce": float(loss_ce.detach()),
            "loss_kl": float(loss_kl.detach()),
            "ce_tokens": int(s_ce_bool.sum()),
            "kl_tokens": int(s_kl_bool.sum()),
        })

        return (loss, student_outputs) if return_outputs else loss


def main():
    config = load_config("configs/config_dream.yaml")

    training_args = TrainingArguments(
        **config['training'],
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False, 
    )
    
    # Save config to output_dir for reference
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    config_save_path = os.path.join(output_dir, "config_used.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"Config saved to: {config_save_path}")
    
    # Prepare student model and teacher model
    student_model, teacher_model, tokenizer = prepare_models(config)

    sample_ratio = config.get("data", {}).get("sample_ratio", 1.0)
    shuffle = config.get("data", {}).get("shuffle", True)
    seed = config.get("data", {}).get("seed", 42)
    train_dataset = TrajectoryDataset(
        data_path="/ossfs/workspace/dllm-inference-acceleration-main/data/dream_data.jsonl",
        tokenizer=tokenizer,
        delta=6, 
        mask_token_id=151666,
        sample_ratio=sample_ratio,
        shuffle=shuffle,
        seed=seed
    )

    trainer = DLMTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=build_collator(tokenizer),
        mask_token_id=151666,
        lambda_ce=1.0,
        lambda_kl=1.0,
        tau=1.0,
    )

    trainer.train()

if __name__ == "__main__":
    main()
