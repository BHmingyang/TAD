# TAD: Temporal-Aware Trajectory Self-Distillation for Fast and Accurate  Diffusion LLM


## 1. Environment Setup

```bash
conda create -n tad python=3.10 -y
conda activate tad

pip install -r requirements.txt
```

## 2. Data Preparation

### 2.1 Generate Math Trajectories

**For LLaDA:**

```bash
python prepare/llada_gen_math_traj.py \
    --model_name <your_llada_model_path> \
    --output_path data/llada_math_traj.jsonl \
    --dataset_name gsm8k \
    --dataset_split train \
    --max_new_tokens 256 \
    --steps 256 \
    --block_length 32 \
    --num_samples 1 \
    --limit 0
```

**For Dream:**

```bash
python prepare/dream_gen_math_traj.py \
    --model_name <your_dream_model_path> \
    --output_path data/dream_math_traj.jsonl \
    --dataset_name gsm8k \
    --dataset_split train \
    --max_new_tokens 256 \
    --block_length 32 \
    --top_p 0.95 \
    --alg entropy \
    --num_samples 1 \
    --limit 0
```

### 2.2 Generate Code Trajectories

**For LLaDA:**

```bash
python prepare/llada_gen_code_traj.py \
    --model_name <your_llada_model_path> \
    --parquet_path <path_to_kodcode_parquet> \
    --output_path data/llada_code_traj.jsonl \
    --max_new_tokens 256 \
    --steps 256 \
    --block_length 32 \
    --num_samples 1 \
```

**For Dream:**

```bash
python prepare/dream_gen_code_traj.py \
    --model_name <your_dream_model_path> \
    --parquet_path <path_to_kodcode_parquet> \
    --output_path data/dream_code_traj.jsonl \
    --max_new_tokens 256 \
    --block_length 32 \
    --top_p 0.95 \
    --alg entropy \
    --num_samples 1 \
```

## 3. Training

### 3.1 Configure

Edit the config file to set your model path and training hyperparameters:

- **LLaDA**: `LLaDA/configs/config_llada.yaml`
- **Dream**: `Dream/configs/config_dream.yaml`


### 3.2 Launch Training

**Train LLaDA:**

```bash
cd LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_llada.py
```

**Train Dream:**

```bash
cd Dream
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed train_dream.py
```

## 4. Evaluation

### 4.1 Merge LoRA Weights

After training, merge the LoRA adapter into the base model. Edit the paths in `merge_lora.py` and run:

**For LLaDA:**

```bash
cd LLaDA
# Edit merge_lora.py:
#   - name = "<your_base_model_path>"
#   - PeftModel.from_pretrained(base_model, "<your_checkpoint_path>")
#   - merged_model.save_pretrained("<save_path>")
python merge_lora.py
```


### 4.2 Run Evaluation

Edit the `model_path` variable in the evaluation script to point to your merged model, then run:

**Evaluate LLaDA:**

```bash
cd eval
# Edit eval_llada.sh: set model_path="<your_merged_model_path>"
bash eval_llada.sh
```


The evaluation scripts cover the following benchmarks:
- **Math**: GSM8K, MATH (Minerva)
- **Code**: HumanEval, MBPP

Results will be saved under `evals_results/`.
