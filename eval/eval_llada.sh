export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ENDPOINT="https://hf-mirror.com"


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=0
steps=256
threshold=0.5
model_path="TAD-LLaDA Path"

# TAD-LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},threshold=${threshold},multi_block=True,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/gsm8k-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock \
--output_path evals_results/TAD-LLaDA/gsm8k-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock --log_samples \


# TAD-LLaDA-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/TPF1-gsm8k-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-LLaDA/TPF1-gsm8k-ns${num_fewshot}-${length}-${block_length} --log_samples \


############################################### math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256

# TAD-LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29603 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},multi_block=True,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/math-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock \
--output_path evals_results/TAD-LLaDA/math-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock --log_samples \



# TAD-LLaDA-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29603 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/TPF1-math-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-LLaDA/TPF1-math-ns${num_fewshot}-${length}-${block_length} --log_samples \


############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=256

# TAD-LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29602 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},multi_block=True,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/humaneval-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock \
--output_path evals_results/TAD-LLaDA/humaneval-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock --log_samples \


# TAD-LLaDA-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29602 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/TPF1-humaneval-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-LLaDA/TPF1-humaneval-ns${num_fewshot}-${length}-${block_length} --log_samples \


############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
threshold=0.5

# TAD-LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29604 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},multi_block=True,,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/mbpp-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock \
--output_path evals_results/TAD-LLaDA/mbpp-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-multiblock --log_samples \


# TAD-LLaDA-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29604 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-LLaDA/TPF1-mbpp-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-LLaDA/TPF1-mbpp-ns${num_fewshot}-${length}-${block_length} --log_samples \



