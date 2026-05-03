export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ENDPOINT="https://hf-mirror.com"


############################################### gsm8k evaluations ###############################################
task=gsm8k_cot_zeroshot
length=256
block_length=32
num_fewshot=0
steps=256
threshold=0.4
block_add_threshold=0.1
decoded_token_threshold=0.95
model_path="Your Model Path"

# # TAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29601 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},diffusion_steps=${steps},multi_block=True,block_add_threshold=${block_add_threshold},decoded_token_threshold=${decoded_token_threshold},alg=entropy_threshold,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/gsm8k-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} \
--output_path evals_results/TAD-Dream/gsm8k-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} --log_samples \
--apply_chat_template

# TAD-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29601 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},diffusion_steps=${steps},alg=entropy_threshold,threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/TPF1-gsm8k-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-Dream/TPF1-gsm8k-ns${num_fewshot}-${length}-${block_length} --log_samples \
--apply_chat_template

############################################### math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256

# TAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29603 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},multi_block=True,block_add_threshold=${block_add_threshold},decoded_token_threshold=${decoded_token_threshold},diffusion_steps=${steps},alg=entropy_threshold,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/math-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} \
--output_path evals_results/TAD-Dream/math-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} --log_samples \
--apply_chat_template


# TAD-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29603 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},diffusion_steps=${steps},alg=entropy_threshold,threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/TPF1-math-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-Dream/TPF1-math-ns${num_fewshot}-${length}-${block_length} --log_samples \
--apply_chat_template

############################################### humaneval evaluations ###############################################
task=humaneval_instruct
length=256
block_length=32
num_fewshot=0
steps=256

# TAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29602 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},multi_block=True,block_add_threshold=${block_add_threshold},decoded_token_threshold=${decoded_token_threshold},diffusion_steps=${steps},alg=entropy_threshold,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/humaneval-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} \
--output_path evals_results/TAD-Dream/humaneval-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} --log_samples \
--apply_chat_template

# TAD-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29602 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},diffusion_steps=${steps},alg=entropy_threshold,threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/TPF1-humaneval-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-Dream/TPF1-humaneval-ns${num_fewshot}-${length}-${block_length} --log_samples \
--apply_chat_template

############################################### mbpp evaluations ###############################################
task=mbpp_instruct
length=256
block_length=32
num_fewshot=0
steps=256

# TAD
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29604 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},multi_block=True,block_add_threshold=${block_add_threshold},decoded_token_threshold=${decoded_token_threshold},diffusion_steps=${steps},alg=entropy_threshold,threshold=${threshold},block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/mbpp-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} \
--output_path evals_results/TAD-Dream/mbpp-ns${num_fewshot}-${length}-${block_length}-threshold${threshold}-MB-block_add_${block_add_threshold}-decoded_token_${decoded_token_threshold} --log_samples \
--apply_chat_template

# TAD-TPF1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m accelerate.commands.launch --main_process_port 29604 eval_dream.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 10000 \
--confirm_run_unsafe_code --model dream_dist \
--model_args model_path=${model_path},max_new_tokens=${length},diffusion_steps=${steps},alg=entropy_threshold,threshold=0,block_length=${block_length},show_speed=True,task=${task},save_dir=evals_results/TAD-Dream/TPF1-mbpp-ns${num_fewshot}-${length}-${block_length} \
--output_path evals_results/TAD-Dream/TPF1-mbpp-ns${num_fewshot}-${length}-${block_length} --log_samples \
--apply_chat_template
