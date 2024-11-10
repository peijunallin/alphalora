#!/bin/bash

# Set the root data path
root_data_path=""

# Define data paths using the root path
data_paths=(
  "$root_data_path/datasets/glue_rte_all.hf/"
  "$root_data_path/datasets/glue_mrpc_all.hf/"
  "$root_data_path/datasets/glue_cola_all.hf/"
  "$root_data_path/datasets/qa_text_scienceq_all.hf/"
  "$root_data_path/datasets/qa_commonq_all.hf/"
  "$root_data_path/datasets/qa_openbook_all.hf/"
)

# Loop through data paths and run experiments
for data_path in "${data_paths[@]}"; do
  # Extract filename from data path
  filename=$(basename "$data_path")
  # Remove trailing slash if present
  filename=${filename%/}
  filename=${filename%.hf}
  echo $filename

  # Run experiment
  CUDA_VISIBLE_DEVICES=0,1,2 python mola_training_mistral.py \
  --base_model "mistralai/Mistral-7B-v0.1" \
  --data_path "$data_path" \
  --output_dir "$root_data_path/mistral_alpha_$filename" \
  --batch_size 128 \
  --micro_batch_size 8 \
  --num_epochs 20 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 1 \
  --lora_r "8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8" \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
  --number_experts "1,3,5,4,4,5,4,4,3,3,2,2,2,2,3,4,9,4,8,7,8,7,8,7,9,8,6,10,6,7,3,2" \
  --top_k "1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
  --train_on_inputs \
  --group_by_length \
  --add_eos_token
done