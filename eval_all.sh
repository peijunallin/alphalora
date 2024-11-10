#!/bin/bash

# Set base paths for data and weights

base_weights_path=""

#The weights path should be the same with the output_dir in run_all.sh
mola_weights=(
  "$base_weights_path/mistral_alpha_glue_rte_all"
  "$base_weights_path/mistral_alpha_glue_mrpc_all"
  "$base_weights_path/mistral_alpha_glue_cola_all"
  "$base_weights_path/mistral_alpha_qa_text_scienceq_all"
  "$base_weights_path/mistral_alpha_qa_commonq_all"
  "$base_weights_path/mistral_alpha_qa_openbook_all"

)

test_data=(
  "$base_weights_path/datasets/glue_rte_test.json"
  "$base_weights_path/datasets/glue_mrpc_test.json"
  "$base_weights_path/datasets/glue_cola_test.json"
  "$base_weights_path/datasets/qa_text_scienceq_test_all.json"
  "$base_weights_path/datasets/qa_commonq_test_all.json"
  "$base_weights_path/datasets/qa_openbook_test_all.json"
)

checkpoints=("checkpoint-10" "checkpoint-15" "checkpoint-20")

# Loop through the arrays and run evaluations
#

for (( i=0; i<${#mola_weights[@]}; i++ )); do
  for checkpoint in "${checkpoints[@]}"; do
    if [[ $checkpoint == "checkpoint-20" ]]; then
      checkpoint_dir="${mola_weights[i]}"
    else
      checkpoint_dir="${mola_weights[i]}/${checkpoint}"
    fi

    if [ -d "$checkpoint_dir" ]; then
      echo "Evaluating $(basename ${mola_weights[i]}) - $checkpoint"
      CUDA_VISIBLE_DEVICES=3 python evaluation.py \
        --test_dataset "${test_data[i]}" \
        --base_model "mistralai/Mistral-7B-v0.1" \
        --mola_weights "$checkpoint_dir" \
        --batch_size 8 \
        --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
        --number_experts "1,3,5,4,4,5,4,4,3,3,2,2,2,2,3,4,9,4,8,7,8,7,8,7,9,8,6,10,6,7,3,2" \
        --top_k "1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2" \
        --save_path "$base_weights_path/results/mistral_alpha/$(basename ${mola_weights[i]})_${checkpoint}.json"
    else
      echo "Checkpoint directory $checkpoint_dir does not exist."
    fi
  done
done
