# AlphaLoRA: Assigning LoRA Experts Based on Layer Training Quality

Welcome to the official GitHub repository for our paper, "[AlphaLoRA: Assigning LoRA Experts Based on Layer Training Quality](https://arxiv.org/html/2410.10054v1)."










1. **Clone the repository**

   ```bash
   git clone https://github.com/peijunallin/alphalora.git
   cd alphalora
   ```

2. **Install dependencies**

   ```bash
   conda create -n alphalora python=3.10 -y
   conda activate alphalora
   pip install -r requirements.txt
   ```

3. **Determine number of experts and Top K**

   Run the `expert_number.py` script to get the number of experts and the top_k parameters:

   ```bash
   CUDA_VISIBLE_DEVICES=3 python expert_number.py \
   --model "mistralai/Mistral-7B-v0.1" \
   --target_sum 160 \
   --beta 2.5
   ```

4. **Train on six datasets**


   ```bash
   bash run_all.sh
   ```

   Before running the script, ensure to adjust the following hyperparameters in `run_all.sh`:

   | Hyperparameters          | Description                                                       |
   |--------------------------|-------------------------------------------------------------------|
   | `base_model`             | The path to the base model.                                       |
   | `root_data_path`         | The path to the six datasets.                                     |
   | `number_experts`         | The number of experts for each layer (32 numbers).        |
   | `top_k`                  | The top K value for each layer (32 numbers).              |
   | `output_dir`             | The directory path to save the LoRA experts' weights.             |

5. **Evaluate on six datasets**

   Ensure that `mola_weights` corresponds to the `output_dir` used during training, and keep the expert number and top_K settings consistent.

   ```bash
   bash eval_all.sh
   ```
   

## Acknowlegements

Our code is based on [MoLA](https://github.com/gcyzsl/mola) and [TempBalance](https://github.com/yefanzhou/tempbalance).

   

