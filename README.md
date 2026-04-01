
<div align="center">
<h1>用于换衣行人重识别的迭代扩散范式</h1>
</div>
###  Environment Setup
Create and activate the conda environment using the configuration file:
```bash
conda env create -f environment_shsg.ymlconda 
activate idm
```
### Dataset Preparation

### High-Quality Person Image Filtering
Modify the dataset paths in the __main__ function of the filtering script:
```python  
if __name__ == "__main__":  
 # Source image directory and target directory 
 person_dir = "/path/to/your/dataset/train"  # Replace with your dataset path 
 output_dir = "/path/to/your/HQ_FIlter_Path/output"  # Replace with your output path     # Execute filtering  
 filter_person_images_per5(person_dir, output_dir)  
```  
Run the filtering script for the target dataset:
  
```bash  
cd ./SHSG-IDM-VTONpython ./Person_HQ_Filter_LTCC.pypython ./Person_HQ_Filter_PRCC.py
```
### Upper Body Clothing Extraction

Clothing Mask Extraction
Use [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) to extract human parsing masks for both datasets from all training images.

Upper Body Clothing Mask Generation：

```bash
cd acquire_upper_clothing_mask
python npy_to_colored.py --input LTCC_ReID(prcc)/train --out out_colored_all 
python delete_non_lower_masks.py --masks-dir out_colored_all/masks 
python generate_upper_from_lower_masks.py --masks-dir out_colored_all/masks --out out_upper_from_lower
```
Upper Body Clothing Image Extraction：

```bash
cd acquire_upper_clothing 
python main.py --masks LTCC_ReID(prcc)/out_upper_from_lower --images LTCC_ReID/train --out LTCC_ReID/processed_upper --resize-mode auto_resize
```
Afterwards, select the most complete and suitable upper body clothing image for each clothing type of each person, and organize them into the corresponding folders.

## Hard Sample Generation

### IDM-VTON Model Preparation
Download the pre-trained IDM-VTON weights from the official repository:   
[IDM-VTON GitHub 仓库](https://github.com/yisol/IDM-VTON)
Ensure the Gradio demo runs properly locally.

### Hard Sample Generation via Clothing Swapping

Modify the dataset paths in the main() function:
```python
def main(): 
	dataset_name = "ltcc" # 可选 ltcc / prcc 
	
	if dataset_name == "ltcc":
		 person_folder = "/path/to/your/person/images" 
		 cloth_folder = "/path/to/your/cloth/images" 
		 output_folder = "/path/to/your/output/folder" 
	elif dataset_name == "prcc": 
		 person_folder = "/path/to/your/prcc/person/images" 
		 cloth_folder = "/path/to/your/prcc/cloth/images" 
		 output_folder = "/path/to/your/prcc/output/folder" 
	else: 
		 raise ValueError("请设置数据集名称！")
```
Run the sample generation script:

```bash
cd ./SHSG-IDM-VTON/IDM-VTON-main
python change-cloth-script.py
```
## High-Fidelity Hard Sample Optimization (Diffusion Model)

### Closed-Loop Diffusion Generation

```bash
cd [Project Root Directory]
python -u -m SHSG_IDM_VTON.pipeline.run_one_seed \
  --prcc_root(ltcc_root) "Path to the rgb subfolder in the PRCC dataset" ("Path to the subfolder of person images after clothing swapping in the LTCC dataset") \
  --work_root "runs\prcc_closedloop_lora_r3_full22350" \
  --seed 0 \
  --backend idmvton \
  --idmvton_model_dir "path/to/your/IDM-VTON" \
  --rounds 3 \
  --gen_per_round \
  --enable_lora_train \
  --lora_topk 2000 \
  --lora_steps 100 \
  --reid_epochs 0 \
  --export_splits_to_cwd
```
### Export Final Optimized Samples
First, find the code below in run_one_seed.py and modify the corresponding number
```python
if bool(args.reuse_final_if_exists) and round1_filtered_root.exists():
				print(f"[train_only] reuse {round1_filtered_root}", flush=True)
	else:
		if args.round1_gen_dir:
				round1_gen_dir = Path(args.round1_gen_dir)
		else:
				round1_gen_dir = work_root / "closed_loop" / "round_1" / "gen"
```
(Change to the number of rounds you have run; default is 1 round)
```bash
python -u -m run_one_seed \
  --dataset_name ltcc(or prcc) \
  --ltcc_root "Path to the LTCC_ReID dataset" \
  --work_root "runs\ltcc_filter_job" \
  --seed 0 \
  --train_only \
  --roundx_export_mode origin_final \
  --roundx_gen_dir "Latest person images generated in round x" \
  --reid_epochs 0
```

```python
if bool(args.reuse_final_if_exists) and round1_filtered_root.exists():
				print(f"[train_only] reuse {round1_filtered_root}", flush=True)
	else:
		if args.round1_gen_dir:
				round1_gen_dir = Path(args.round1_gen_dir)
		else:
				round1_gen_dir = work_root / "closed_loop" / "round_1" / "gen"
```
## Re-Identification Model Training and Testing

### Download Pre-trained ViT Weights

Download [vit_base_patch16_224](https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA)
Extract code: eu9f
Place the weights in the corresponding pre-trained model folder.
### Training on the PRCC Dataset

```bash
cd [Project Root Directory]
D:\anaconda3\envs\idm\python -u -m SHSG-IDM-VTON\pipelines\run_prcc_train \
    --train_only \
    --round1_export_mode origin_final \
    --reuse_final_if_exists \
    --reid_train_mode train_plus_final \
    --prcc_root "Path to the rgb subfolder in the PRCC dataset" \
    --timm_name vit_base_patch16_224 \
    --local_pretrained "Path to the model weights" \
    --work_root "runs\reid_b_test_once_20ep" \
    --seed 0 \
    --reid_epochs 120 \
    --reid_lr 0.00035 \
    --batch_size 64 \
    --grad_accum_steps 2
```

### Training on the LTCC Dataset

```bash
cd [Project Root Directory]
D:\anaconda3\envs\idm\python -u SHSG-IDM-VTON/pipelines/run_ltcc_train.py \
    --train_only \
    --reuse_final_if_exists \
    --reid_train_mode train_plus_final \
    --ltcc_root "Path to the LTCC_ReID dataset" \
    --timm_name vit_base_patch16_224 \
    --local_pretrained "Path to the model weights" \
    --work_root "runs\ltcc_reid_test_once" \
    --seed 0 \
    --reid_epochs 1 \
    --reid_lr 0.00035 \
    --batch_size 64 \
    --grad_accum_steps 2 \
    --pk_k 4 \
    --min_images_per_id 4
```
