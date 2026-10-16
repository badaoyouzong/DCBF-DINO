import os
import torch
import numpy as np
import random
import warnings
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage import measure
from medpy import metric
from fvcore.nn import FlopCountAnalysis
from peft import LoraConfig, get_peft_model


from fine_grained_decoder.dpt_dcam_learn import DPT
from dataset import FolderDataset, MedicalAugmentation
from make_split import build_fixed_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    data_dir = "../dataset"

    datasets_list = ["BUSI","Dataset_b","BUSI_WHU","BUS"]

    split_seed = 41

    img_dir_name = "images"
    label_dir_name = "masks"

    repo_dir = "./dinov3"
    dino_ckpt = "../dataset/dinov3_vitb16.pth"
    dino_size = "b"
    num_classes = 1
    input_h = 256
    input_w = 256
    dice_thr = 0.5
    pixel_spacing = 1.0

    weight_paths = {
        "BUSI":{
            314: "",
            404: "",
            2025: ""
        },
        "BUSI_WHU": {
            314:"",
            404:"",
            2025:""
        },
        "BUS": {
            314: "",
            404: "",
            2025: ""
        },
        "Dataset_b": {
            314: "",
            404: "",
            2025: ""
        }
    }


def keep_largest_connected_component(mask):
    labels = measure.label(mask)
    if labels.max() == 0: return mask
    areas = np.bincount(labels.flat)[1:]
    largest_idx = np.argmax(areas) + 1
    return (labels == largest_idx).astype(mask.dtype)


HD95_EMPTY_PENALTY = 360.624


def calculate_metrics_numpy(pred_mask, true_mask, spacing=1.0):
    eps = 1e-6
    sum_pred = np.sum(pred_mask)
    sum_true = np.sum(true_mask)

    TP = np.sum((pred_mask == 1) & (true_mask == 1))
    FP = np.sum((pred_mask == 1) & (true_mask == 0))
    FN = np.sum((pred_mask == 0) & (true_mask == 1))
    TN = np.sum((pred_mask == 0) & (true_mask == 0))

    if sum_true == 0 and sum_pred == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0, 0.0
    elif sum_true == 0 or sum_pred == 0:
        specificity = TN / (TN + FP + eps)
        return 0.0, 0.0, 0.0, 0.0, specificity, HD95_EMPTY_PENALTY
    else:
        dice = (2. * TP) / (2. * TP + FP + FN + eps)
        iou = TP / (TP + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        se = TP / (TP + FN + eps)
        specificity = TN / (TN + FP + eps)
        try:
            hd95 = metric.binary.hd95(pred_mask, true_mask, voxelspacing=spacing)
        except:
            hd95 = np.nan
    return dice, iou, precision, se, specificity, hd95


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_transform = MedicalAugmentation(size=(Config.input_h, Config.input_w), is_train=False)

    print("🚀 初始化 Ours_Full 模型 (LoRA + DCAM + UFA)...")

    base = torch.hub.load(Config.repo_dir, 'dinov3_vitb16', source='local', weights=Config.dino_ckpt)
    lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    backbone = get_peft_model(base, lora_cfg)

    model = DPT(nclass=Config.num_classes, backbone=backbone, use_dcam=True, use_ufa=True).to(device)

    metrics_list = ["DICE", "mIOU", "Precision", "Se", "Specificity", "HD95"]

    base_save_dir = "/home/yz/CODE/dataset/runs"

    for current_dataset in Config.datasets_list:
        print("\n" + "🔥" * 90)
        print(f"🚀 开始评估数据集: {current_dataset}")
        print("🔥" * 90)

        root = os.path.join(Config.data_dir, current_dataset)
        if not os.path.exists(root) or current_dataset not in Config.weight_paths:
            continue

        curr_mask_ext = ".bmp" if "WHU" in current_dataset.upper() else ".png"
        current_dataset_weights = Config.weight_paths[current_dataset]

        split_info = build_fixed_split(
            dataset_root=root, split_seed=Config.split_seed, img_dir_name=Config.img_dir_name
        )
        _cnt = split_info["counts"]
        print(f"\n👑 [固定划分] split_seed={Config.split_seed} (8:1:1) | "
              f"train={_cnt['train']} / val={_cnt['val']} / test={_cnt['test']} "
              f"(总图数 {split_info['total_images']})")

        test_dataset = FolderDataset(
            root=root, split="test", img_dir_name=Config.img_dir_name,
            label_dir_name=Config.label_dir_name, transform=val_transform,
            mask_ext=curr_mask_ext
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(f"📊 固定测试集样本数: {len(test_dataset)}")

        seed_results = {}
        agg_metrics = {metric: [] for metric in metrics_list}

        for seed, weight_path in current_dataset_weights.items():
            if not os.path.exists(weight_path): continue

            model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
            model.eval()

            seed_metrics_collector = {k: [] for k in metrics_list}

            model_name = "Ours_Full"
            current_save_dir = os.path.join(base_save_dir, f"{model_name}_{current_dataset}_seed{seed}")
            os.makedirs(current_save_dir, exist_ok=True)

            with torch.no_grad():
                for idx, batch in enumerate(tqdm(test_loader, desc=f"Seed {seed:<5}")):
                    images, true_masks, filenames = batch
                    images = images.to(device)

                    outputs = model(images)
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                    pred_np = (torch.sigmoid(logits) > Config.dice_thr).float().squeeze().cpu().numpy()
                    true_np = true_masks.squeeze().numpy()

                    try:
                        raw_fname = filenames[0] if isinstance(filenames, (list, tuple)) else str(filenames)
                        clean_fname = os.path.splitext(os.path.basename(raw_fname))[0] + ".png"
                    except:
                        clean_fname = f"idx_{idx}.png"

                    save_path = os.path.join(current_save_dir, clean_fname)

                    gt_img = (true_np * 255).astype(np.uint8)
                    canvas = np.stack([gt_img] * 3, axis=-1)

                    pred_img = (pred_np * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(canvas, contours, -1, (0, 0, 255), 2)

                    cv2.imwrite(save_path, canvas)

                    m = calculate_metrics_numpy(pred_np, true_np, spacing=Config.pixel_spacing)
                    for i, k in enumerate(metrics_list):
                        seed_metrics_collector[k].append(m[i])

            seed_row = {k: np.nanmean(seed_metrics_collector[k]) for k in metrics_list}
            seed_results[seed] = seed_row
            for k in metrics_list: agg_metrics[k].append(seed_row[k])

        if not seed_results: continue

        print("\n" + "🌟" * 48)
        print(f"       🏆 {current_dataset} 最终论文指标汇总 (Ours_Full) 🏆")
        print("🌟" * 48)
        header = f"| {'Seed':<8} | " + " | ".join([f"{k:<11}" for k in metrics_list]) + " |"
        sep = "-" * len(header)
        print(sep);
        print(header);
        print(sep)
        for seed in sorted(seed_results.keys()):
            res = seed_results[seed]
            print(f"| {seed:<8} | " + " | ".join([f"{res[k]:<11.4f}" for k in metrics_list]) + " |")
        print(sep)
        print(f"| {'MEAN':<8} | " + " | ".join([f"{np.mean(agg_metrics[k]):<11.4f}" for k in metrics_list]) + " |")
        print(sep)

    print("\n🎉 Ours_Full 评估出图任务圆满完成！[白块+原始红线] 已存入 runs 目录。")


if __name__ == "__main__":
    main()
