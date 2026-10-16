import os
import re
import torch
import numpy as np
import random
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from medpy import metric
from glob import glob
from peft import LoraConfig, get_peft_model
import importlib


from fine_grained_decoder.dpt_dcam_learn import DPT
from dataset import FolderDataset, MedicalAugmentation
from make_split import build_fixed_split


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    data_dir = "../dataset"

    datasets_list = ["BUSI"]
    seeds_list = [314, 404, 2025]

    split_seed = 41

    img_dir_name = "images"
    label_dir_name = "masks"

    decoder_path = "fine_grained_decoder.dpt_dcam_learn"

    weight_paths = {
        "BUSI": {
            314: "",
            404: "",
            2025: "",
        },
    }

    repo_dir = "./dinov3"
    dino_ckpt = "../dataset/dinov3_vitb16.pth"
    dino_size = "b"
    num_classes = 1
    input_h = 256
    input_w = 256
    dice_thr = 0.5
    pixel_spacing = 1.0


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

    metrics_list = ["DICE", "mIOU", "Precision", "Se", "Specificity", "HD95"]

    print("🚀 初始化基础模型架构...")
    if Config.dino_size == "b":
        base_model = torch.hub.load(Config.repo_dir, 'dinov3_vitb16', source='local', weights=Config.dino_ckpt)
    else:
        base_model = torch.hub.load(Config.repo_dir, 'dinov3_vits16', source='local', weights=Config.dino_ckpt)

    lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    backbone = get_peft_model(base_model, lora_cfg)

    decoder_module = importlib.import_module(Config.decoder_path)
    model = decoder_module.DPT(nclass=Config.num_classes, backbone=backbone, use_dcam=True, use_baff=True)
    model = model.to(device)

    for current_dataset in Config.datasets_list:
        print("\n" + "🔥" * 90)
        print(f"🚀 开始全量测试并提取极品权重 | 数据集: {current_dataset}")
        print("🔥" * 90)

        root = os.path.join(Config.data_dir, current_dataset)
        if not os.path.exists(root):
            print(f"⏩ 跳过 {current_dataset}: 找不到数据集目录")
            continue

        curr_mask_ext = ".bmp" if "WHU" in current_dataset.upper() else ".png"

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

        dataset_weights = Config.weight_paths.get(current_dataset, {})

        for seed in Config.seeds_list:
            print(f"\n" + "=" * 60)
            print(f"🔬 正在测试 Seed {seed} 验证集选定的权重...")
            print("=" * 60)

            w_path = dataset_weights.get(seed, "")
            if not w_path:
                print(f"⚠️ Seed {seed}: Config.weight_paths 里没填权重路径，跳过。")
                continue

            if os.path.isdir(w_path):
                cands = sorted(glob(os.path.join(w_path, "best_ep*.pth")))
                if not cands:
                    print(f"⚠️ Seed {seed}: 目录里没有 best_ep*.pth: {w_path}")
                    continue
                w_path = cands[-1]

            if not os.path.isfile(w_path):
                print(f"⚠️ Seed {seed}: 权重文件不存在: {w_path}")
                continue

            print(f"📥 权重: {w_path}")
            model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
            model.eval()

            seed_metrics_collector = {k: [] for k in metrics_list}
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Testing seed{seed}", leave=False):
                    images, true_masks, _ = batch
                    images = images.to(device)

                    outputs = model(images)
                    logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

                    pred_np = (torch.sigmoid(logits) > Config.dice_thr).float().squeeze().cpu().numpy()
                    true_np = true_masks.squeeze().numpy()

                    m = calculate_metrics_numpy(pred_np, true_np, spacing=Config.pixel_spacing)
                    for i, k in enumerate(metrics_list):
                        seed_metrics_collector[k].append(m[i])

            seed_row = {k: np.nanmean(seed_metrics_collector[k]) for k in metrics_list}
            print(f"  └─ seed{seed} -> Test DICE: {seed_row['DICE']:.4f}")

            seed_results[seed] = seed_row
            for k in metrics_list:
                agg_metrics[k].append(seed_row[k])

        if not seed_results:
            continue

        print("\n" + "🌟" * 55)
        print(f"       🏆 {current_dataset} 最终论文指标汇总 (DCAM Learn 极品聚合) 🏆")
        print("🌟" * 55)
        header = f"| {'Seed':<8} | " + " | ".join([f"{k:<11}" for k in metrics_list]) + " |"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for seed in sorted(seed_results.keys()):
            res = seed_results[seed]
            print(f"| {seed:<8} | " + " | ".join([f"{res[k]:<11.4f}" for k in metrics_list]) + " |")
        print(sep)
        print(f"| {'MEAN':<8} | " + " | ".join([f"{np.mean(agg_metrics[k]):<11.4f}" for k in metrics_list]) + " |")
        print(sep)

        print(f"\n📊 [{current_dataset}] 学术论文填表格式 (Mean ± Std):")
        for m_name in metrics_list:
            print(f"🎯 {m_name:<11} : {np.mean(agg_metrics[m_name]):.4f} ± {np.std(agg_metrics[m_name]):.4f}")

    print("\n🎉 全部种子的固定测试已完成（均使用验证集选定权重，无测试集选优）。")


if __name__ == "__main__":
    main()
