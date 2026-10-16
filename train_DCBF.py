import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import random
from datetime import datetime
from glob import glob
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import importlib
from make_split import build_fixed_split


class Config:
    data_dir = "../dataset"

    datasets_list = ["BUSI"]
    decoder_path = "fine_grained_decoder.dpt_dcam_learn"

    repo_dir = "./dinov3"
    dino_ckpt = "../dataset/dinov3_vitb16.pth"

    img_dir_name = "images"
    label_dir_name = "masks"
    img_ext = ".bmp"
    mask_ext = ".bmp"

    dino_size = "b"
    num_classes = 1
    input_h = 256
    input_w = 256

    epochs = 100
    batch_size = 8
    lr = 5e-4
    weight_decay = 1e-3

    seeds_list = [314, 404, 2025]

    split_seed = 41

    num_workers = 4
    dice_thr = 0.5


def save_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('bmh')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue', linewidth=1.5)
    axs[0, 0].plot(epochs, history['val_loss'], label='Val Loss', color='tab:red', linewidth=1.5)
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, history['train_iou'], label='Train IoU', color='tab:blue', linewidth=1.5)
    axs[0, 1].plot(epochs, history['val_iou'], label='Val IoU', color='tab:red', linewidth=1.5)
    axs[0, 1].set_title('IoU')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('IoU')
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, history['train_dice'], label='Train Dice', color='tab:blue', marker='.', linewidth=1.5)
    axs[1, 0].plot(epochs, history['val_dice'], label='Val Dice', color='tab:red', linewidth=1.5)
    axs[1, 0].set_title('Dice')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Dice')
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, history['val_dice'], label='Val Dice', color='tab:blue', marker='.', linewidth=1.5)
    axs[1, 1].plot(epochs, history['val_iou'], label='Val IoU', color='tab:red', marker='.', linewidth=1.5)
    axs[1, 1].set_title('Validation Metrics')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Metrics')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    return (inter + eps) / union


def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return (2 * inter + eps) / union


def train_one_epoch(model, train_loader, optimizer, device, num_classes=1, dice_thr=0.5, epoch=0):
    model.train()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    decay_factor = max(0.0, 1.0 - (epoch / Config.epochs))
    current_w1 = 0.4 * decay_factor
    current_w2 = 0.8 * decay_factor

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch}/{Config.epochs} [W_aux:{current_w1:.2f}/{current_w2:.2f}]")

    for i, (inputs, targets, _) in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        main_logits, aux1_logits, aux2_logits = model(inputs)
        loss_target = targets if num_classes == 1 else targets.squeeze(1).long()

        loss_main = criterion(main_logits, loss_target)
        loss_aux1 = criterion(aux1_logits, loss_target)
        loss_aux2 = criterion(aux2_logits, loss_target)

        loss = loss_main + current_w1 * loss_aux1 + current_w2 * loss_aux2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            dice = dice_binary_torch(main_logits, targets, thresh=dice_thr).mean().item()
            iou = iou_binary_torch(main_logits, targets, thresh=dice_thr).mean().item()
            dice_scores.append(dice)
            iou_scores.append(iou)

        pbar.set_postfix(Loss=f"{loss.item():.4f}", Dice=f"{dice:.4f}")

    return total_loss / max(1, len(train_loader)), float(np.mean(dice_scores)), float(np.mean(iou_scores))


@torch.no_grad()
def evaluate(model, val_loader, device, num_classes=1, dice_thr=0.5):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []

    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    for inputs, targets, _ in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        if num_classes == 1:
            loss = criterion(logits, targets)
        else:
            loss = criterion(logits, targets.squeeze(1).long())

        total_loss += loss.item()

        dice = dice_binary_torch(logits, targets, thresh=dice_thr).mean().item()
        iou = iou_binary_torch(logits, targets, thresh=dice_thr).mean().item()

        dice_scores.append(dice)
        iou_scores.append(iou)

    avg_loss = total_loss / max(1, len(val_loader))
    avg_dice = float(np.mean(dice_scores)) if len(dice_scores) > 0 else 0.0
    avg_iou = float(np.mean(iou_scores)) if len(iou_scores) > 0 else 0.0

    return avg_loss, avg_dice, avg_iou


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    var_name = "dcam_learn"
    use_dcam = True
    use_baff = True

    for current_dataset in Config.datasets_list:
        root = os.path.join(Config.data_dir, current_dataset)
        if not os.path.exists(root):
            print(f"⚠️ [警告] 找不到数据集目录: {root}，已跳过 {current_dataset} 的训练。")
            continue

        split_info = build_fixed_split(
            dataset_root=root,
            split_seed=Config.split_seed,
            img_dir_name=Config.img_dir_name,
        )
        _cnt = split_info["counts"]
        print(f"\n👑 [固定划分] split_seed={Config.split_seed} (8:1:1) | "
              f"train={_cnt['train']} / val={_cnt['val']} / test={_cnt['test']} "
              f"(总图数 {split_info['total_images']})")

        for current_seed in Config.seeds_list:
            print("\n" + "★" * 70)
            print(f"🚀 [START] Dataset: {current_dataset} | Variant: {var_name} | Seed: {current_seed}")
            print("★" * 70)

            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            random.seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)

            current_time = datetime.now().strftime("%m%d_%H%M")
            save_root = f"./runs/{var_name}/{var_name}_{Config.dino_size}_{Config.input_h}_{current_dataset}_seed{current_seed}_{current_time}"
            ckpt_dir = os.path.join(save_root)

            os.makedirs(save_root, exist_ok=True)
            print(f"📂 权重及曲线图将保存在: {save_root}")

            if Config.dino_size == "b":
                backbone = torch.hub.load(Config.repo_dir, 'dinov3_vitb16', source='local',
                                          weights=Config.dino_ckpt)
            else:
                backbone = torch.hub.load(Config.repo_dir, 'dinov3_vits16', source='local',
                                          weights=Config.dino_ckpt)

            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["qkv", "proj"],
                lora_dropout=0.05,
                bias="none"
            )
            backbone = get_peft_model(backbone, lora_config)

            decoder_module = importlib.import_module(Config.decoder_path)
            model = decoder_module.DPT(nclass=Config.num_classes, backbone=backbone,
                                       use_dcam=use_dcam, use_baff=use_baff)
            model = model.to(device)

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=Config.lr, weight_decay=Config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-6)

            from dataset import FolderDataset, MedicalAugmentation
            train_transform = MedicalAugmentation(size=(Config.input_h, Config.input_w), is_train=True)
            val_transform = MedicalAugmentation(size=(Config.input_h, Config.input_w), is_train=False)

            train_dataset = FolderDataset(
                root=root, split="train", img_dir_name=Config.img_dir_name,
                label_dir_name=Config.label_dir_name, transform=train_transform
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=Config.batch_size, shuffle=True,
                num_workers=Config.num_workers, drop_last=True
            )

            val_dataset = FolderDataset(
                root=root, split="val", img_dir_name=Config.img_dir_name,
                label_dir_name=Config.label_dir_name, transform=val_transform
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=False,
                num_workers=Config.num_workers, drop_last=False
            )

            best_val_dice = -1.0
            last_best_path = None

            history = {
                'train_loss': [], 'val_loss': [],
                'train_dice': [], 'val_dice': [],
                'train_iou': [], 'val_iou': []
            }

            for epoch in range(1, Config.epochs + 1):
                train_loss, train_dice, train_iou = train_one_epoch(
                    model, train_loader, optimizer, device,
                    num_classes=Config.num_classes, dice_thr=Config.dice_thr, epoch=epoch
                )

                val_loss, val_dice, val_iou = evaluate(
                    model, val_loader, device,
                    num_classes=Config.num_classes, dice_thr=Config.dice_thr
                )

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_dice'].append(train_dice)
                history['val_dice'].append(val_dice)
                history['train_iou'].append(train_iou)
                history['val_iou'].append(val_iou)

                current_lr = scheduler.get_last_lr()[0]
                print(f"📊 Epoch {epoch} Result: "
                      f"Train [Loss:{train_loss:.4f} Dice:{train_dice:.4f} IoU:{train_iou:.4f}]\n"
                      f">>>>>>>>>>>>>>>>Val [Loss:{val_loss:.4f} Dice:{val_dice:.4f} IoU:{val_iou:.4f}] LR:{current_lr:.6f}")

                scheduler.step()

                if val_dice > best_val_dice:
                    if last_best_path is not None and os.path.exists(last_best_path):
                        os.remove(last_best_path)

                    best_val_dice = val_dice
                    best_path = os.path.join(ckpt_dir, f"best_ep{epoch:03d}_dice{val_dice:.4f}.pth")
                    torch.save(model.state_dict(), best_path)
                    last_best_path = best_path
                    print(f"✨ New Best Model Saved!")

                save_training_curves(history, os.path.join(ckpt_dir, "training_curves.png"))

            print("=" * 60)
            print(f"🏆 [Finished] Best Val Dice = {best_val_dice:.4f}")
            print("=" * 60)

            del model, optimizer, scheduler, backbone
            torch.cuda.empty_cache()

    print(f"\n🎉 所有的实验（{len(Config.datasets_list)}个数据集 × {len(Config.seeds_list)}个种子）已跑完！")


if __name__ == "__main__":
    main()
