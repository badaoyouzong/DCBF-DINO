import os
import json
import random
import hashlib


class SplitConfig:
    data_dir = "../dataset"

    datasets_list = ["BUSI", "Dataset_b", "BUSI_WHU"]

    img_dir_name = "images"

    split_seed = 41
    train_ratio = 0.8

    verify = True


def _list_valid_files(img_dir: str):
    files = [f for f in os.listdir(img_dir)
             if "normal" not in f.lower()
             and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    files.sort()
    return files


def build_fixed_split(dataset_root: str, split_seed: int = 41,
                      img_dir_name: str = "images", verify: bool = True):
    img_dir = os.path.join(dataset_root, img_dir_name)
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    files = _list_valid_files(img_dir)
    if len(files) == 0:
        raise RuntimeError(f"No valid image files found in {img_dir}")

    random.seed(split_seed)
    random.shuffle(files)

    total = len(files)
    n_train = int(total * 0.8)
    n_val = int((total - n_train) / 2)

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    if verify:
        sets = {k: set(v) for k, v in splits.items()}
        assert len(sets["train"] | sets["val"] | sets["test"]) == total, "划分并集不等于全量图片！"
        assert not (sets["train"] & sets["val"]), "train 与 val 存在重叠！"
        assert not (sets["train"] & sets["test"]), "train 与 test 存在重叠！"
        assert not (sets["val"] & sets["test"]), "val 与 test 存在重叠！"

    md5s = {}
    for name, names in splits.items():
        content = "\n".join(names)
        out_path = os.path.join(dataset_root, f"{name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        md5s[name] = hashlib.md5(content.encode("utf-8")).hexdigest()

    manifest = {
        "dataset": os.path.basename(os.path.normpath(dataset_root)),
        "split_seed": split_seed,
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "total_images": total,
        "counts": {k: len(v) for k, v in splits.items()},
        "excluded": "文件名含 'normal' 的样本不参与划分",
        "md5": md5s,
    }
    manifest_name = f"split_seed{split_seed}_8-1-1.json"
    with open(os.path.join(dataset_root, manifest_name), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main():
    print("=" * 72)
    print(f"👑 固定划分生成器 | split_seed = {SplitConfig.split_seed} | 比例 8:1:1")
    print("=" * 72)

    for current_dataset in SplitConfig.datasets_list:
        root = os.path.join(SplitConfig.data_dir, current_dataset)
        if not os.path.exists(root):
            print(f"⚠️ [跳过] 找不到数据集目录: {root}")
            continue

        info = build_fixed_split(
            dataset_root=root,
            split_seed=SplitConfig.split_seed,
            img_dir_name=SplitConfig.img_dir_name,
            verify=SplitConfig.verify,
        )

        c = info["counts"]
        print(f"\n📂 {current_dataset}  (总图数 {info['total_images']})")
        print(f"   train = {c['train']:>4}  val = {c['val']:>3}  test = {c['test']:>3}"
              f"   | 实际比例 {c['train'] / info['total_images']:.3f} : "
              f"{c['val'] / info['total_images']:.3f} : {c['test'] / info['total_images']:.3f}")
        print(f"   md5  train={info['md5']['train'][:12]}  val={info['md5']['val'][:12]}  "
              f"test={info['md5']['test'][:12]}")
        print(f"   ✓ 写入 train.txt / val.txt / test.txt 与 {current_dataset}/split_seed"
              f"{SplitConfig.split_seed}_8-1-1.json")

    print("\n🎉 固定划分完成：所有训练种子都会共用上面这份 train/val/test。")


if __name__ == "__main__":
    main()
