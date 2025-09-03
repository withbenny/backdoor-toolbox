import os
import re
import json
import shutil
import zipfile
import tempfile
from urllib.request import urlopen
from typing import Dict
from tqdm import tqdm
from datasets import load_dataset


def _pil_ext_from_format(fmt: str) -> str:
    mapping = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "BMP": ".bmp",
        "GIF": ".gif",
        "WEBP": ".webp",
        "TIFF": ".tif",
    }
    return mapping.get(fmt.upper() if fmt else "", ".jpg")


def _safe_name(name):
    name = name.strip().lower()
    name = re.sub(r"[ \t]+", "_", name)
    name = re.sub(r"[^\w\.-]", "", name, flags=re.ASCII)
    name = re.sub(r"_+", "_", name)
    return name or "unknown"


def _save_split(ds, split_name, outpath, id2name):
    split_path = os.path.join(outpath, split_name)
    os.makedirs(split_path, exist_ok=True)

    class_to_idx: Dict[str, int] = {}
    for idx, class_name in id2name.items():
        class_name_safe = _safe_name(class_name)
        os.makedirs(os.path.join(split_path, class_name_safe), exist_ok=True)
        class_to_idx[class_name_safe] = idx

    for i in tqdm(range(len(ds)), desc=f"Saving {split_name}"):
        rec = ds[i]
        img = rec["image"]
        label = int(rec["label"])
        class_name = _safe_name(id2name[label])

        ext = None
        if hasattr(img, "format") and img.format:
            ext = _pil_ext_from_format(img.format)
        else:
            ext = ".jpg"

        filename = f"{i:08d}{ext}"
        outpath = os.path.join(split_path, class_name_safe, filename)

        try:
            if ext.lower() == ".jpg":
                img = img.convert("RGB")
                img.save(outpath, quality=95)
            else:
                img.save(outpath)
        except OSError:
            img = img.convert("RGB")
            outpath = outpath.with_suffix(".jpg")
            img.save(outpath, quality=95)

    with open(os.path.join(split_path, f"{split_name}.json"), "w") as f:
        json.dump(class_to_idx, f)


def get_from_hf(data_name, outpath, **kwargs):
    ds_train = load_dataset(data_name, split="train", **kwargs)
    try:
        ds_val = load_dataset(data_name, split="validation", **kwargs)
    except Exception:
        try:
            ds_val = load_dataset(data_name, split="val", **kwargs)
        except Exception:
            ds_val = None

    label_features = ds_train.features["label"]
    if hasattr(label_features, "names"):
        id2name = {i: name for i, name in enumerate(label_features.names)}
    else:
        num_classes = len(set(int(rec["label"]) for rec in ds_train))
        id2name = {i: f"class_{i:03d}" for i in range(num_classes)}

    if os.path.exists(outpath):
        print(f"Warning: {outpath} already exists and will be overwritten.")
    else:
        os.makedirs(outpath)

    _save_split(ds_train, "train", outpath, id2name)
    _save_split(ds_val, "val", outpath, id2name) if ds_val else None

    print(f"{data_name} saved in {outpath} successfully.")


def get_from_web(data_name, outpath, **kwargs):
    if os.path.exists(outpath):
        print(f"Warning: {outpath} already exists and will be overwritten.")
    else:
        os.makedirs(outpath)
    if data_name.lower() == "tinyimagenet" or data_name.lower() == "tiny-imagenet":
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "tiny-imagenet-200.zip")
            with urlopen(url) as response, open(zip_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            extracted_path = os.path.join(tmpdir, "tiny-imagenet-200")
            for item_name in os.listdir(extracted_path):
                item_path = os.path.join(extracted_path, item_name)
                shutil.move(item_path, os.path.join(outpath, item_name))

        ann = os.path.join(outpath, "val", "val_annotations.txt")
        imgs_dir = os.path.join(outpath, "val", "images")
        with open(ann, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                filename, cls = parts[0], parts[1]
                cls_dir = os.path.join(outpath, "val", cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(imgs_dir, filename)
                dst = os.path.join(cls_dir, filename)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
        print(f"TinyImageNet saved in {outpath} successfully.")


def get_dataset(
    data_name="clane9/imagenet-100", outpath="./data/imagenet100", hf=True, **kwargs
):
    if hf:
        get_from_hf(data_name, outpath=outpath, **kwargs)
    else:
        get_from_web(data_name, outpath=outpath, **kwargs)


if __name__ == "__main__":
    get_dataset(data_name="clane9/imagenet-100", outpath="./data/imagenet100", hf=True)
