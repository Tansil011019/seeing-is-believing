# rotary_oversample_isic_mt.py
import os
import random
import pandas as pd
from PIL import Image
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import shutil

def rotate_image(img, angle, expand=False):
    return img.rotate(angle, resample=Image.BILINEAR, expand=expand)

def copy_file(src, dst):
    try:
        shutil.copy(src, dst)
        return True
    except Exception:
        return False

def save_rotated_image(src_path, out_path, angle, expand=False):
    try:
        img = Image.open(src_path).convert("RGB")
        aug = rotate_image(img, angle, expand)
        aug.save(out_path, format="JPEG", quality=95)
        return True
    except Exception:
        return False

def rotary_oversample_from_csv(
    csv_path="/data/datasets/cleaned_data.csv",
    image_dir="/data/datasets/ISIC2018_Task3_Training_Input",
    out_dir="/data/datasets/ISIC2018_Task3_Training_Input_balanced",
    angles=(90, 180, 270),
    random_jitter=5,
    ext=".jpg",
    num_threads=8,
    seed=42
):
    random.seed(seed)
    df = pd.read_csv(csv_path)
    counts = Counter(df["label"])
    max_count = max(counts.values())
    print("Class counts before oversampling:", counts)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    new_rows = []
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["label"]].append(row["image"])

    # Copy originals first
    copy_tasks = []
    for label, files in grouped.items():
        cls_dir = os.path.join(out_dir, label)
        os.makedirs(cls_dir, exist_ok=True)
        for f in files:
            src = os.path.join(image_dir, f + ext)
            dst = os.path.join(cls_dir, f + ext)
            copy_tasks.append((src, dst, label, f))

    print("Copying originals...")
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(copy_file, src, dst) for src, dst, _, _ in copy_tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Copying"):
            pass

    new_rows.extend({"image": f"{label}/{f}", "label": label} for _, _, label, f in copy_tasks)

    # Prepare oversampling tasks
    rot_tasks = []
    for label, files in grouped.items():
        needed = max_count - len(files)
        if needed <= 0:
            continue
        cls_dir = os.path.join(out_dir, label)
        for i in range(needed):
            src_name = random.choice(files)
            src_path = os.path.join(image_dir, src_name + ext)
            base_angle = random.choice(angles)
            jitter = random.uniform(-random_jitter, random_jitter)
            angle = base_angle + jitter
            new_name = f"{src_name}_rot{int(angle)}_{i}"
            out_path = os.path.join(cls_dir, new_name + ext)
            rot_tasks.append((src_path, out_path, angle, label, new_name))

    print("Generating rotated images...")
    created = 0
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {ex.submit(save_rotated_image, src, dst, angle): (label, name)
                   for src, dst, angle, label, name in rot_tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rotating"):
            ok = future.result()
            if ok:
                label, name = futures[future]
                new_rows.append({"image": f"{label}/{name}", "label": label})
                created += 1

    new_df = pd.DataFrame(new_rows)
    new_csv_path = os.path.join(out_dir, "balanced_data.csv")
    new_df.to_csv(new_csv_path, index=False)

    print(f"Oversampling complete. Created {created} new images.")
    print(f"Balanced dataset saved at {out_dir}")
    print(f"New CSV written to {new_csv_path}")

if __name__ == "__main__":
    rotary_oversample_from_csv(
        csv_path="/data/datasets/cleaned_data.csv",
        image_dir="/data/datasets/ISIC2018_Task3_Training_Input",
        out_dir="/data/datasets/ISIC2018_Task3_Training_Input_balanced",
        angles=(90, 180, 270),
        random_jitter=5,
        ext=".jpg",        # Change to ".png" if needed
        num_threads=12,    # Adjust to CPU cores
        seed=42
    )
