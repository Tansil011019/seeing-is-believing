"""
. . .
like leaves, we gather a handful of images
from the forest of datasets — a small, kindred grove
to practice on, to learn from, to carry forward.
. . .

Generates a smaller "dummy_datasets/" containing a random subset of
images from the input folder and their associated ground-truth files.

Usage example:
  python gen_dummy_seg.py --size 10
  python gen_dummy_seg.py --size 5 --seed 42 --outdir my_dummy

Defaults (can be overridden):
  input_dir: datasets/ISIC2018_Task1-2_Training_Input
  gt_dirs:   datasets/ISIC2018_Task1_Training_GroundTruth,
             datasets/ISIC2018_Task2_Training_GroundTruth

The script copies the selected input files into <outdir>/inputs/
and all matching groundtruth files starting with <id>_ into <outdir>/gt/.
A CSV manifest <outdir>/manifest.csv maps input -> associated GT files.
"""

from pathlib import Path
import argparse
import random
import shutil
import csv
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Generate dummy segmentation dataset")
    p.add_argument('--size', '-n', type=int, required=True,
                   help='number of random files to pick from the input folder')
    p.add_argument('--input_dir', type=str,
                   default='datasets/ISIC2018_Task1-2_Training_Input',
                   help='directory containing input images in {id}.{ext} format')
    p.add_argument('--gt_dirs', type=str, nargs='*',
                   default=['datasets/ISIC2018_Task1_Training_GroundTruth',
                            'datasets/ISIC2018_Task2_Training_GroundTruth'],
                   help='one or more ground-truth directories to search for {id}_{any_text}.{ext} files')
    p.add_argument('--outdir', type=str, default='dummy_datasets',
                   help='output directory to create the dummy dataset')
    p.add_argument('--seed', type=int, default=None,
                   help='random seed for reproducibility')

    return p.parse_args()


def safe_copy(src: Path, dst_dir: Path) -> Path:
    """Copy src into dst_dir preserving name. If name exists, append a counter."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    # avoid clobbering
    stem = src.stem
    suffix = src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    input_dir = Path(args.input_dir)
    gt_dirs = [Path(p) for p in args.gt_dirs]
    outdir = Path(args.outdir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input_dir '{input_dir}' does not exist or is not a directory", file=sys.stderr)
        sys.exit(1)

    # list input files (files only)
    input_files = [p for p in sorted(input_dir.iterdir()) if p.is_file()]
    if not input_files:
        print(f"ERROR: no files found in input_dir '{input_dir}'", file=sys.stderr)
        sys.exit(1)

    available = len(input_files)
    if args.size > available:
        print(f"WARNING: requested size {args.size} > available files {available}. Setting size = {available}")
        size = available
    else:
        size = args.size

    # choose random files
    selected = random.sample(input_files, k=size)

    inputs_out = outdir / 'inputs'
    gt_out = outdir / 'gt'
    inputs_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []  # list of (input_relpath, [gt_relpaths])

    for inp in selected:
        # copy input file
        copied_input = safe_copy(inp, inputs_out)

        # extract id (stem). Assumes filename format {id}.{ext}
        file_id = inp.stem

        # find matches in all gt_dirs: pattern {id}_*
        matches = []
        for gtdir in gt_dirs:
            if not gtdir.exists() or not gtdir.is_dir():
                # skip non-existing gt dirs but warn
                print(f"WARNING: ground-truth directory '{gtdir}' does not exist, skipping", file=sys.stderr)
                continue
            # glob for files that start with file_id + '_'
            pattern = f"{file_id}_*"
            for candidate in gtdir.glob(pattern):
                if candidate.is_file():
                    # copy and collect
                    copied = safe_copy(candidate, gt_out)
                    matches.append(str(copied.relative_to(outdir)))

        if not matches:
            print(f"WARNING: no ground-truth files found for id '{file_id}'", file=sys.stderr)

        manifest_rows.append((str(copied_input.relative_to(outdir)), ';'.join(matches)))

    # write manifest.csv
    manifest_path = outdir / 'manifest.csv'
    with manifest_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['input', 'ground_truths'])
        for inp_rel, gts in manifest_rows:
            writer.writerow([inp_rel, gts])

    # summary
    print(f"Done. Created dummy dataset in: {outdir}")
    print(f" - inputs: {len(list(inputs_out.iterdir()))} files")
    print(f" - gt:     {len(list(gt_out.iterdir()))} files")
    print(f" - manifest: {manifest_path}")


if __name__ == '__main__':
    main()