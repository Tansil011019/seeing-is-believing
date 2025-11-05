"""Preprocessing wrapper - imports from preprocessing/ module"""
import argparse
from preprocessing import process_dataset_parallel as preprocess_dataset, SegmentationDataset
from utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True)
    parser.add_argument('--mask_folder', required=True)
    parser.add_argument('--output_image_folder', required=True)
    parser.add_argument('--output_mask_folder', required=True)
    parser.add_argument('--no_augmentation', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--logfile', default='preprocess.log')
    args = parser.parse_args()
    
    logger = setup_logger(args.logfile, args.verbose, 'preprocess')
    preprocess_dataset(
        args.image_folder, args.mask_folder,
        args.output_image_folder, args.output_mask_folder,
        not args.no_augmentation, args.num_workers, logger
    )
