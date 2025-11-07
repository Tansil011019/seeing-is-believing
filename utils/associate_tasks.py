import os
import pandas as pd
from glob import glob

def associate_task1_3(task1_dir: str, task3_dir: str) -> pd.DataFrame:
    """
    Cross-verify images between Task 1 and Task 3 directories and find overlapping images.
    
    Args:
        task1_dir (str): Directory containing Task 1 images
        task3_dir (str): Directory containing Task 3 images
        
    Returns:
        pd.DataFrame: DataFrame containing overlapping image information
    """
    # Get all jpg files from both directories
    task1_files = glob(os.path.join(task1_dir, "*.jpg"))
    task3_files = glob(os.path.join(task3_dir, "*.jpg"))
    
    # Extract just the image IDs (filename without extension)
    task1_ids = set(os.path.splitext(os.path.basename(f))[0] for f in task1_files)
    task3_ids = set(os.path.splitext(os.path.basename(f))[0] for f in task3_files)
    
    # Find overlapping image IDs
    overlapping_ids = task1_ids.intersection(task3_ids)
    
    # Create a DataFrame with the overlapping information
    df = pd.DataFrame({
        'image_id': list(overlapping_ids),
        'task1_path': [os.path.join(task1_dir, f"{id}.jpg") for id in overlapping_ids],
        'task3_path': [os.path.join(task3_dir, f"{id}.jpg") for id in overlapping_ids]
    })
    
    # Sort by image_id for consistency
    df = df.sort_values('image_id').reset_index(drop=True)
    
    # Print summary information
    print(f"Total images in Task 1: {len(task1_ids)}")
    print(f"Total images in Task 3: {len(task3_ids)}")
    print(f"Number of overlapping images: {len(overlapping_ids)}")
    print("\nFirst 5 overlapping images:")
    print(df.head().to_string())
    
    return df

if __name__ == "__main__":
    # Example usage
    task1_dir = "datasets/ISIC2018_Task1-2_Training_Input"
    task3_dir = "datasets/ISIC2018_Task3_Training_Input"
    
    overlap_df = associate_task1_3(task1_dir, task3_dir)
