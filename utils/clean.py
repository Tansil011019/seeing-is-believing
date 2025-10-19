def clean_dataframe(ground_truth_file: str,
                   output_file: str) -> None:
    """
    Melts ground truth table from wide to long format and filter it.
    Args:
        ground_truth_file (str): Path to the ground truth CSV
        output_file (str): Path to save the melted CSV
    
    """

    import pandas as pd
    df = pd.read_csv(ground_truth_file)
    melted_df = df.melt(id_vars=["image"], var_name="label", value_name
                          ="value")
    melted_df['image'] = melted_df['image'].astype(str) + '.jpg'
    melted_df = melted_df[melted_df["value"] == 1]
    melted_df = melted_df.drop(columns=["value"])
    melted_df.to_csv(output_file, index=False)