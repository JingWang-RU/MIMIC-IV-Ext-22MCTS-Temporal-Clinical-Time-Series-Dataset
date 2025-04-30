import os
import pandas as pd
from tqdm import tqdm
import re
import multiprocessing as mp
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# ===== 2. Define File Processing Function =====
def process_file(file_path):
    """
    Process a single file to extract hadm_id, patient_id, event, and time.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['patient_id', 'hadm_id', 'Event', 'Time']
    """
    # Extract filename
    filename = os.path.basename(file_path)
    
    # Use regex to extract patient_id and hadm_id
    match = re.match(r'(\d+)\.csv', filename)
    if match:
        # patient_id = int(match.group())
        hadm_id = int(match.group(1))
    else:
        # Handle files that do not match the expected pattern
        # patient_id = None
        hadm_id = None
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Add identifiers
    # df['patient_id'] = patient_id
    df['hadm_id'] = hadm_id
    
    # Reorder columns
    df = df[['hadm_id', 'Event', 'Time']]
    
    return df

# ===== 3. Consolidate All Files into a Single DataFrame =====
def consolidate_data(data_directory, empty_files, num_processes=8):
    """
    Consolidate all CSV files in the data_directory into a single DataFrame.
    
    Args:
        data_directory (str): Path to the directory containing CSV files.
        num_processes (int): Number of parallel processes to use.
        
    Returns:
        pd.DataFrame: Consolidated DataFrame with all events and identifiers.
    """
    # List all CSV files
    all_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.csv') and f not in empty_files]
    
    # Initialize a pool of workers
    pool = mp.Pool(processes=num_processes)
    
    # Use tqdm for progress bar
    results = []
    for result in tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files), desc="Processing Files"):
        if not result.empty:
            results.append(result)
    # Close the pool
    pool.close()
    pool.join()
    
    # Concatenate all DataFrames
    master_df = pd.concat(results, ignore_index=True)
    return master_df

def write_text_blocks_to_file(text_blocks, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for block in text_blocks:
            f.write(block + "\n\n")  # two newlines between sequences

def convert_id_to_text(df_subset):
    """
    Groups the subset by 'id', sorts by time,
    and converts each group into a multiline string.
    """
    text_blocks = []
    
    for id_val, group_df in df_subset.groupby('hadm_id', sort=False):
        # Convert each row in group_df into a line of text
        lines = []
        for _, row in group_df.iterrows():
            line = f"[TIME] {row['Time']} [EVENT] {row['Event']}"
            lines.append(line)
        
        # Join lines with newlines
        block_text = "\n".join(lines)
        
        # You can also add an extra blank line between IDs if you like
        text_blocks.append(block_text)
    
    return text_blocks
# ===== 5. Example Usage =====
if __name__ == "__main__":
    # Define the data directory
    result_path = '/data/wangj47/script/annote/result/train_clean'
    save_directory = '/data/wangj47/script/annote/result/train_clean/gptpretrain'
    empty_df = pd.read_csv(os.path.join(result_path,"empty_files.csv"))#, index=False)
    empty_files = empty_df['emptyfiles'].values

    data_directory = '/data/wangj47/mimic4/combined_bm_emb_annotate' # Replace with your actual path
    # Consolidate data
    # master_df = consolidate_data(data_directory, empty_files, num_processes=16)  # Adjust based on your CPU cores
    # master_df = master_df.dropna(subset=['Time'])
    # master_df['Time'] = master_df['Time'].replace('-', '0')
    # # Now, proceed to convert to numeric
    # master_df['Time'] = pd.to_numeric(master_df['Time'], errors='coerce')
    # # Check for any remaining NaNs
    # remaining_nans = master_df['Time'].isnull().sum()
    # print(f"Number of remaining NaN in 'Time' after replacement: {remaining_nans}")
    # # Optionally, fill NaNs with 0 or another appropriate value
    # master_df['Time'] = master_df['Time'].fillna(0).astype(float)
    master_df = pd.read_csv(os.path.join(result_path, 'annotation_data.csv'))#, index=False)
    print(f"Consolidated DataFrame shape: {master_df.shape}")
    print(master_df.head())
    print(f"Total records: {len(master_df)}")
    
    # Get all unique IDs
    unique_ids = master_df['hadm_id'].unique()
    # Shuffle them
    np.random.shuffle(unique_ids)
    
    # We'll do an 80-10-10 split
    num_ids = len(unique_ids)
    train_end = int(0.8 * num_ids)
    val_end = int(0.9 * num_ids)
    
    train_ids = unique_ids[:train_end]
    val_ids = unique_ids[train_end:val_end]
    test_ids = unique_ids[val_end:]
    
    # print("Train IDs:", train_ids)
    # print("Val IDs:", val_ids)
    # print("Test IDs:", test_ids)
    
    # Filter df
    df_train = master_df[master_df['hadm_id'].isin(train_ids)].copy()
    df_val = master_df[master_df['hadm_id'].isin(val_ids)].copy()
    df_test = master_df[master_df['hadm_id'].isin(test_ids)].copy()
    
    train_text_blocks = convert_id_to_text(df_train)
    val_text_blocks   = convert_id_to_text(df_val)
    test_text_blocks  = convert_id_to_text(df_test)
    
    print("Example train block:\n", train_text_blocks[0])
    
    write_text_blocks_to_file(train_text_blocks, os.path.join(save_directory, "train.txt"))
    write_text_blocks_to_file(val_text_blocks, os.path.join(save_directory,"val.txt"))
    write_text_blocks_to_file(test_text_blocks, os.path.join(save_directory,"test.txt"))
