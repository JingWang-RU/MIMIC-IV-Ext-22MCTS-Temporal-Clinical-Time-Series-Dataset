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
# ===== 1. Define Label Mappings =====
label_mapping = {
    'No Correlation': 0,
    'Positive Correlation': 1,  # Potential Outcome
    'Negative Correlation': 2   # Reason
}

inverse_label_mapping = {v: k for k, v in label_mapping.items()}
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

# ===== 4. Revised generate_event_pairs Function =====
def generate_event_pairs(df, label_mapping):
    """
    Generate question-context pairs with labels based on temporal relationships,
    including time_bin information.
    Args:
        df (pd.DataFrame): DataFrame containing 'patient_id', 'hadm_id', 'Event', 'Time', and 'time_bin' columns, ordered chronologically.
        label_mapping (dict): Dictionary mapping label names to integers.
                              Example: {'No Correlation': 0, 'Positive Correlation': 1, 'Negative Correlation': 2}

    Returns:
        list of tuples: Each tuple contains (question, context, label, time_bin)
    """
    pairs = []

    # Group by 'patient_id' and 'hadm_id' to handle multiple admissions per patient
    grouped = df.groupby(['hadm_id'])

    for (hadm_id), group in tqdm(grouped, desc="Generating Event Pairs"):
        # Sort events by Time
        group_sorted = group.sort_values('Time')

        events = group_sorted['Event'].tolist()
        times = group_sorted['Time'].tolist()
        time_bins = group_sorted['time_bin'].tolist()

        # Generate pairs within this group
        for i in range(len(events) - 1):
            current_event = events[i]
            current_time = times[i]
            current_time_bin = time_bins[i]
            next_event = events[i + 1]
            next_time = times[i + 1]
            next_time_bin = time_bins[i + 1]

            # Determine the label based on time comparison
            if next_time < current_time:
                # Context event occurred before the question event - Negative Correlation (Reason)
                label = label_mapping.get('Negative Correlation', 2)
            elif next_time > current_time:
                # Context event occurred after the question event - Positive Correlation (Outcome)
                label = label_mapping.get('Positive Correlation', 1)
            else:
                # Context event occurred simultaneously - No Correlation
                label = label_mapping.get('No Correlation', 0)
            # Append the pair and its label to the list
            pairs.append((current_event, next_event, label, current_time_bin))
    return pairs
    
def bin_time(master_df):
    """
    Bin the 'Time' column into discrete bins and add 'time_bin' column.
    Args: master_df (pd.DataFrame): Consolidated DataFrame with 'Time' column.
    Returns: pd.DataFrame: DataFrame with added 'time_bin' column.
    """
    # Define bins (adjust based on data distribution)
    bins = [-np.inf, -60, -30, -15, 0, 15, 30, 60, 120, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Example labels
    master_df['time_bin'] = pd.cut(master_df['Time'], bins=bins, labels=labels, right=False).astype(int)
    
    # Handle any NaN in 'time_bin' by assigning the last bin
    master_df['time_bin'] = master_df['time_bin'].fillna(len(labels)-1).astype(int)

    return master_df

# ===== 5. Example Usage =====
if __name__ == "__main__":
    # Define the data directory
    result_path = '/data/wangj47/script/annote/result/train_clean'
    empty_df=pd.read_csv(os.path.join(result_path,"empty_files.csv"))#, index=False)
    empty_files = empty_df['emptyfiles'].values

    data_directory = '/data/wangj47/mimic4/combined_bm_emb_annotate' # Replace with your actual path
    # Consolidate data
    master_df = consolidate_data(data_directory, empty_files, num_processes=16)  # Adjust based on your CPU cores
    master_df = master_df.dropna(subset=['Time'])
    master_df['Time'] = master_df['Time'].replace('-', '0')
    
    # Now, proceed to convert to numeric
    master_df['Time'] = pd.to_numeric(master_df['Time'], errors='coerce')
    
    # Check for any remaining NaNs
    remaining_nans = master_df['Time'].isnull().sum()
    print(f"Number of remaining NaN in 'Time' after replacement: {remaining_nans}")
    
    # Optionally, fill NaNs with 0 or another appropriate value
    master_df['Time'] = master_df['Time'].fillna(0).astype(float)

    print(f"Consolidated DataFrame shape: {master_df.shape}")
    print(master_df.head())
    print(f"Total records: {len(master_df)}")
    # Bin the Time column
    print("\nBinning the 'Time' column...")
    master_df = bin_time(master_df)

    # Generate event pairs
    event_pairs = generate_event_pairs(master_df, label_mapping)
    print(f"Total event pairs: {len(event_pairs)}")
    print("\nSample Event Pairs:")
      
    master_df.to_csv(os.path.join(result_path,'consolidated_events.csv'), index=False)
    print("Consolidated DataFrame saved as 'consolidated_events.csv'.")
    
    import pickle
    # Save event_pairs using pickle
    with open(os.path.join(result_path,'event_pairs.pkl'), 'wb') as f:
        pickle.dump(event_pairs, f)
    print("Event pairs saved as 'event_pairs.pkl'.")
   