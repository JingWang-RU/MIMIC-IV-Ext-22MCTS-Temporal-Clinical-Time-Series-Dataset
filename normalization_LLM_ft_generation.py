import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import time
import re

# Please kindly consider cite our work if you feel this dataset useful. Thank you so much for your interest!
# Wang, J., Niu, X., Zhang, T., Shen, J., Kim, J., & Weiss, J. (2025). 
# MIMIC-IV-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp 
# (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/dkj6-r828
# @article{wang2025mimic,
#   title={MIMIC-$\backslash$RNum $\{$4$\}$-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp for Risk Prediction},
#   author={Wang, Jing and Niu, Xing and Kim, Juyong and Shen, Jie and Zhang, Tong and Weiss, Jeremy C},
#   journal={arXiv preprint arXiv:2505.00827},
#   year={2025}
# }

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the clinical events dataset.
    
    Args:
        file_path: Path to the clinical_event_timestamp.csv file
        
    Returns:
        Preprocessed DataFrame ready for feature extraction
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Remove gender-related events
    gender_events = ['Female', 'Male', 'F', 'M']
    df_clean = df[~df['Event'].isin(gender_events)].copy()
    print(f"After removing gender events: {df_clean.shape}")
    
    # Remove purely numeric events
    numeric_only = df_clean["Event"].astype(str).str.fullmatch(
        r"\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*", na=False
    )
    df_clean = df_clean[~numeric_only].copy()
    print(f"After removing numeric events: {df_clean.shape}")
    
    # Remove time-related patterns from events
    time_rx = re.compile(
        r"(?i)("
        r"\b\d+(?:\.\d+)?\s*(?:years?|yrs?|y|months?|mos?|mo|weeks?|wks?|wk|days?|d|hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|sec)\b"
        r"|"
        r"\b\d{1,2}:\d{2}\b"
        r"|"
        r"\b\d{1,2}\s*(?:am|pm)\b"
        r"|"
        r"\b\d+\s*(?:y/o|yo|yr[- ]?old|years?\s*old)\b"
        r")"
    )
    
    years_only_rx = re.compile(
        r"(?i)^\s*(?:"
        r"\d+(?:\.\d+)?\s*(?:years?|yrs?|yr|y/o|yo|yr[- ]?old|year[- ]?old|years?\s*old)"
        r"|"
        r"(?:years?|yrs?|yr)"
        r")\s*$"
    )
    
    ev_str = df_clean["Event"].astype(str)
    has_time_inside = ev_str.str.contains(time_rx, na=False)
    is_years_only = ev_str.str.fullmatch(years_only_rx, na=False)
    
    df_clean = df_clean[~(has_time_inside | is_years_only)].copy()
    print(f"After removing time-related events: {df_clean.shape}")
    
    # Sort by time and keep first occurrence of each event per patient
    df_sorted = df_clean.sort_values(by='Time')
    df_final = df_sorted.drop_duplicates(subset=['Hadm_id', 'Event'], keep='first')
    print(f"After deduplication: {df_final.shape}")
    
    return df_final


def make_feature_label_windows_per_id(
    df: pd.DataFrame,
    id_col: str = "Hadm_id",
    event_col: str = "Event",
    time_col: str = "Time",
    samples_per_id: int = 10,
    window_size: int = 11,
    drop_incomplete: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Build a dataset with columns [id_col, 'feature', 'label'].

    For each patient:
      - take all contiguous windows of length 11 (sorted by time),
      - for each window and for each position j in 0..10:
          label  = time_j:event_j    (the single left-out event)
          feature= comma-joined of the 10 remaining time_i:event_i in chronological order
      - deduplicate (feature, label) pairs
      - randomly sample up to `samples_per_id` pairs

    Returns a DataFrame: [id, feature, label]
       feature: "t0:e0, t1:e1, ... (10 pairs, in time order)"
       label:   "t_label:e_label"
    """
    assert window_size == 11, "Spec requires 10 features + 1 label → window_size=11."

    # Keep only needed columns and drop missing
    work = df[[id_col, event_col, time_col]].dropna().copy()
    rng = np.random.default_rng(random_state)
    out_rows = []

    for pid, g in work.groupby(id_col, sort=False):
        # sort by time ascending to define contiguity
        g_sorted = g.sort_values(time_col, kind="stable")
        n = len(g_sorted)

        if n < window_size:
            if drop_incomplete:
                continue

        # collect all candidate (feature, label) pairs for this patient
        cand = []

        for start in range(0, max(0, n - window_size + 1)):
            win = g_sorted.iloc[start:start + window_size]

            # precompute the (time:event) strings for the window in order
            pairs = [f"{str(t)}:{str(e)}"
                     for t, e in zip(win[time_col].tolist(), win[event_col].tolist())]

            # for each possible label position, build a candidate
            for j in range(window_size):
                label_str = pairs[j]
                feature_str = ", ".join(pairs[:j] + pairs[j+1:])  # keep order, omit the label
                cand.append((feature_str, label_str))

        if not cand:
            continue

        # deduplicate within patient
        cand_unique = list(dict.fromkeys(cand))  # preserves order, removes duplicates

        # sample up to `samples_per_id` without replacement
        k = min(samples_per_id, len(cand_unique))
        sel_idx = rng.choice(len(cand_unique), size=k, replace=False)
        for idx in sel_idx:
            feat, lab = cand_unique[idx]
            out_rows.append({id_col: pid, "feature": feat, "label": lab})

    return pd.DataFrame(out_rows, columns=[id_col, "feature", "label"])


def main():
    """Main function to run the entire pipeline."""
    start_time = time.time()
    
    # UPDATE THIS PATH to your dataset location
    data_file_path = "clinical_event_timestamp.csv"  # Change this path as needed
    
    try:
        # Load and preprocess data
        df_processed = load_and_preprocess_data(data_file_path)
        
        # Generate feature-label windows
        print("Generating feature-label windows...")
        out = make_feature_label_windows_per_id(
            df_processed,
            id_col="Hadm_id",
            event_col="Event", 
            time_col="Time",
            samples_per_id=10,
            random_state=42,
        )
        
        # Rename columns for LLM fine-tuning
        out.rename(
            columns={"feature": "observation", "label": "hypothesis"},
            inplace=True
        )
        
        end_time = time.time()
        
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        print(f"Final dataset shape: {out.shape}")
        print("\nSample of the output:")
        print(out.head())
        
        # Optional: Save the result
        # out.to_csv('clinical_events_processed.csv', index=False)
        # print("Results saved to 'clinical_events_processed.csv'")
        
    except FileNotFoundError:
        print(f"Error: File not found at {data_file_path}")
        print("Please update the 'data_file_path' variable with the correct path to your dataset.")
    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()