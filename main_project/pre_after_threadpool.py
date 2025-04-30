import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Directories
org_chunks = "/*/mimic4/org_chunks"
context_preaft_chunks = "/*/mimic4/pre_aft_context_chunks"
os.makedirs(context_preaft_chunks, exist_ok=True)

# List all files in the directory
chunk_files = os.listdir(org_chunks)

# Function to process each file
def process_file(file):
    try:
        file_path = os.path.join(org_chunks, file)
        output_path = os.path.join(context_preaft_chunks, file)
        
        # Read the file
        df = pd.read_csv(file_path)
        
        # Ensure all values in the 'chunks' column are strings
        df['chunks'] = df['chunks'].fillna('').astype(str)

        # Add context column
        context = []
        for i in range(len(df)):
            prev_rows = df['chunks'].iloc[max(0, i-2):i].tolist()
            next_rows = df['chunks'].iloc[i+1:min(len(df), i+3)].tolist()
            context_text = " ".join(prev_rows + [df['chunks'].iloc[i]] + next_rows)
            context.append(context_text)

        df['context'] = context
        df.to_csv(output_path, index=False)
        return f"Processed {file}"
    except Exception as e:
        return f"Error processing {file}: {e}"

# Use ThreadPoolExecutor to parallelize the file processing
num_threads = os.cpu_count()  # Adjust based on the number of CPUs available
results = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_file, file) for file in chunk_files]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Files"):
        results.append(future.result())

# Log results
with open("processing_results.log", "w") as log_file:
    for result in results:
        log_file.write(result + "\n")

