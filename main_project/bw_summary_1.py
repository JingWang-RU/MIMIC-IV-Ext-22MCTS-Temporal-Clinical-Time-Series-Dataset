from tqdm import tqdm
from datasets import Dataset  # Import Hugging Face datasets
import os
import time
import torch
import pandas as pd
import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
common_dir = "/*/script/annote/"
sys.path.append(common_dir)
from common.utils import read_file, clean_text_mimic, write_file

# Initialize Accelerator
accelerator = Accelerator()
# Directories
model_dir = '/*/model/llama-3.1-8b-instruct'
notes_dir = '/*/mimic4/notes'
notes_files = os.listdir(notes_dir)
summary_dir = '/*/mimic4/summary-8B'
os.makedirs(summary_dir, exist_ok=True)
existing_summaries = set(os.listdir(summary_dir))  # List of completed summaries
notes_files = [f for f in notes_files if f"{os.path.splitext(f)[0]}_summary.txt" not in existing_summaries]

# Load tokenizer and model with Accelerator
print("Loading model and tokenizer...")
with accelerator.main_process_first():
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

# Prepare model for distributed setup
model = accelerator.prepare(model)

# Get list of files and distribute among processes
num_chunks = accelerator.state.num_processes  # Total parallel processes
chunk_size = len(notes_files) // num_chunks + (len(notes_files) % num_chunks > 0)
start_idx = accelerator.state.local_process_index * chunk_size
end_idx = min(start_idx + chunk_size, len(notes_files))

my_files = notes_files[start_idx:end_idx]

# Setup the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.bfloat16,
)
# Process files
results = []
start_time = time.time()

for filename in tqdm(my_files, desc=f"Processing Files for Process {accelerator.state.local_process_index}"):
    output_file = os.path.join(summary_dir, f"{os.path.splitext(filename)[0]}.txt")
    
    # Skip processing if summary file already exists
    if os.path.exists(output_file):
        print(f"Summary for {filename} already exists. Skipping...")
        continue

    try:
        # Read and clean text
        text = read_file(os.path.join(notes_dir, filename))
        start_keyword = "Sex"
        start_index = text.find(start_keyword)
        extracted_content = text[start_index:] if start_index != -1 else text
        clean_text = clean_text_mimic(extracted_content)

        # Generate prompt
        prompt = (
            f"Based only on the information provided, summarize the following discharge summary "
            f"without adding any assumptions or outside information, such as the age of the patient:\n\n{clean_text}\n\nSummary:"
        )

        # Generate summary using the pipeline
        response = pipe(prompt, truncation=True, max_new_tokens=200, num_return_sequences=1, do_sample=True)
        summary = response[0]["generated_text"]
        real_summary = summary.split("Summary:")[1].strip() if "Summary:" in summary else summary

        # Write summary to file
        write_file(real_summary, output_file)
        results.append({"filename": filename, "status": "success"})

    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        results.append({"filename": filename, "status": "failure", "error": str(e)})

# Finalize timing
summary_time = time.time() - start_time

# Save timing and results information (only main process writes)
if accelerator.is_main_process:
    timing_df = pd.DataFrame({"time_taken": [summary_time]})
    timing_df.to_csv(os.path.join(summary_dir, "summary_time.csv"), index=False)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(summary_dir, "processing_results.csv"), index=False)

print(f"Process {accelerator.state.local_process_index} finished processing {len(my_files)} files in {summary_time:.2f} seconds.")
print(f"Model device: {model.device}")

# Finalize Accelerator
accelerator.wait_for_everyone()
