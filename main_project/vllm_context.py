import os
import time
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
import sys
common_dir = "/*/script/annote/"
sys.path.append(common_dir)
from common.utils import read_file, write_file

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust based on available GPUs
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable NCCL Peer-to-Peer communication

# Model path and directories
model_path = "/*/model/llama-3.1-8b-instruct"
notes_dir = '/*/mimic4/notes'
chunk_dir = '/*/mimic4/org_chunks'
output_dir = '/*/mimic4/context_chunks'
os.makedirs(output_dir, exist_ok=True)

# Define the prompt template
def create_prompt(chunk):
    return (
        f"A physician is reading the following chunk from a discharge summary:\n\n"
        f"Chunk: {chunk}\n\n"
        f"Based on the physician's understanding, identify the succinct context of this chunk within the overall discharge summary. "
        f"The context should clarify its relevance or meaning, such as:\n"
        f"- For a symptom, describe the full symptom and any related details.\n"
        f"- For a medication dose, specify the medication, its dosage, and purpose.\n"
        f"If the chunk is not related to the patient (e.g., administrative details or unrelated text), respond with an empty context.\n\n"
        f"Context:"
    )

# Initialize the LLM
print("Loading model...")
start_time = time.time()
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=4,  # Adjust based on the number of GPUs
    dtype="bfloat16",
)
print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds.")

# Sampling parameters for text generation
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
    num_return_sequences=1
)

# Get list of files
notes_files = os.listdir(notes_dir)

# Process each file
for file in tqdm(notes_files, desc="Processing Notes Files"):
    filename = file.split('.')[0]
    output_file = os.path.join(output_dir, filename + ".csv")

    # Skip processing if context file already exists
    if os.path.exists(output_file):
        #print(f"Context file for {filename} already exists. Skipping...")
        continue

    try:
        # Read original text and chunks
        original_text = read_file(os.path.join(notes_dir, filename + '.txt'))
        chunks = pd.read_csv(os.path.join(chunk_dir, filename + '.csv'))['chunks'].values

        # Create a DataFrame to store chunks and their contexts
        df = pd.DataFrame({"chunks": chunks, "context": [""] * len(chunks)})

        # Annotate each chunk with context
        for row in tqdm(df.itertuples(index=True), total=len(df), desc=f"Annotating Chunks in {filename}"):
            chunk = row.chunks
            prompt = create_prompt(chunk)

            # Generate the context using vLLM
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            context = generated_text.split("Context:")[1].strip() if "Context:" in generated_text else generated_text
            df.at[row.Index, "context"] = context

        # Save the annotated DataFrame
        df.to_csv(output_file, index=False)
        print(f"Processed file: {filename}, results saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

print("All files processed.")
