import os
import math
import time
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
import torch
import transformers
import gzip
import time

# ---------------------------
# 1. SLURM_ARRAY_TASK_ID Setup
# ---------------------------
# Get the current array task ID
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))  


# ---------------------------
# 2. Define Paths and Load Data
# ---------------------------
notes_dir = '/data/wangj47/mimic4/notes'
chunk_dir = '/data/wangj47/mimic4/combined_bm_emb'
annote_dir = '/data/wangj47/mimic4/combined_bm_emb_annotate'
chunks_files = os.listdir(chunk_dir)
# Gather all chunk files
# chunks_files = os.listdir(chunk_dir)
# Existing files set (those already processed)
existing_files = set(
    os.path.splitext(os.path.basename(f))[0]
    for f in os.listdir(annote_dir)
)

# 3c. Filter out files that have already been processed
files_to_process = [
    os.path.splitext(f)[0]  # remove ".csv"
    for f in chunks_files
    if os.path.splitext(f)[0] not in existing_files
]

num_files = len(files_to_process)
print(f"Total files pending processing: {num_files}")

ARRAY_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
NUM_TASKS = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 30))  # Default to 30 if not set
chunk_size = math.ceil(num_files / NUM_TASKS)
start_index = ARRAY_TASK_ID * chunk_size
end_index = min(start_index + chunk_size, num_files)
print(NUM_TASKS, start_index, end_index)
# ---------------------------
# 3. Initialize Model / Pipeline
# ---------------------------
model_path = "/data/wangj47/model/llama-3.1-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
text_generation_pipeline = pipeline(
    "text-generation",
    model=model_path,
    torch_dtype=torch.bfloat16,
    eos_token_id=tokenizer.eos_token_id,
    device=0  # Use GPU (device=0, assuming each sub-job has one GPU)
)

# Optional: define prefix/suffix if you have them
prefix = r'''
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a physician. Extract the clinical events and estimate the related time stamp from the discharge summary. 
the discharge summary and a list of chunks that may contain clinical events. here are the guidance:

1. Identify clinical events **only** from the provided chunks. A clinical event is a free-text specification of an entity pertaining to or with the potential to affect the person's health that can temporally located.
2. Include as many clinical events as possible. Each event must come directly from the chunks provided. Do not create or infer events from the full document if they are not explicitly present in the chunks.
3. For each identified clinical event:
   - Extract the event directly from the chunk or use it as-is if it is concise enough.
   - Separate conjunctive phrases into their component events and assign them the same timestamp.
     For example: "fever and rash" → "fever" | -72 and "rash" | -72.
   - Include events with durations (e.g., treatments or symptoms) and assign the time as the **start** of the interval.
   - Include all relevant events: termination/discontinuation events, pertinent negative findings (e.g., "no shortness of breath" or "denies chest pain").
4. Estimate the relative timing of the event:
   - Base the timing on the **context of the entire document** but ensure the event itself comes from the chunks.
   - Use explicit or inferred temporal information to assign a numeric timestamp.
   - If explicit timing is not mentioned, use your medical knowledge to approximate a value in hours.
5. Formatting rules for timing:
   - Do not use complex temporal expressions like "four weeks ago" or timestamps like `-672`.
   - Convert such expressions into simple numeric approximations in hours.
     For example:
       - "1 week" → -168 hours
       - "1 month" → -720 hours
   - Use consistent numeric values for approximations when the exact timing is not available.
6. **Output format**:
   - Return only a list of clinical events(usually 3 to 6 tokens)  and their estimated timestamps.
   - Separate the clinical event and its timestamp with a pipe (`|`), like this:
     `<clinical event> | <timestamp>`
7. If no clinical events are found in the chunks, output `none` (case-insensitive) and nothing else.

**Timing Guidelines:**
- The admission event is always assigned a timestamp of 0.
- Events occurring before admission have negative timestamps, measured in hours.
- Events occurring after admission have positive timestamps, measured in hours.
- When explicit timing is not available:
   - Use inferred durations from the document's context.
   - Apply clinical judgment to estimate a reasonable time.
   - Avoid using large or complex time expressions like `-672`. Instead, provide approximate values (e.g., "a few weeks ago" → `-336` hours).

**Important Notes:**
- Only use events that appear in the chunks provided.
- Do not infer or create events from the document if they are not present in the chunks.
- Maximize inclusion of events from the chunks.

**Input Details:**
- **Document**: A comprehensive discharge summary providing the patient’s history and context.
- **Chunks**: A list of small text segments (3 to 10 tokens each) extracted from the document.

**Output Format:**
Only output the events (1 to 5 tokens) and related timestamp in a list, each item has two elements, the first is the clinical event, \
the second is the estimated relative timestamp in hours, separate the two elements with a pipe ('|')

**Important Notes:**
- Maximize the inclusion of relevant clinical events.
- Avoid omitting events that can reasonably be inferred.
- If a time range is mentioned (e.g., "a few weeks to months"), average the duration and convert it to hours (e.g., 1 month ≈ 720 hours).
- If no explicit temporal information is available, estimate the timing based on the sequence of events or overall document context.
- Avoid using large or complex time expressions like `-672`. Instead, provide approximate values (e.g., "a few weeks ago" → `-336` hours).

**Example Input:**
Document:
You are a physician.  Extract the clinical events and the related time stamp from the case report. The admission event has timestamp 0. If the event is not available, we treat the event, e.g. current main clinical diagnosis or treatment with timestamp 0. The events happened before event with 0 timestamp have negative time, the ones after the event with 0 timestamp have positive time. The timestamp are in hours. The unit will be omitted when output the result. If there is no temporal information of the event, please use your knowledge and events with temporal expression before and after the events to provide an approximation. We want to predict the future events given the events happened in history. For example, here is the case report.\
Sex: M  was admitted to the hospital with a 3-day history of fever and rash. Four weeks ago, he was diagnosed with acne and received the treatment with minocycline, 100 mg daily, for 3 weeks. With increased WBC count, eosinophilia, and systemic involvement, this patient was diagnosed with DRESS syndrome. The fever and rash persisted through admission, and diffuse erythematous or maculopapular eruption with pruritus was present. One day later the patient was discharged.\

reasoning steps:
Let's find the locations of event in the case report, it shows that four weeks ago of fever and rash, four weeks ago, he was diagnosed with acne and receive treatment. So the event of fever and rash happen four weeks ago, 672 hours, it is before admitted to the hospital, so the time stamp is -672. diffuse erythematous or maculopapular eruption with pruritus was documented on the admission exam, so the time stamp is 0 hours, since it happens right at admission. DRESS syndrome has no specific time, but it should happen soon after admission to the hospital, so we use our clinical judgment to give the diagnosis of DRESS syndrome the timestamp 0.

chunks:
1. "Sex: M  was admitted"
2. "to the hospital with a"
3. "3-day history of fever and rash"
4. "diagnosed with acne"
5. "received the treatment with minocycline"
6. "increased WBC count"
7. "eosinophilia, and systemic involvement"
8. "diffuse erythematous or maculopapular eruption with"
9. "was diagnosed with DRESS syndrome"
10. "fever and rash persisted through admission"
11. "day later the patient was discharged"
Separate conjunctive phrases into its component events and assign them the same timestamp (for example, the separation of 'fever and rash' into 2 events: 'fever' and 'rash')  If the event has duration, assign the event time as the start of the time interval. Attempt to use the text span without modifications except 'history of' where applicable. Include all patient events, even if they appear in the discussion; do not omit any events; include termination/discontinuation events; include the pertinent negative findings, like 'no shortness of breath' and 'denies chest pain'.  Show the events and timestamps in rows, each row has two columns: one column for the event, the other column for the timestamp.  The time is a numeric value in hour unit. The two columns are separated by a pipe '|' as a bar-separated file. Skip the title of the table. Reply with the table only. Create a table from the following case:

**Example Output:**
Sex: M| 0\
admitted to the hospital | 0\
fever | -72\
rash | -72\
acne |  -672\
minocycline |  -672\
increased WBC count | 0\
eosinophilia| 0\
systemic involvement| 0\
diffuse erythematous or maculopapular eruption| 0\
DRESS syndrome | 0\
fever persisted | 0\
rash persisted | 0\
discharged | 24\

Now analyze the following input, do not include any instructions, explanations, or other text in your output. Only provide the event and estimated timestamps:

Document:
{document}

Chunks:
{chunks}

<|eot_id|><|start_header_id|>user<|end_header_id|>
'''
suffix = r'''
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''
# Additional containers for metadata
id_list = []
sex_list = []
sum_time = []

# Example parse function (replace with your own)
def parse_clinical_events(lines):
    # Split the output into lines
    # lines = llm_output.strip().split("\n")
    
    # Initialize a list to store parsed data
    data = []

    # Iterate over lines and extract valid clinical events
    for line in lines:
        # Skip introductory text and empty lines
        if "|" in line and not line.startswith("Based on the provided input"):
            # Split the line into event and timestamp parts
            parts = line.split("|")
            if len(parts) == 2:
                event = parts[0].split(".", 1)[-1].strip()  # Remove numbering (e.g., "1.")
                timestamp = parts[1].strip()
                if event:  # Ensure the event is not empty
                    data.append([event, timestamp])

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=["Event", "Time"])
    return df

find_output = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
# ---------------------------
# 4. Process Each File in Range
# ---------------------------
for index in tqdm(range(start_index, end_index), desc=f"Processing: Job {array_id}"):
    filename = files_to_process[index]
    annote_file = os.path.join(annote_dir, filename + '.csv')
    
    # Skip if this file has already been processed
    if os.path.exists(annote_file):
        continue
    
    # Start timer
    start_time = time.time()
    
    # 4a. Read content
    with open(os.path.join(notes_dir, filename + '.txt'), 'r') as f:
        content = f.read()
    
    # 4b. Read CSV chunks
    chunks = pd.read_csv(os.path.join(chunk_dir, filename + '.csv'))
    chunks_list = [
        x for x in chunks['chunks'].values
        if isinstance(x, str) and x.strip()
    ]
    
    # 4c. Build prompt
    prompt = prefix.format(document=content, chunks="\n".join(chunks_list)) + suffix
    
    # 4d. Run text-generation pipeline
    sequences = text_generation_pipeline(
        prompt,
        do_sample=True,
        top_k=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 4e. Parse the output
    # Adjust this logic based on how you parse
    
    output = sequences[0]['generated_text'].split('\n')
    if find_output in output:
        result = output[output.index(find_output) + 1:]
    else:
        # If not found, handle gracefully
        result = output
    
    result_df = parse_clinical_events(result)
    result_df.to_csv(annote_file, index=False)
    
    # 4f. Collect metadata
    sex_ind = content.find('Sex: ')
    if sex_ind != -1:
        sex_info = content[sex_ind: sex_ind + 8]
        sex_list.append(sex_info)
    else:
        sex_list.append("NotFound")
    
    id_list.append(filename)
    
    process_time = time.time() - start_time
    sum_time.append(process_time)

# ---------------------------
# 5. Save Summaries (Optional)
# ---------------------------
# If you want each sub-job to save time or metadata, do so here:
time_file = f"/data/wangj47/annotate_time_part_{array_id}.csv"
pd.DataFrame({'filename': id_list, 'time': sum_time, 'sex': sex_list}).to_csv(time_file, index=False)

print(f"Job {array_id} completed. Processed indices {start_index} to {end_index-1}.")
