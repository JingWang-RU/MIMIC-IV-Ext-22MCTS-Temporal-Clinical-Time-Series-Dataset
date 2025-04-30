import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, AdamW,BertTokenizer
import xml.etree.ElementTree as ET
import re
from transformers import BertForSequenceClassification
from datetime import datetime
from time import gmtime, strftime
import pickle
import pandas as pd
import csv
from ranx import evaluate
from transformers import logging
logging.set_verbosity_error()
import os
import torch
from transformers import BertModel

def load_local_bert_and_classifier(model_dir, checkpoint_file, device="cuda"):
    """
    Loads a pretrained BERT encoder and a custom classifier head from a local checkpoint.
    
    Args:
        model_dir (str): Directory where the checkpoint file is stored.
        checkpoint_file (str): Filename of the checkpoint (e.g., 'bert_gpus_best_model_with_time.bin').
        device (str): Device ("cuda" or "cpu").
    
    Returns:
        base_model (BertModel): The loaded BERT encoder.
        classifier_head (nn.Module): The loaded custom classifier head.
    """
    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize the base BERT model.
    base_model = BertModel.from_pretrained("bert-base-uncased")
    base_model.to(device)
    base_model.eval()
    
    # Initialize your custom classifier head.
    # Make sure that ClassifierHead is defined in your code.
    classifier_head = ClassifierHead(input_dim=768, hidden_dim=256, num_labels=3)
    classifier_head.to(device)
    classifier_head.eval()
    
    # Prepare dictionaries for weights for the BERT encoder and the classifier head.
    new_checkpoint_base = {}
    new_checkpoint_classifier = {}
    
    for key, value in checkpoint.items():
        # For keys that start with "module.bert.", remove that prefix and store for the encoder.
        if key.startswith("module.bert."):
            new_key = "bert." + key[len("module.bert."):]
            new_checkpoint_base[new_key] = value
        # For keys that start with "module.classifier.", remove that prefix and store for classifier.
        elif key.startswith("module.classifier."):
            new_key = key[len("module.classifier."):]
            new_checkpoint_classifier[new_key] = value
        # If the key starts with "module." but does not match, try to assign appropriately.
        elif key.startswith("module."):
            if "bert" in key:
                new_key = key[len("module."):]
                new_checkpoint_base[new_key] = value
            elif "classifier" in key or "time_embedding" in key:
                # For classifier/time_embedding keys, remove "module." and store.
                new_key = key[len("module."):]
                new_checkpoint_classifier[new_key] = value
            else:
                continue
        else:
            # If key already starts with "bert." assign to encoder;
            # if it starts with "classifier." assign to classifier.
            if key.startswith("bert."):
                new_checkpoint_base[key] = value
            elif key.startswith("classifier."):
                new_checkpoint_classifier[key[len("classifier."):]] = value
            else:
                continue

    # Load the state dictionaries into the respective modules.
    missing_base, unexpected_base = base_model.load_state_dict(new_checkpoint_base, strict=False)
    missing_classifier, unexpected_classifier = classifier_head.load_state_dict(new_checkpoint_classifier, strict=False)
    
    # print("BERT encoder missing keys:", missing_base)
    # print("BERT encoder unexpected keys:", unexpected_base)
    # print("Classifier head missing keys:", missing_classifier)
    # print("Classifier head unexpected keys:", unexpected_classifier)
    
    return base_model, classifier_head


def read_trec_qrels(trec_qrel_file):
    '''
    Read a TREC style qrel file and return a dict:
        QueryId -> docName -> relevance
    '''
    qrels = {}
    with open(trec_qrel_file) as fh:
        for line in fh:
            try:
                query, zero, doc, relevance = line.strip().split()
                docs = qrels.get(query, {})
                docs[doc] = float(relevance)
                qrels[query] = docs
            except Exception as e:
                print ("Error: unable to split line in 4 parts", line)
                raise e
    return qrels

import torch

def predict_label(base_model, classifier_head, tokenizer, device, text_a, text_b):
    """
    Predicts the class label (0,1,2) for the given pair of texts.
    """
    # Put models in eval
    base_model.eval()
    classifier_head.eval()

    # Tokenize
    encoding = tokenizer(
        text_a,
        text_b,
        truncation=True,
        max_length=512,
        padding='longest',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        if outputs.pooler_output is not None:
            cls_embedding = outputs.pooler_output
        else:
            # If no pooler_output, use [CLS] embedding from last_hidden_state
            cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = classifier_head(cls_embedding)
        pred = torch.argmax(logits, dim=1).item()
    
    return pred

# ====== 1) Data Preparation ======
def parse_topics(topics_file):
    """
    Reads a file like topics-2014_2015-description.topics, extracting:
      - <NUM>...</NUM> => query_id
      - <TITLE>...</TITLE> => query_text
    Returns a dict: { query_id (str): query_text (str) }
    """
    with open(topics_file, 'r', encoding='utf-8') as f:
        data = f.read()

    blocks = data.split('<TOP>')
    query_dict = {}
    for block in blocks:
        num_match = re.search(r"<NUM>(.*?)</NUM>", block, re.DOTALL)
        title_match = re.search(r"<TITLE>(.*?)</TOP>", block, re.DOTALL)
        if num_match and title_match:
            query_id = num_match.group(1).strip()
            query_text = title_match.group(1).strip()
            query_dict[query_id] = query_text
    return query_dict

def parse_qrels(qrels_file):
    """
    Returns a list of (query_id, doc_id, label_int),
    e.g. [("20141", "NCT00000408", 0), ...],
    where label_int is typically 0, 1, or 2.
    """
    doc_list = []
    pairs = []
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Format: query_id, 0, doc_id, relevance
            # e.g. 20141   0   NCT00000408   0
            parts = line.strip().split()
            if len(parts) == 4:
                query_id = parts[0]
                doc_id = parts[2]
                rel = int(parts[3])  # 0,1,2
                pairs.append((query_id, doc_id, rel))
                if doc_id not in doc_list:
                    doc_list.append(doc_id)
    return pairs, doc_list

def clean_text(text):
    """Clean up XML artifacts/newlines, etc."""
    if not text:
        return ''
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('&#xD;', ' ')
    text = ' '.join(text.split())
    return text.strip()

def read_and_extract(file_path):
    """
    Read <brief_title> and <criteria><textblock> from a clinical trial XML.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    brief_title_el = root.find('.//brief_title')
    brief_title = clean_text(brief_title_el.text) if brief_title_el is not None else ''
    criteria_el = root.find('.//criteria/textblock')
    criteria_text = clean_text(criteria_el.text) if criteria_el is not None else ''
    paragraph = f"{brief_title} {criteria_text}".strip()
    return paragraph

def build_training_samples(qrels_list, query_dict, doc_dict):
    """
    Combine the query text & doc text into (sent_a, sent_b, label).
    """
    all_samples = []
    for (query_id, doc_id, label) in qrels_list:
        q_text = query_dict.get(query_id, "")
        d_text = doc_dict.get(doc_id, "")
        if not q_text or not d_text:
            continue
        # label is assumed 0,1,2
        all_samples.append((q_text, d_text, label))
    return all_samples

# ====== 2) Dataset & Collation ======
class PairwiseTextDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List of tuples (text_a, text_b, label_int)
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    """
    Takes a list of (text_a, text_b, label) from the dataset, and tokenizes them.
    """
    sentences_a = [item[0] for item in batch]
    sentences_b = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    encoded = tokenizer(
        sentences_a,
        sentences_b,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels
    }

# ====== 3) Classifier Head (Your Custom Architecture) ======
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels=3):
        super(ClassifierHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class MyModel(nn.Module):
    def __init__(self, bert_name="bert-base-uncased", hidden_dim=256, num_labels=3):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.classifier = ClassifierHead(input_dim=768, hidden_dim=hidden_dim, num_labels=num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # Use pooler_output if available; otherwise, use the [CLS] token from last_hidden_state.
        cls_embedding = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits
# ====== 4) Main Training Loop ======

# ========== Example Data Loading Process ==========
if __name__ == "__main__":
    path = "/data/wangj47/datasets/TrialGPT/dataset/sigir/data/"

    # 1) Parse topics -> query_dict
    query_dict = parse_topics(os.path.join(path, "topics-2014_2015-description.topics"))

    # 2) Parse qrels -> list of (query_id, doc_id, label_int)
    qrel_path = os.path.join(path, "qrels-clinical_trials.txt")
    qrels_list, doc_ids = parse_qrels(qrel_path)

    # 3) Build doc_dict by reading each .xml
    base_dir = os.path.join(path, "clinicaltrials.gov-16_dec_2015")
    doc_dict = {}
    for i, xml_id in enumerate(doc_ids):
        if i % 2000 == 0:
            print(f"Reading doc {i}/{len(doc_ids)}: {xml_id}")
        file_path = os.path.join(base_dir, xml_id + ".xml")
        doc_text = read_and_extract(file_path)
        doc_dict[xml_id] = doc_text

    # 4) Combine into (text_a, text_b, label) samples
    train_samples = build_training_samples(qrels_list, query_dict, doc_dict)
    print("Total training samples:", len(train_samples))

    # 5) Create dataset & dataloader
    train_dataset = PairwiseTextDataset(train_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )


# Load pretrained model
    # bert_model_name = "bert-base-uncased"
    # tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Initialize BERT model
    # base_model = AutoModel.from_pretrained(bert_model_name)
    # classifier_head = ClassifierHead(input_dim=768, hidden_dim=256, num_labels=3)

    # Load pretrained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_directory = "/data/wangj47/script/annote/result/models/bert0124"  # Adjust the path
    checkpoint_file = 'bert_gpus_best_model_with_time.bin'
    base_model, classifier_head = load_local_bert_and_classifier(model_directory, checkpoint_file, device=device)
    
    base_model.to(device)
    classifier_head.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        list(base_model.parameters()) + list(classifier_head.parameters()),
        lr=3e-5,
        weight_decay=0.01
    )

# Just an example of the number of epochs/batch_size
    num_epochs = 3
    batch_size = 16
    # Optionally, define a scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))

    # ========== Training Loop ========== 
    for epoch in range(num_epochs):
        base_model.train()
        classifier_head.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            
            # Forward through base BERT
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim=768)
            # outputs.pooler_output: (batch_size, hidden_dim=768) or None
            # If pooler_output is None, use the [CLS] embedding:
            if outputs.pooler_output is not None:
                cls_embedding = outputs.pooler_output
            else:
                cls_embedding = outputs.last_hidden_state[:, 0, :]

            # Forward through custom classifier
            logits = classifier_head(cls_embedding)

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

    # ========== Example Evaluation ========== 
    # If you have a separate test set or dev set, you can replicate
    # the above logic with base_model.eval() & classifier_head.eval(), 
    # collect predictions, then compute accuracy or F1.
    model_path = "/data/wangj47/checkpoints/trail/0226classifier_ft_bert/"
    # Save model
    torch.save(base_model.state_dict(), model_path + "bert_base_model_classification.pt")
    torch.save(classifier_head.state_dict(), model_path + "classifier_head_classification.pt")
    print("Model saved.")

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))   
    print("inference")
    # inference
    # 1) Load the model
    # model_path = "/data/wangj47/checkpoints/trail/0226classifier_bert/"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # base_model, classifier_head = load_bert_and_classifier(
    #     model_path=model_path,
    #     device=device,
    #     classifier_cls=ClassifierHead,  # your class
    #     input_dim=768,
    #     hidden_dim=256,
    #     num_labels=3
    # )
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # or your saved tokenizer
    
    # 2) Inference loop
    
    result_path = "/data/wangj47/script/annote/annote_app/trail/dataset/trec_2021/"
    qrel_path = result_path + "qrels/test.tsv"
    qrel = {}
    with open(qrel_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        # Skip the first row (header)
        header = next(reader, None)
        for row in reader:
            # row looks like: ['trec-20211', 'NCT00002569', '1']
            query_id = row[0]        # e.g. 'trec-20211'
            corpus_id = row[1]       # e.g. 'NCT00002569'
            score = int(row[2])      # e.g. 1
            if query_id not in qrel:
                qrel[query_id] = {}
            qrel[query_id][corpus_id] = score
    
    output_path = result_path + 'retrieved_trials.json'
    # Write the dictionary to the file as JSON
    with open(output_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    output_path = result_path + 'retrieved_trials.json'
    # Write the dictionary to the file as JSON
    with open(output_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    result_data = {}
    topic_dict = {}
    for i in range(len(original_data)):
        pid = original_data[i]['patient_id']
        result_data[pid]={}
        topic_dict[pid] = original_data[i]['patient']
        tmp_data = original_data[i]
        for j in range(len(tmp_data['0'])):
            result_data[pid][tmp_data['0'][j]['NCTID']] = 0
        for j in range(len(tmp_data['1'])):
            result_data[pid][tmp_data['1'][j]['NCTID']] = 1
        for j in range(len(tmp_data['2'])):
            result_data[pid][tmp_data['2'][j]['NCTID']] = 2
    print("original result")
    metrics = ["mrr","ndcg@10","precision@10", "recall@10","ndcg@100","precision@100","recall@100", "r-precision","recall@1000"]
    results = evaluate(qrel, result_data, metrics)
    print(results) 
    
    trail_path = result_path + 'trial_info.json'
    with open(trail_path, 'r', encoding='utf-8') as f:
        trail_df = json.load(f)
    trial_dict = {}
    for key,val in enumerate(trail_df):
        tmp_data = trail_df[val]
        trial_dict[val] = tmp_data['brief_title'] + \
        tmp_data['inclusion_criteria']#+ tmp_data['brief_summary']
    
    # 1) Load the model
    
    # 2) Inference loop
    new_data = {}
    for topic_id, sub_dict in result_data.items():
        topic_text = topic_dict.get(topic_id, "")
        if not topic_text:
            continue
        new_data[topic_id] = {}
    
        for trial_id, old_value in sub_dict.items():
            trial_text = trial_dict.get(trial_id, "")
            if not trial_text:
                continue
            # Predict a label 0,1,2
            pred_label = predict_label(
                base_model, classifier_head, tokenizer, device,
                topic_text, trial_text
            )
            new_data[topic_id][trial_id] = pred_label
    
        # with open(output_json_path, 'w', encoding='utf-8') as f:
        #     json.dump(new_data, f, indent=2)
        # print(f"Similarity-based updated data saved to: {output_json_path}")
    
    from ranx import evaluate
    
    # metrics = ["mrr","ndcg@10","precision@10", "recall@10","ndcg@100","precision@100","recall@100", "r-precision","recall@1000"]
    results = evaluate(qrel, new_data, metrics)
    print(results)
    
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    # 2025-02-26 17:52:23
# original result
# {'mrr': 1.0, 'ndcg@10': 0.9993948403910857, 'precision@10': 1.0, 'recall@10': 0.10939231430788807, 'ndcg@100': 0.946461477818535, 'precision@100': 0.8336, 'recall@100': 0.6498655215936092, 'r-precision': 0.8301469068847344, 'recall@1000': 0.8301469068847344}
# 2025-02-26 17:52:24
# {'mrr': 0.5576904761904762, 'ndcg@10': 0.33933777650184077, 'precision@10': 0.49066666666666664, 'recall@10': 0.03631630170117061, 'ndcg@100': 0.5011065367731242, 'precision@100': 0.5716, 'recall@100': 0.4315676626279181, 'r-precision': 0.5451549852254659, 'recall@1000': 0.8301469068847344}
# Reading doc 0/3626: NCT00000408
# Reading doc 2000/3626: NCT02241044
# Total training samples: 3870
# /home/wangj47/.local/lib/python3.9/site-packages/transformers/optimization.py:640: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
#   warnings.warn(
# Epoch 1/3 - Training: 100%|██████████| 242/242 [00:22<00:00, 10.84it/s]
# Epoch 1/3, Train Loss: 0.9182
# Epoch 2/3 - Training: 100%|██████████| 242/242 [00:22<00:00, 10.89it/s]
# Epoch 2/3, Train Loss: 0.8445
# Epoch 3/3 - Training: 100%|██████████| 242/242 [00:22<00:00, 10.90it/s]
# Epoch 3/3, Train Loss: 0.7900
# Model saved.
# 2025-02-26 22:41:14
# inference
# original result
# {'mrr': 1.0, 'ndcg@10': 0.9993948403910857, 'precision@10': 1.0, 'recall@10': 0.10939231430788807, 'ndcg@100': 0.946461477818535, 'precision@100': 0.8336, 'recall@100': 0.6498655215936092, 'r-precision': 0.8301469068847344, 'recall@1000': 0.8301469068847344}
# {'mrr': 0.6468517225130901, 'ndcg@10': 0.3417323646449638, 'precision@10': 0.508, 'recall@10': 0.03159709395764077, 'ndcg@100': 0.35924242686997654, 'precision@100': 0.4225333333333333, 'recall@100': 0.31589443901778497, 'r-precision': 0.41841810992061457, 'recall@1000': 0.8301469068847344}
# 2025-02-26 22:44:07