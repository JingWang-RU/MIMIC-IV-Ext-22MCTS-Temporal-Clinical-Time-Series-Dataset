import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel, AdamW

# ====== Data Processing ======
def get_samples_data(data):
    """Extracts question-context pairs and labels from JSON data."""
    all_samples = []
    for key, value in data.items():
        sent_a = value["QUESTION"]
        sent_b = ' '.join(value["CONTEXTS"])
        decision = value["final_decision"]
        if decision == 'yes':
            label = 0
        elif decision == 'no':
            label = 1
        elif decision == 'maybe':
            label = 2
        else:
            continue  
        all_samples.append((sent_a, sent_b, label))
    return all_samples

def collate_fn(batch):
    """Tokenizes input text and prepares tensors for BERT."""
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

# ====== Dataset Class ======
class PairwiseTextDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ====== Improved Classifier Head ======
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(ClassifierHead, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)  # Normalize activations
        self.activation = nn.GELU()  # Better than ReLU for NLP
        self.dropout = nn.Dropout(0.3)  # Reduces overfitting
        self.linear2 = nn.Linear(hidden_dim, num_labels)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ====== Main Training Loop ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
base_model = AutoModel.from_pretrained(bert_model_name).to(device)
classifier_head = ClassifierHead(input_dim=768, hidden_dim=256, num_labels=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(
    list(base_model.parameters()) + list(classifier_head.parameters()),
    lr=3e-5,
    weight_decay=0.01  # Helps prevent overfitting
)

num_epochs = 3
batch_size = 16

for fold in range(10):  # Iterate over pqal_fold0 to pqal_fold9
    pubmedqa_td = f'/data/wangj47/datasets/pubmedqa/data/pqal_fold{fold}'

    # Load test set
    with open('/data/wangj47/datasets/pubmedqa/data/test_set.json') as f:
        data = json.load(f)
    all_test_samples = get_samples_data(data)

    # Load train set for the fold
    train_set_path = os.path.join(pubmedqa_td, 'train_set.json')
    with open(train_set_path) as f:
        data = json.load(f)
    all_train_samples = get_samples_data(data)

    train_dataset = PairwiseTextDataset(all_train_samples)
    test_dataset = PairwiseTextDataset(all_test_samples)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader))

    print(f"\n===== Training on Fold {fold} =====")
    for epoch in range(num_epochs):
        base_model.train()
        classifier_head.train()
        total_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Fold {fold} - Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

            logits = classifier_head(cls_embedding)

            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Fold {fold} - Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # ===== Evaluation =====
        base_model.eval()
        classifier_head.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Fold {fold} - Epoch {epoch+1}/{num_epochs} - Testing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

                logits = classifier_head(cls_embedding)
                loss = criterion(logits, labels)
                test_loss += loss.item()

                _, preds = torch.max(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_dataloader)
        acc = accuracy_score(all_labels, all_preds)
        maf = f1_score(all_labels, all_preds, average='macro')

        print(f"Fold {fold} - Epoch {epoch+1}, Test Loss: {avg_test_loss:.4f}, Accuracy: {acc:.4f}, Macro F1: {maf:.4f}")

    # Save model for each fold
    torch.save(base_model.state_dict(), f"bert_base_model_fold{fold}.pt")
    torch.save(classifier_head.state_dict(), f"classifier_head_fold{fold}.pt")
