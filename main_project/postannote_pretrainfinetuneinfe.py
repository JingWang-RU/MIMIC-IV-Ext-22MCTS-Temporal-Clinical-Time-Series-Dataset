import os
import pandas as pd
from tqdm import tqdm
import re
import multiprocessing as mp
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sentence_transformers import InputExample
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch.optim as optim
import pickle
# ===== 6. Define Dataset Class =====
class EventPairDataset(Dataset):
    def __init__(self, event_pairs, tokenizer, max_length=128):
        """
        Args:
            event_pairs (list of tuples): Each tuple contains (question, context, label, time_bin)
            tokenizer (transformers tokenizer): Tokenizer to process text data
            max_length (int): Maximum sequence length for tokenizer
        """
        self.event_pairs = event_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.event_pairs)

    def __getitem__(self, idx):
        question, context, label, time_bin = self.event_pairs[idx]
        combined_text = f"{question} [SEP] {context}"
        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),        # [max_length]
            'attention_mask': encoding['attention_mask'].flatten(),  # [max_length]
            'time_bin': torch.tensor(time_bin, dtype=torch.long),  # Scalar
            'labels': torch.tensor(label, dtype=torch.long)        # Scalar
        }

# ===== 7. Define the Model =====
class BertWithTimeClassifier(nn.Module):
    def __init__(self, bert_model, num_time_bins, time_embedding_dim, hidden_dim, num_labels):
        super(BertWithTimeClassifier, self).__init__()
        self.bert = bert_model
        self.time_embedding = nn.Embedding(num_time_bins, time_embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + time_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, time_bin):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        time_emb = self.time_embedding(time_bin)  # [batch_size, time_embedding_dim]
        combined = torch.cat((cls_output, time_emb), dim=1)  # [batch_size, hidden_size + time_embedding_dim]
        logits = self.classifier(combined)  # [batch_size, num_labels]
        return logits

# ===== 8. Training and Evaluation Functions =====
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        time_bin = batch['time_bin'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, time_bin=time_bin)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    maf = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, maf

def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            time_bin = batch['time_bin'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, time_bin=time_bin)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    maf = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, maf

# ===== 9. Main Execution =====
if __name__ == "__main__":
    # Define the data directory
    result_directory = "/*/script/annote/result/train_clean"
    figure_directory = "/*/script/annote/result/figures"
    model_directory = "/*/script/annote/result/models/bert_batch2046"
    os.makedirs(model_directory, exist_ok=True) 
    with open(os.path.join(result_directory,'event_pairs.pkl'), 'rb') as f:
        event_pairs = pickle.load(f)

    master_df = pd.read_csv(os.path.join(result_directory,'consolidated_events.csv'))
    # Display sample event pairs
    # print("\n Sample Event Pairs:")
    # for idx, (question, context, label, time_bin) in enumerate(event_pairs[:5]):
    #     print(f" Pair {idx + 1}:")
    #     print(f"  Question: {question}")
    #     print(f"  Context: {context}")
    #     print(f"  Label: {inverse_label_mapping[label]}")
    #     print(f"  Time Bin: {time_bin}\n")

    # Create InputExample instances (if using SentenceTransformers)
    input_examples = [
        InputExample(texts=[question, context], label=label)
        for question, context, label, time_bin in event_pairs
    ]
    print(f"Total InputExamples: {len(input_examples)}")

    # Alternatively, create a PyTorch Dataset
    print("\n Creating PyTorch Dataset...")
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Split event_pairs into train, val, test
    from sklearn.model_selection import train_test_split
    train_pairs, temp_pairs = train_test_split(event_pairs, test_size=0.2, random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=42)

    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")

    # Create Datasets
    train_dataset = EventPairDataset(train_pairs, tokenizer, max_length=128)
    val_dataset = EventPairDataset(val_pairs, tokenizer, max_length=128)
    test_dataset = EventPairDataset(test_pairs, tokenizer, max_length=128)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=2046, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2046, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2046, shuffle=False)

    label_mapping = {
    'No Correlation': 0,
    'Positive Correlation': 1,  # Potential Outcome
    'Negative Correlation': 2   # Reason
    }

    inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    # Initialize the model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    num_time_bins = master_df['time_bin'].nunique()  # e.g., 9
    time_embedding_dim = 50
    hidden_dim = 256
    num_labels = len(label_mapping)  # 3

    model = BertWithTimeClassifier(
        bert_model=bert_model,
        num_time_bins=num_time_bins,
        time_embedding_dim=time_embedding_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    # Training parameters
    num_epochs = 5

    # Training Loop
    best_val_f1 = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"  Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Macro F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = eval_model(model, val_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Macro F1: {val_f1:.4f}\n")

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(model_directory,'best_model_with_time.bin'))
            print(f"  Best model updated (Macro F1: {best_val_f1:.4f})\n")

    # ===== 10. Evaluation on Test Set =====
    print("Evaluating on Test Set...")
    state_dict = torch.load(os.path.join(model_directory, 'bert_gpus_best_model_with_time.bin'), map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Load the state_dict
    model.load_state_dict(new_state_dict)
    model.eval()
    test_loss, test_acc, test_f1 = eval_model(model, test_loader, criterion, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}")

    # ===== 11. Generate Confusion Matrix =====
    all_test_preds = []
    all_test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Confusion Matrix"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            time_bin = batch['time_bin'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, time_bin=time_bin)
            preds = torch.argmax(outputs, dim=1)

            all_test_preds.extend(preds.detach().cpu().numpy())
            all_test_labels.extend(labels.detach().cpu().numpy())

    # Create confusion matrix
    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    
    plt.savefig(os.path.join(figure_directory,'bert_figure.png'))
    plt.show()

    # ===== 12. Saving the Fine-Tuned Model =====
    torch.save(model.state_dict(), os.path.join(model_directory,'fine_tuned_bert_with_time_final.bin'))
    print("Fine-tuned model saved as 'fine_tuned_bert_with_time_final.bin'.")
