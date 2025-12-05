import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from prepare_dataset import prepare_dataset

class CustomTextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_length):
        self.texts = texts
        self.labels = [1 if y == 'SLI' else 0 for y in labels]
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length = self.max_seq_length,
            padding = 'max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return{
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }

class CustomBertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(CustomBertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(512),
            nn.Linear(512,256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        pooled_output = torch.mean(hidden_states, dim=1)
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        
        return logits
        
def train_model(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader, desc='Train'):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate_model(model, data_loader, device):
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            
            preds.extend(predictions.cpu().tolist())
            actuals.extend(labels.cpu().tolist())
            
    acc = accuracy_score(actuals, preds)
    report = classification_report(actuals,preds, target_names=['TD', 'SLI'])    
    
    return acc, report     
        
def main():
    dataset = prepare_dataset()
    x_train, y_train = dataset['train']
    x_dev, y_dev = dataset['dev']
    
    bert_model_name = 'bert-base-uncased'
    
    max_seq_len = 256
    batch_size = 16
    epochs = 10
    lr = 1e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    train_dataset = CustomTextClassificationDataset(x_train, y_train, tokenizer, max_seq_len)
    dev_dataset = CustomTextClassificationDataset(x_dev, y_dev, tokenizer, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size)
    
    model = CustomBertClassifier(bert_model_name, num_classes=2).to(device)
    
    optimizer = AdamW(model.parameters(), lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    
    best_acc = 0
    save_path = 'bert_finetuned.pth'
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_model(model, train_loader, optimizer, scheduler, device)

        acc, report = evaluate_model(model, dev_loader, device)
        print("Validation Accuracy:", round(acc, 4))
        print(report)

        if acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = acc
            print("Saved new best model:", save_path)

if __name__=='__main__':
    main()