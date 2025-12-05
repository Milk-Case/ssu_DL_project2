import argparse

import torch
from transformers import BertTokenizer
from train_bert import CustomBertClassifier
from utils import extract_utterances

def load_model(model_pth, bert_model_name, device):
    model = CustomBertClassifier(bert_model_name, num_classes=2)
    model.load_state_dict(torch.load(model_pth, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def extract_text(cha_file):
    utterances = extract_utterances(cha_file, speakers=['CHI'])
    merged_txt = " ".join([utt.clean_text for utt in utterances])
    
    return merged_txt

def predict(model, tokenizer, text, device, max_len=256):
    encoding = tokenizer(text,
                         max_len,
                         padding='max_length',
                         truncation=True,
                         return_tensors='pt')
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prob = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(prob, dim=1).item()
        
    label_map = {0: 'TD', 1: 'SLI'}
    label = label_map[pred_class]
    
    return label, prob.squeeze().cpu().tolist()

def main():
    parser = argparse.ArgumentParser(description='SLI/TD prediction for .cha files')
    parser.add_argument('--cha', type=str, required=True, help='.cha file path')
    
    args = parser.parse_args()
    
    cha_file = args.cha
    model_pth = 'bert_finetuned.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_name = 'bert-base-uncased'
    
    txt = extract_text(cha_file)
    
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    model = load_model(model_pth, bert_model_name, device)
    
    label, probs = predict(model, tokenizer, txt, device)
    
    print("\n===============================")
    print("🔰 PREDICTION RESULT")
    print("===============================")
    print(f"File: {cha_file}")
    print(f"Predicted Label: {label}")
    print(f"Probabilities (TD, SLI): {probs}")
    print("===============================")

if __name__=='__main__':
    main()