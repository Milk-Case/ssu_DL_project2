import pandas as pd
from utils import extract_utterances

DEBUG = True

def load_extract(path):
    df = pd.read_csv(path)
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        cha_file = row['filename']
        group = row['group']
        
        utterances = extract_utterances(cha_file, speakers=['CHI'])
        
        merged_text = " ".join([utt.clean_text for utt in utterances])
        
        texts.append(merged_text)
        labels.append(group)
        
    return texts, labels

def prepare_dataset():
    train_csv =  'gillam_train.csv'
    dev_csv = 'gillam_dev.csv'
    test_csv = 'gillam_test.csv'
    
    x_train, y_train = load_extract(train_csv)
    x_dev, y_dev = load_extract(dev_csv)
    x_test, y_test = load_extract(test_csv)
    
    if DEBUG:
        print(f"Train: {len(x_train)} samples")
        print(f"Dev:   {len(x_dev)} samples")
        print(f"Test:  {len(x_test)} samples")
    
    return {
        "train": (x_train, y_train),
        "dev": (x_dev, y_dev),
        "test": (x_test, y_test)
    }

if __name__ == '__main__':
    prepare_dataset()
