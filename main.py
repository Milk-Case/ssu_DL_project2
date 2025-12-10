import argparse
import sys
import pickle

import numpy as np
from prepare_dataset import prepare_dataset
from utils import extract_utterances

DEBUG = True

class Vectorizer:
    def __init__(self):
        self.voca = {}
    
    def fit(self, texts):
        voca_set = set()
        
        for txt in texts:
            tokens = txt.lower().split()
            voca_set.update(tokens)
            
        self.voca = {word: idx for idx, word in enumerate(sorted(voca_set))}
        if DEBUG:
            print(f"Vocabulary size: {len(self.voca)}")
    
    def transform(self, texts):
        vec = np.zeros((len(texts), len(self.voca)))
        
        for i, txt in enumerate(texts):
            tokens = txt.lower().split()
            
            for tok in tokens:
                if tok in self.voca:
                    vec[i][self.voca[tok]] += 1
                    
            vec[i] = vec[i] / np.sum(vec[i])
             
        return vec
    
class LogisticRegression:
    def __init__(self, lr=0.01, epochs=20, l2=0.01):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
    
    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))
    
    def init_weights(self, n_features):
        self.W = np.random.normal(0, 0.01, size=n_features)
        self.b = 0
        
    def compute_loss(self, y, y_pred):
        eps = 1e-10 
        
        ce_loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
        l2_loss = self.l2 * np.sum(self.W ** 2)
        
        return ce_loss + l2_loss
    
    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        
        n_samples, n_features = x.shape
        self.init_weights(n_features)
        
        for epoch in range(1, self.epochs + 1):
            z = np.dot(x, self.W) + self.b
            y_pred = self.sigmoid(z)
            
            dw = (np.dot(x.T, (y_pred - y)) / n_samples) + self.l2 * self.W
            db = np.mean(y_pred - y)
            
            self.W -= self.lr * dw
            self.b -= self.lr * db
            
            if DEBUG and (epoch == 1 or epoch % 5 == 0):
                loss = self.compute_loss(y, y_pred)
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f}")
                
    def pred(self, x):
        z = np.dot(x, self.W) + self.b
        prob = self.sigmoid(z)
        
        return prob, (prob >= 0.5).astype(int)
    
def encoding(y):
    return np.array([1 if label == "SLI" else 0 for label in y])
    
def eval(model, x, y):
    _, preds = model.pred(x)
    acc = np.mean(preds == y)
    
    return acc

def task_train():
    dataset = prepare_dataset()
    
    x_train, y_train = dataset['train']
    x_dev, y_dev = dataset['dev']
    
    y_train = encoding(y_train)
    y_dev = encoding(y_dev)
    
    vec = Vectorizer()
    vec.fit(x_train)
    
    x_train = vec.transform(x_train)
    x_dev = vec.transform(x_dev)
    
    if DEBUG:
        print("Shapes:", x_train.shape, x_dev.shape)
        
    
    model = LogisticRegression(epochs=50)
    model.fit(x_train, y_train)
    
    dev_acc = eval(model, x_dev, y_dev)
    
    if DEBUG:
        print(f"\nDev Accuracy:  {dev_acc:.4f}")
        
    with open("vec.pkl", "wb") as f:
        pickle.dump(vec.voca, f)
        
    np.savez('mod_weights.npz', W=model.W, b=model.b)
    
    if DEBUG:
        print("model saved!")
        
def task_test():
    dataset = prepare_dataset()
    x_test, y_test = dataset['test']
    
    y_test = encoding(y_test)
    
    with open('vec.pkl', 'rb') as f:
        voca = pickle.load(f)
    vec = Vectorizer()
    vec.voca = voca
    
    x_test = vec.transform(x_test)
    
    data = np.load('mod_weights.npz')
    model = LogisticRegression()
    model.W = data['W']
    model.b = data['b']
    
    test_acc = eval(model, x_test, y_test)
    if DEBUG:
        print(f"Test Accuracy: {test_acc:.4f}")
        
def task_pred(cha_pth):
    with open('vec.pkl', 'rb') as f:
        voca = pickle.load(f)
    vec = Vectorizer()
    vec.voca = voca
        
    data = np.load('mod_weights.npz')
    model = LogisticRegression()
    model.W = data['W']
    model.b = data['b']
    
    utterances = extract_utterances(cha_pth, speakers=['CHI'])
    text = " ".join([utt.clean_text for utt in utterances])
    
    x = vec.transform([text])
    
    prob, pred = model.pred(x)
    prob = prob[0]
    pred = pred[0]
    
    label = 'SLI' if pred == 1 else 'TD'
    
    if DEBUG:
        print("\nPrediction")
        print("--------------------")
        print(f"File: {cha_pth}")
        print(f"Predicted Label: {label}")
        print(f"SLI Probability: {prob:.4f}")
        print(f"TD Probability:  {1 - prob:.4f}")
        
def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--task", required=True, type=str,
                   choices=["train", "test", "pred"], help="task")
    p.add_argument("--input", type=str, default=None)
    
    return p.parse_args()
        
def main():
    args = parse_args()
    
    if args.task == "train":
        task_train()
    elif args.task == "test":
        task_test()
    elif args.task == "pred":
        if args.input is None:
            print("insert cha file to pred")
            return 1
        task_pred(args.input)
    
    return 0
      
if __name__ == '__main__':
    sys.exit(main())