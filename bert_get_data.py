from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

# 初始化 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

class MyDataset(Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=35,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        self.labels = torch.tensor(df['label'].values, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.texts[idx]
        return {
            'input_ids': item['input_ids'].squeeze(0),
            'attention_mask': item['attention_mask'].squeeze(0),
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 10)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用 [CLS] 標記的輸出
        dropout_output = self.dropout(pooled_output)
        logits = self.linear(dropout_output)
        return logits

def GenerateData(mode):
    train_data_path = './THUCNews/data/train.txt'
    dev_data_path = './THUCNews/data/dev.txt'
    test_data_path = './THUCNews/data/test.txt'

    train_df = pd.read_csv(train_data_path, sep='\t', header=None, names=['text', 'label'], dtype=str)
    dev_df = pd.read_csv(dev_data_path, sep='\t', header=None, names=['text', 'label'], dtype=str)
    test_df = pd.read_csv(test_data_path, sep='\t', header=None, names=['text', 'label'], dtype=str)

    train_df['label'] = train_df['label'].astype(int)
    dev_df['label'] = dev_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    train_dataset = MyDataset(train_df)
    dev_dataset = MyDataset(dev_df)
    test_dataset = MyDataset(test_df)
    
    if mode == 'train':
        return train_dataset
    elif mode == 'val':
        return dev_dataset
    elif mode == 'test':
        return test_dataset
    else:
        raise ValueError("Mode must be 'train', 'val', or 'test'")
