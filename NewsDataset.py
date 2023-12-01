from transformers import BertTokenizer,BertModel,BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
import pandas as pd
import numpy as np
from tqdm import *
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    #这里的idx是为了让后面的DataLoader成批处理成迭代器，按idx映射到对应数据
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.tensor(self.encodings['input_ids'][idx])
        item['attention_mask'] = torch.tensor(self.encodings['attention_mask'][idx])

        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    #数据集长度。通过len(这个实例对象)，可以查看长度
    def __len__(self):
        return len(self.labels)

class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    #这里的idx是为了让后面的DataLoader成批处理成迭代器，按idx映射到对应数据
    def __getitem__(self, idx):
        item = {}
        item['input_ids'] = torch.tensor(self.encodings['input_ids'][idx])
        item['attention_mask'] = torch.tensor(self.encodings['attention_mask'][idx])

        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    #数据集长度。通过len(这个实例对象)，可以查看长度
    def __len__(self):
        return len(self.encodings['input_ids'])
