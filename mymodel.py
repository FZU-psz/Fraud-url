from transformers import BertTokenizer,BertModel,BertConfig,AlbertModel
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
import pandas as pd
import numpy as np
from tqdm import *
import torch.nn.functional as F 
import time

# Bert模型
class bert_model(nn.Module):
    def __init__(self, num_class,freeze_bert=True, hidden_size=768):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.update({'output_hidden_states':True})
        self.bert = BertModel.from_pretrained("bert-base-uncased",config=config)
        self.fc = nn.Linear(hidden_size*4,num_class)

        #是否冻结bert，不让其参数更新
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        all_hidden_states = torch.stack(outputs[2])  #因为输出的是所有层的输出，是元组保存的，所以转成矩阵
        concat_last_4layers = torch.cat((all_hidden_states[-1],   #取最后4层的输出
                                         all_hidden_states[-2],
                                         all_hidden_states[-3],
                                         all_hidden_states[-4]), dim=-1)

        cls_concat = concat_last_4layers[:,0,:]   #取 [CLS] 这个token对应的经过最后4层concat后的输出
        result = self.fc(cls_concat)

        return result

class FineTunedAlbert(nn.Module):
    def __init__(self, num_classes, freeze_albert=True):
        super(FineTunedAlbert, self).__init__()
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        if freeze_albert:
            for param in self.albert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.albert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.albert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        # print(f'output_shape:{pooled_output.shape}')
        # print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # time.sleep(0.5)
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


def load_pretrained_model(model_name,num_class,freeze_bool):
    if model_name == "bert-base-uncased":
        # model_name = 'bert-base-uncased'
        config = BertConfig.from_pretrained('./pretrained_model/bert-base-uncased-config.json', num_labels=num_class)
        model = BertForSequenceClassification.from_pretrained('./pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
        return model

    if model_name == "albert-base-v2":
        return FineTunedAlbert(num_class,freeze_bool)
