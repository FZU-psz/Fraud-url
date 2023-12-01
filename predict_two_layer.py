import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import os
import sys
import pickle
# Set log level to ERROR
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def run_model_0(input_ids, attention_mask, model, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1)
    return pred_class.cpu()


def run_model_1(input_ids, attention_mask, model, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    outputs = model(input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1) + 1
    return pred_class.cpu()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def process_data_and_run_model_0(data, q, model_0, device,max_len):
    logging.info("==============================Tokenizing===================================")
    inputs = tokenizer(data, padding=True, truncation=True,max_length=max_len)
    test_dataset = TestDataset(inputs)
    test_dataloader = DataLoader(test_dataset, batch_size=128)
    
    data_iter = test_dataloader
    # data_iter = tqdm(test_dataloader)
    # data_iter.set_description("Processing data and running model 0...")
    logging.info("==============================Predicting====================================")
    for batch in data_iter:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pred_class_cpu = run_model_0(input_ids, attention_mask, model_0, device)
        serialized_batch = pickle.dumps(batch)
        serialized_pred_class_cpu = pickle.dumps(pred_class_cpu)
        q.put((serialized_batch, serialized_pred_class_cpu))
    q.put("done")

def process_model_1_and_output_results(q, model_1, device):
    predictions = []

    while True:
        item = q.get()
        if item == "done":  # 如果获取到特殊任务，表示所有任务已经执行完毕
            break
        batch, pred_class_cpu_0 = item
        batch = pickle.loads(batch)
        pred_class_cpu_0 = pickle.loads(pred_class_cpu_0)
        mask = pred_class_cpu_0 != 0
        if mask.any():
            input_ids_1 = batch['input_ids'][mask]
            attention_mask_1 = batch['attention_mask'][mask]
            pred_class_cpu_12 = run_model_1(input_ids_1, attention_mask_1, model_1, device)

            pred_class_cpu_0[mask] = pred_class_cpu_12
        
        for i, (input_id, label) in enumerate(zip(batch['input_ids'], pred_class_cpu_0.numpy())):
            url_tensor = batch['input_ids'][i]
            url = tokenizer.decode(url_tensor, skip_special_tokens=True)
            logging.info(f"预测URL:{url},预测标签类别:{int(label)}")
        predictions.extend(pred_class_cpu_0.numpy())

    # Add predictions to the dataframe and write to CSV
    # data['label'] = predictions
    # data.to_csv('../data/invalid_data_copy.csv', index=False)

    for i in range(13):
        logging.info(f"Class {i} count: {predictions.count(i)}")
    logging.info("==============================Predict Finished==============================")


def main():
    device_0 = torch.device("cuda:0")
    device_1 = torch.device("cuda:1")
    logging.info("==============================流水线预测万级预测集============================")
    logging.info("==============================Loading Models================================")
    # Load model
    # 二分类
    checkpoint_path = "/data/zqy/URL_Classify/model_trained/bert-base-uncased_0.tar"
    checkpoint = torch.load(checkpoint_path)
    config = BertConfig.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-config.json', num_labels=2)
    model_0 = BertForSequenceClassification.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
    model_0.load_state_dict(checkpoint['model_state_dict'])
    model_0.to(device_0)
    model_0.eval()

    # 十二分类
    checkpoint_path = "/data/zqy/URL_Classify/model_trained/bert-base-uncased_1_2_3_4_5_6_7_8_9_10_11_12.tar"
    checkpoint = torch.load(checkpoint_path)
    config = BertConfig.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-config.json', num_labels=12)
    model_1 = BertForSequenceClassification.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
    model_1.load_state_dict(checkpoint['model_state_dict'])
    model_1.to(device_1)
    model_1.eval()
    
  

    logging.info("==============================Loading data==================================")
     # Load data
    data_path = '../data/valid_12.csv'
    # num_lines = sum(1 for _ in open(data_path)) - 1  # 获取文件总行数（不包括header行）
    # skip_rows = sorted(random.sample(range(1, num_lines+1), num_lines-100000))  # 随机选择要跳过的行数
    # data = pd.read_csv(data_path, skiprows=skip_rows)
    data = pd.read_csv(data_path)
    input_data = list(data['url'])
    max_len = 0
    for item in input_data:
        if len(item) > max_len:
            max_len = len(item)

    # Initialize the queue
    q =Queue()


    # Start the processes
    p1 = mp.Process(target=process_data_and_run_model_0, args=(input_data, q, model_0, device_0,max_len))
    p1.start()
    p2 = mp.Process(target=process_model_1_and_output_results, args=(q, model_1, device_1))
    p2.start()

    # Join the processes
    p1.join()
    p2.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    main()

