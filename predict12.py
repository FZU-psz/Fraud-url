import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm

# Set log level to ERROR
import logging
logging.basicConfig(level=logging.INFO)


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读数据
data_path = '../data/valid_12.csv'
data = pd.read_csv(data_path)
input_data = list(data['url'])
max_len = 0
for item in input_data:
    if len(item) > max_len:
        max_len = len(item)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(input_data, padding=True, truncation=True, max_length=max_len)
# DataLoader
test_dataset = TestDataset(inputs)
test_dataloader = DataLoader(test_dataset, batch_size=128)

# Load model
 # 十二分类
checkpoint_path = "/data/zqy/URL_Classify/model_trained/bert-base-uncased_1_2_3_4_5_6_7_8_9_10_11_12.tar"
checkpoint = torch.load(checkpoint_path)
config = BertConfig.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-config.json', num_labels=12)
model = BertForSequenceClassification.from_pretrained('/data/zqy/URL_Classify/code/pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Predict!
data_iter = tqdm(test_dataloader)
data_iter.set_description("Predict!")
predictions = []

for batch in data_iter:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1)+1
    predictions.extend(pred_class.cpu().numpy())

# Add predictions to the dataframe and write to CSV
data['label'] = predictions
data.to_csv('../data/valid_12.csv', index=False)

for i in range(12):
    logging.info(f"Class {i+1} count: {predictions.count(i+1)}")