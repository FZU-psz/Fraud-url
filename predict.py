import torch
from transformers import BertTokenizer,BertModel,BertConfig,AlbertModel
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader, TensorDataset
import mymodel
from tqdm import *
import random
import numpy as np
import pandas as pd
import logging
import tarfile
from transformers import logging as log
from NewsDataset import NewsDataset
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def load_model_and_tokenizer(checkpoint_path,num_class,device):
    log.set_verbosity_error()
    checkpoint = torch.load(checkpoint_path)
    config = BertConfig.from_pretrained('./pretrained_model/bert-base-uncased-config.json', num_labels=num_class)
    model = BertForSequenceClassification.from_pretrained('./pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
    model.load_state_dict(checkpoint['model_state_dict'])

    best_acc = checkpoint['best_acc']
    
    logger.info("模型：%s,Best_ACC:%.4f",checkpoint_path,best_acc)
    # 其中，checkpoint_path 是之前保存 checkpoint 的路径，
    # model_state_dict 和 optimizer_state_dict 是之前保存的模型和优化器的状态字典，
    # best_acc 和 loss 是之前记录的最佳准确率和损失值，seed 是之前记录的随机种子。
    # 使用 load_state_dict 方法可以将保存的模型和优化器状态加载到当前模型和优化器中。

    return model

def create_dataloader(inputs,labels,batch_size):
    #将数据集包装成torch的Dataset形式
    test_dataset = NewsDataset(inputs, labels)
    # 单个读取到批量读取
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    batch = next(iter(test_dataloader))
    print(batch["input_ids"].shape)
    return test_dataloader



def map_labels(predictions, class_mapping):
    return [class_mapping[pred] for pred in predictions]

def hierarchical_predict(model1, model2, model3, model4,input_data,mask,device,threshold1, threshold2):
    predictions = []
    
    # Stage 1: Predict class 0 vs others

    model1.to(device)
    outputs = model1(input_data,attention_mask=mask).logits
    probs = torch.softmax(outputs, dim=1)
    pred_class1 = torch.argmax(probs, dim=1).tolist()

    input_data.unsqueeze_(1)
    mask.unsqueeze_(1)
    
    for idx, pred in enumerate(pred_class1):
        if pred == 0:
            predictions.append(0)
        else:
            # Stage 2: Classify among the 3 major classes
            # inputs = tokenizer(dataset[idx], return_tensors='pt', padding=True, truncation=True,max_length=max_len)
            model2.to(device)

            outputs = model2(input_data[idx],attention_mask=mask[idx]).logits
            probs = torch.softmax(outputs, dim=1)
            pred_class2 = torch.argmax(probs, dim=1).tolist()[0]
            
            if probs[0, pred_class2] < threshold1:
                # Stage 3: Classify within the 2nd major class
                # inputs = tokenizer(dataset[idx], return_tensors='pt', padding=True, truncation=True,max_length=max_len)
                model3.to(device)

                outputs = model3(input_data[idx],attention_mask=mask[idx]).logits
                probs = torch.softmax(outputs, dim=1)
                pred_class3 = torch.argmax(probs, dim=1).tolist()[0]
                
                if probs[0, pred_class3] < threshold2:
                    # Stage 4: Classify within the 3rd major class
                    # inputs = tokenizer(dataset[idx], return_tensors='pt', padding=True, truncation=True,max_length=max_len)

                    model4.to(device)

                    outputs = model4(input_data[idx],attention_mask=mask[idx]).logits
                    probs = torch.softmax(outputs, dim=1)
                    pred_class4 = torch.argmax(probs, dim=1).tolist()[0]

                    
                    # def map_labels(predictions, class_mapping):
                    #     return [class_mapping[pred] for pred in predictions]
                    
                    class_mapping = {0: 1, 1: 3, 2: 5, 3: 7, 4: 12}
                    predictions.append(map_labels([pred_class4], class_mapping)[0])

                else:
                    # Map the labels of the 2nd major class
                    class_mapping = {0: 2, 1: 4, 2: 6}
                    predictions.append(map_labels([pred_class3], class_mapping)[0])
            else:
                # Map the labels of the 1st major class
                class_mapping = {0: 8, 1: 9, 2: 10, 3: 11}
                predictions.append(map_labels([pred_class2], class_mapping)[0])

               
    return predictions

def find_best_thresholds(model1, model2, model3, model4,test_dataloader,device,num_trials=10):
    best_threshold1, best_threshold2 = 0, 0
    best_accuracy = 0
    total_iter = len(test_dataloader)
    batch_iter = tqdm(test_dataloader)
    with torch.no_grad():
        for i in range(num_trials):
            logger.info("第%d轮阈值搜索",i)
            threshold1 = torch.rand(1).item() * 0.1 + 0.588
            threshold2 = torch.rand(1).item() * 0.1 + 0.620
            for batch in tqdm(test_dataloader,desc="Predicting"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                # with autocast():
                predictions = hierarchical_predict(model1, model2, model3, model4,input_ids,attention_mask,device,threshold1, threshold2)
                accuracy = accuracy_score(labels, predictions)
            # torch.cuda.empty_cache()
            logger.info("第%d轮搜索,Acc:%.4f,threshold1:%.4f,threshold2:%.4f",i,accuracy,threshold1,threshold2)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold1 = threshold1
                best_threshold2 = threshold2
            
    return best_threshold1, best_threshold2, best_accuracy

if __name__ == "__main__":

    setup_seed(42)
    checkpoint_path_list = ["/data/zqy/URL_Classify/model_trained/bert-base-uncased_0.tar","/data/zqy/URL_Classify/model_trained/bert-base-uncased_8_9_10_11.tar","/data/zqy/URL_Classify/model_trained/bert-base-uncased_2_4_6.tar","/data/zqy/URL_Classify/model_trained/bert-base-uncased_1_3_5_7_12.tar"]
    num_class_list = [2,4,3,5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models and tokenizer
    models = [load_model_and_tokenizer(checkpoint_path_list[i],num_class_list[i],device) for i in trange(0, 4,desc="Loading model")]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define your dataset and labels here
    data_path = '../data/newdata.csv'
    data = pd.read_csv(data_path)
    grouped = data.groupby('label')

    threshold = 2000
    data=[]
    # 遍历每个分组，将分组数据保存在一个新的DataFrame中
    for label, group in grouped:
        # 构造新的DataFrame
        if len(group['label']) < 15 :
            group = group.sample(n=15, replace=True)
            data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
        elif len(group['label']) > threshold:
            group = group.sample(n=threshold, replace=True)
            data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
        else :
            data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))

    merged_df=pd.DataFrame()
    for class_data in data:
        merged_df = pd.concat([merged_df,class_data], axis=0)

    print(merged_df['label'].value_counts())

    max_len = 0
    max_url = ""
    for item in merged_df['url']:
        if len(item) > max_len:
            max_len = len(item)
            max_url = item

    logger.info("max_url:%s,max_len:%d",max_url,max_len)

    input_data=list(merged_df['url'])
    labels=list(merged_df['label'])

    inputs = tokenizer(input_data, padding=True, truncation=True,max_length=max_len)

    test_dataloader = create_dataloader(inputs,labels,batch_size=128)

    # Find the best thresholds and accuracy
    threshold1, threshold2, accuracy = find_best_thresholds(*models,test_dataloader,device)
    print(f"Best thresholds: {threshold1:.3f}, {threshold2:.3f}, Accuracy: {accuracy:.3f}")
