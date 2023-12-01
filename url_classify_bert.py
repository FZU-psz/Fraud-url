from transformers import BertTokenizer,BertModel,BertConfig,AlbertTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
import random
import numpy as np
import os
from tqdm import *
import argparse
import time
from sklearn.model_selection import train_test_split
import pandas as pd
from NewsDataset import NewsDataset
import mymodel
from torch.cuda.amp import autocast 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import logging
from transformers import logging as log
logger = logging.getLogger(__name__)



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

 #二分类数据集
def data_process(temp,threshold,data):
    data1 = []
    data1.append(data[temp].sample(n=threshold, replace=True))
    data1[0]['label']=0
    data1.append(pd.concat([data[i].sample(n=threshold//12, replace=True) for i in range(12) if(i!=temp)],axis=0))
    #还需要调整一下
    data1[1]['label']=1
    merged_df = pd.concat([data1[0], data1[1]], axis=0)
    
    X=list(merged_df['url'])
    Y=list(merged_df['label'])
    return X,Y

def data_pre(tokenizer):
    # 读入数据
    # data = pd.read_csv('../data/newdata.csv')
    # print(data['label'].value_counts())
    data1=pd.read_csv('../data/train1.csv')
    data2=pd.read_csv('../data/train2.csv')
    data = pd.concat([data1, data2], axis=0)

    # data = pd.read_csv('../data/train_invalid_data.csv')
    print(data['label'].value_counts())

    # 定义阈值
    threshold = 5000
    smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
    max_len = 0
    max_url = ""
    


    # 分类标签
    if args.num_class == 2:
        class_label=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    elif args.num_class == 3:
        class_label=[2,4,6]
    elif args.num_class == 4:
        class_label=[8,9,10,11]
    elif args.num_class == 5:
        class_label=[1,3,5,7,12]
    elif args.num_class == 12:
        class_label=[1,2,3,4,5,6,7,8,9,10,11,12]
    elif args.num_class == 13:
        class_label=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    logger.info("class_label: %s", ', '.join(map(str, class_label)))


    # 根据label的值将数据拆分成不同的组
    grouped = data.groupby('label')

    data=[]
    # 遍历每个分组，将分组数据保存在一个新的DataFrame中
    i = 0
    for label, group in grouped:
        # 构造新的DataFrame
        if label in class_label:
            # if label == 0:
            #     group['label']=0
            group['label']=label
            # else :
            #     group['label']=1
            # if label == 4:
            #     group['label']=1
            # if label == 6:
            #     group['label']=2
            # if label == 11:
            #     group['label']=3
            # if label == 12:
            #     group['label']=4

            # 简单进行上下采样
            if len(group['label']) < 15 :
                group = group.sample(n=30, replace=True)
                data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
            elif len(group['label']) > threshold:
                group = group.sample(n=threshold, replace=True)
                data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
            else :
                data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
    
    merged_df=pd.DataFrame()
    for class_data in data:
        merged_df = pd.concat([merged_df,class_data], axis=0)

    for item in merged_df['url']:
        if len(item) > max_len:
            max_len = len(item)
            max_url = item

    X=list(merged_df['url'])
    Y=list(merged_df['label'])
    
    # 划分数据集
    x_train, x_test, y_train, y_test =  train_test_split(X, Y, test_size=0.2)

    train_df = pd.DataFrame({'url': x_train, 'label': y_train})
    test_df = pd.DataFrame({'url':x_test,'label':y_test})
    logger.info("train_src_data")
    print(train_df['label'].value_counts())
    logger.info("test_src_data")
    print(test_df['label'].value_counts())

    # train data分组
    train_grouped = train_df.groupby('label')

    data = []
    # 按分组上采样
    for label, group in train_grouped:
        # data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
        # 构造新的DataFrame
        if len(group['label']) < 30 :
            group = group.sample(n=30, replace=True)
            data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
        # elif len(group['label']) > 400 :
        #     group = group.sample(n=3*len(group['label']), replace=True)
        #     data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))
        else :
            data.append(pd.DataFrame({'url': group['url'], 'label': group['label']}))

    # 拼接上采样样本

    merged_df=pd.DataFrame()
    for class_data in data:
        merged_df = pd.concat([merged_df,class_data], axis=0)

    print(merged_df['label'].value_counts())
        
    x_train=list(merged_df['url'])
    y_train=list(merged_df['label'])



    # Tokenizer
    y_temp = y_train

    train_encoding = tokenizer(x_train, truncation=True, padding=True, max_length=max_len)
    test_encoding = tokenizer(x_test, truncation=True, padding=True, max_length=max_len)

    
    train_encoding['input_ids'], y_train = smote.fit_resample(train_encoding['input_ids'],y_train)
    train_encoding['attention_mask'], y_temp = smote.fit_resample(train_encoding['attention_mask'],y_temp)
    
   
    return train_encoding,test_encoding,y_train,y_test

def create_dataloader(train_encoding,test_encoding,y_train,y_test,batch_size):
    #将数据集包装成torch的Dataset形式
    train_dataset = NewsDataset(train_encoding, y_train)
    test_dataset = NewsDataset(test_encoding, y_test)
    # 单个读取到批量读取
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    batch = next(iter(train_dataloader))
    print(batch["input_ids"].shape)
    return train_dataloader,test_dataloader

                                        
# 训练函数
def train(train_dataloader,test_dataloader,device,model):
    criterion = nn.CrossEntropyLoss().to(device)

    # 优化方法
    #过滤掉被冻结的参数，反向传播需要更新的参数
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,no_deprecation_warning=True)
    total_steps = len(train_dataloader) * 1
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_dataloader)
    batch_iter = tqdm(train_dataloader)
    best_acc = 0
    total_train_accuracy = 0
    for i,batch in enumerate(batch_iter):
        # 正向传播
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)    
            torch.where(torch.isnan(loss), torch.full_like(loss, 0), loss)

        batch_iter.set_description("loss %.4f"% loss)
        total_train_loss += loss.item()

        logits = outputs
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_train_accuracy += flat_accuracy(logits, label_ids)

        loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        
        

        if (i+1) % args.accumulation_steps == 0:
            # scaler.unscale_(optim)
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        else:
            scaler.update()
            optim.zero_grad()

        iter_num += 1
        # if(iter_num % 100==0):
        #     print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (args.num_train_epochs, iter_num, loss.item(), iter_num/total_iter*100))
    acc = validation(test_dataloader,model,device)
    logger.info("Train Acc:%.4f"% (total_train_accuracy/len(batch_iter)))
    if acc > best_acc : 
        best_acc = acc
        checkpoint = {
                'best_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
                'seed': args.seed
            }
        checkpoint_path = args.checkpoint_path
        if os.path.exists(checkpoint_path):
            last_checkpoint = torch.load(checkpoint_path)
            last_best_acc = last_checkpoint['best_acc']
        else :
            last_best_acc = 0
        if best_acc > last_best_acc :
            logger.info("last_best_acc %.4f,current_best_acc %.4f",last_best_acc,best_acc)
            logger.info("保存模型到： %s", checkpoint_path)
            torch.save(checkpoint, checkpoint_path)

# 测试函数

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def validation(test_dataloader,model,device):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    total_labels =[]
    total_pre = []
    total_eval_accuracy = 0
    total_eval_loss = 0
    num_iter = 0
    with torch.no_grad():
        test_iter = tqdm(test_dataloader,desc="Eval")
        for batch in test_iter:
            # 正常传播
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits

            total_labels.extend(labels.tolist())
            total_pre.extend(torch.argmax(outputs, dim=-1).tolist())
            
            # if num_iter == 0 :
            #     preds = torch.argmax(outputs, dim=-1)  # 找到预测类别的索引
            #     print("预测类别为：", preds)  # 打印预测类别
            #     time.sleep(0.5)
            

            loss = criterion(outputs, labels)
            logits = outputs

            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            num_iter += 1
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    # cm = confusion_matrix(np.array(total_labels), np.array(total_pre))
    precision, recall, _, _ = precision_recall_fscore_support(total_labels, total_pre, average=None)
    for i in range(args.num_class):
        logger.info("第%d类 Precision: %.4f,Recall: %.4f" % (i,precision[i], recall[i]))
    logger.info("Test ACC:%.4f"% (avg_val_accuracy))
    return avg_val_accuracy

    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(
        "当前节点号: %s, 使用设备: %s, 是否采用半精度训练: %s",
        args.local_rank,
        device,
        bool(args.fp16),
    )
    
    if args.model_name == "bert-base-uncased" :
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model_name == "albert-base-v2" :
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    train_encoding,test_encoding,y_train,y_test= data_pre(tokenizer)
    train_dataloader,test_dataloader = create_dataloader(train_encoding,test_encoding,y_train,y_test,batch_size=args.batch_size)
    
    logger.info(
        "预训练模型：%s,分类数：%d",
        args.model_name,
        args.num_class
    )

    logger.info("Training/evaluation parameters %s", args)

    log.set_verbosity_error()

    # checkpoint_path = "../model_trained/bert-base-uncased_2_4_6.tar"
    # checkpoint = torch.load(checkpoint_path)
    # config = BertConfig.from_pretrained('./pretrained_model/bert-base-uncased-config.json', num_labels=args.num_class)
    # model = BertForSequenceClassification.from_pretrained('./pretrained_model/bert-base-uncased-pytorch_model.bin', config=config)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.to(device)


    model = mymodel.load_pretrained_model(args.model_name,args.num_class,args.freeze_bool).to(device)

    

    for epoch in range(args.num_train_epochs):
        logger.info("--------------------------------------Epoch: %d ------------------------------------" ,epoch)
        train(train_dataloader,test_dataloader,device,model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='随机数种子')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)                    
    parser.add_argument('--debug',default =False, action='store_true')
    parser.add_argument('--model_name',type=str,default="albert-base-v2",help="pretrained model: bert-base-uncased/albert-base-v2")
    parser.add_argument("--freeze_bool",type=bool,default=True,help="whether to freeze pretrained parameters")
    parser.add_argument("--checkpoint_path",type=str,default="../model_trained/",help="the path of model checkpoint")
    parser.add_argument("--accumulation_steps",type=int,default=1,help="grandient accumulation ")
    parser.add_argument("--num_train_epochs", default=15, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.",)
    parser.add_argument("--learning_rate", default=3e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_class",type=int, help="number of classify")
    
    
    # Other parameters
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--fp16",
	    type=bool,default=True,help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    args = parser.parse_args()

    setup_seed(args.seed)
    main()
