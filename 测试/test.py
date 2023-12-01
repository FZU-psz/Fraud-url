import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
import random
# Set log level to ERROR
import logging
import requests
from bs4 import BeautifulSoup
import jieba
import pygtrans
from pygtrans import Translate
from collections import Counter
import re
import time
logging.basicConfig(level=logging.INFO)
logging.getLogger("jieba").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import http.client
import hashlib
import urllib
import random
import json


def trans_lang(q,httpClient):
    trans_result = q
    # 百度appid和密钥需要通过注册百度【翻译开放平台】账号后获得
    appid = '20231124001891095'  # 填写你的appid
    secretKey = 'oqmxP6XXaXPC02maIWyC'  # 填写你的密钥

    myurl = '/api/trans/vip/translate'  # 通用翻译API HTTP地址

    fromLang = 'en'  # 原文语种
    toLang = 'zh'  # 译文语种
    salt = random.randint(32768, 65536)
    # 手动录入翻译内容，q存放
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()

    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + \
            '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

    # 建立会话，返回结果
    try:
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        trans_result = result['trans_result'][0]['dst']

    except Exception as e:
        print("error occured in translation")
        print(e)

    return trans_result

def get_context(url):
    try:
        Client = http.client.HTTPConnection('api.fanyi.baidu.com')
        if not url.startswith("http"):
                url = "http://" + url
        response = requests.get(url, timeout=2)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        seg_list = jieba.cut(text)
        word_counts = Counter(seg_list)
        if len(word_counts) == 0:
            output = ","
        else:
            top5 = word_counts.most_common(100)
            top5 = [word[0] for word in top5[:100] if len(word[0]) > 1] 
            output = ",".join(top5[:5])
            text = trans_lang(output,Client)
            pattern = re.compile('[^0-9a-zA-Z,/:\-]')
            output  = pattern.sub('', text)
        return output  
    except Exception as e:
        print(e)
        return 'Unable access'

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

import tldextract
def is_valid_url(url):
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return False
    else:
        return True
'''
标签类别说明：
incorrect 不合理数据（非URL）
unable_to_access 无法访问的数据

'''
def classify_url(url):

        #检查是否合法
        if not is_valid_url(url):
            return "Invalid url!"

        dic ={0:'正常',1:'购物消费',2:'婚恋交友',3:'假冒身份',4:'钓鱼网站',5:'冒充公检法',6:
'平台诈骗',7:'招聘兼职',8:'杀猪盘',9:'博彩赌博',10:'信贷理财',11:'刷单诈骗',12:'中奖诈骗'}

        #检查网址是否可到达
        input_data = url.replace('https:','http:').strip('/')

        #查验数据库
        tmp_url = url.replace('https://','').replace("http://",'').strip('/')
        dbResult = websitedb.queryUrl(tmp_url)

        if dbResult !=None:
            try :
                return dic[int(dbResult)]#如果是数字，则转换
            except:
                return dbResult#如果是unableaccess 之类的就直接返回结果
        else :
            input_data = [input_data]
            max_len = 0
            for item in input_data:
                if len(item) > max_len:
                    max_len = len(item)

            for input_url in input_data:
                url_0 = tokenizer(input_url, padding=True, truncation=True, max_length=max_len)
                input_ids = torch.tensor(url_0['input_ids']).unsqueeze(0)
                attention_mask = torch.tensor(url_0['attention_mask']).unsqueeze(0)

                # 第一层二分类预测
                input_ids_0 = input_ids.to(device_0)
                attention_mask_0 = attention_mask.to(device_0)
                outputs_0 = model_0(input_ids_0, attention_mask=attention_mask_0, ).logits
                probs_0 = torch.softmax(outputs_0, dim=1)
                pred_class_0 = torch.argmax(probs_0, dim=1)
                ans_class = pred_class_0.cpu()
                if ans_class == 1:
                    # 非0类进入第二张卡上的模型进行十二分类预测
                    url_1 = get_context(input_url)
                    if(url_1=='Unable access'):
                        return url_1
                    url_1 = url_1.replace(" ", "")
                    url_1 = tokenizer(url_1, padding=True, truncation=True, max_length=max_len)
                    input_ids = torch.tensor(url_1['input_ids']).unsqueeze(0)
                    attention_mask = torch.tensor(url_1['attention_mask']).unsqueeze(0)

                    input_ids_1 = input_ids.to(device_0)
                    attention_mask_1 = attention_mask.to(device_0)
                    outputs_1 = model_1(input_ids_1, attention_mask=attention_mask_1).logits
                    probs_1 = torch.softmax(outputs_1, dim=1)
                    pred_class_12 = torch.argmax(probs_1, dim=1) + 1
                    ans_class = pred_class_12.cpu()

                return dic[int(ans_class)]
            # logging.info(f"URL:{input_url} PREDICT:{ans_class}类")

#数据库查询
import sqlite3
class websiteDb():
    def __init__(self,table_name='website'):
        self.db= sqlite3.connect('./website.db')
        self.cursor = self.db.cursor()
        self.table_name = table_name

    def queryUrl(self,url):
        execute_sql = "select label from {} where url='{}'".format(self.table_name,url)
        self.cursor.execute(execute_sql)
        result = self.cursor.fetchone()
        if result is None:
            return None
        else:
            return result[0]



#服务
from http.server import BaseHTTPRequestHandler, HTTPServer
class RequestHandler(BaseHTTPRequestHandler):
    def _set_response(self, status_code=200, content_type='text/plain'):
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_POST(self):
        if self.path == '/diting':
            content_length = int(self.headers['Content-Length'])
            print(content_length)
            post_data = self.rfile.read(content_length).decode('utf-8')

            print(post_data)
            response = classify_url(post_data)

            self._set_response()
            self.wfile.write(response.encode('utf-8'))
        else:
            self._set_response(status_code=404)
            self.wfile.write('Not Found'.encode('utf-8'))



if __name__ == "__main__":
    setup_seed(42)
    device_0 = torch.device("cuda:0")
    # device_1 = torch.device("cuda:1")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('../pretrained_model/')

    logging.info("Model Loading")
    # Load model
    # 二分类
    checkpoint_path = "../model_trained/bert-base-uncased_0.tar"
    checkpoint = torch.load(checkpoint_path)
    config = BertConfig.from_pretrained('../pretrained_model/config.json', num_labels=2)
    model_0 = BertForSequenceClassification.from_pretrained('../pretrained_model/pytorch_model.bin', config=config)
    model_0.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model_0.to(device_0)
    model_0.eval()

    # 十二分类
    checkpoint_path = "../model_trained/bert-base-uncased_1_2_3_4_5_6_7_8_9_10_11_12.tar"
    checkpoint = torch.load(checkpoint_path)
    config = BertConfig.from_pretrained('../pretrained_model/config.json', num_labels=12)
    model_1 = BertForSequenceClassification.from_pretrained('../pretrained_model/pytorch_model.bin', config=config)
    model_1.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model_1.to(device_0)
    model_1.eval()


    #加载数据库
    websitedb = websiteDb('website')

    logging.info("Predicting")
    #启动服务器
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Server started on port 8080')
    httpd.serve_forever()









