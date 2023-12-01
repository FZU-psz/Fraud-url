# 大数据 彭诗忠
#开发时间  9:48 2023/11/28


#检查网址合法性
import tldextract
def is_valid_url(url):
    ext = tldextract.extract(url)
    if not ext.domain or not ext.suffix:
        return False
    else:
        return True


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

website = websiteDb()
url = 'www.baidu.com'

# 检查网址是否可到
tmp_url = url.replace('https://', '').replace("http://",'').strip('/')
result = website.queryUrl(tmp_url)
print(result)