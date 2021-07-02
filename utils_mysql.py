import os
import pymysql
from transformers.utils.dummy_tf_objects import TFAutoModel
def get_connection():
    conn = pymysql.connect(host='localhost', port=3306, db='graph', user='root', password='123456')
    return conn

def add_article(title , doc):
    sql = "insert into article value('" + title + "','" + doc + "')"
    conn = get_connection()
    cursor=conn.cursor()
    cursor.execute(sql)
    cursor.close()
    conn.commit()
    conn.close()

flag = 0
for root, dirs, files in os.walk("./data", topdown=False):
    for name in files:
        if ".DS" in name: continue

        with open(os.path.join(root, name), "r", encoding="utf-8") as file:
            lines = file.readlines()
            title = lines[0].replace("\n", "")
            doc = " ".join(lines[1:]).replace("'","\\'")
            print(flag , doc)
            flag += 1
            print(name)
            add_article(title,doc)
