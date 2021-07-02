from logging import debug
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
import torch
import pymysql
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

from utils import preprocess, kw_model, get_tags, update_single_node_size

def get_connection():
    conn = pymysql.connect(host='localhost', port=3306, db='graph', user='root', password='taoliankuan')
    return conn

def _get_node_size(k=50):
    # idx : 0 - 50
    node_size = torch.arange(0, 20, 20./k).float()
    node_size = torch.softmax(node_size,dim=0) * k* 2 + 10.0

    node_size = node_size.tolist()
    print(node_size)

    return node_size

tag_list = []
with open("tag_list.txt", "r") as file:
    for line in file.readlines():
        tag_list.append(line.replace("\n", ""))

tag_list = np.array(tag_list)
def query(tag_list):
    sql = "match (n:tag)-[r]-(m:article) where "
    for i in range(len(tag_list)):
        if i != len(tag_list) - 1:
            sql += "n.name=\"" + tag_list[i] + "\" or "
        else:
            sql += "n.name=\"" + tag_list[i] + "\" "
    # sql +=  "return n1 ,r ,n2"
    return sql

@app.route('/')
def hello_world():
    #ret = t.output(raw_text,ner_model, tokenizer, sim_model, sim_processor)
    return render_template('index.html')

@app.route('/addanddelete')
def add_and_delete():
    return render_template('addanddelete.html')

@app.route('/content')
def show_content():
    return render_template('content.html')

@app.route('/tag_query', methods=['POST', 'GET'])
def tag_query():
    data = request.args.get('name')
    k = 50
    res = get_tags(data, tag_list, k)
    node_size = _get_node_size(k)
    for tag, s in zip(res, node_size):
        update_single_node_size(tag, s)

    sql = query(res)
    print(sql)
    return jsonify(answer = sql)

@app.route('/add', methods=['POST', 'GET'])
def add_title():
    title = request.args.get('title')
    content = request.args.get('content')
    # mysql 
    sql = "insert into article value('" + title + "','" + content + "')"
    print(sql)
    conn = get_connection()
    cursor=conn.cursor()
    cursor.execute(sql)
    cursor.close()
    conn.commit()
    conn.close()

   #neo4j
   

    return "success"

@app.route('/delete_title', methods=['POST', 'GET'])
def delete_title():
    title = request.args.get('title')
    print(title)
    
    conn = get_connection()
    cursor=conn.cursor()
    # 查
    sql = "select * from article where title='" + title +"'" 
    count = cursor.execute(sql)
    if count == 0:
        return "Not found"
    # 删
    sql = "delete from article where title='" + title +"'" 
    print(sql)
    cursor.execute(sql)
    cursor.close()
    conn.commit()
    conn.close()
    return "success"

@app.route('/select_article', methods=['POST','GET'])
def select_article():
    caption = request.args.get('caption')
    print(caption)
    conn = get_connection()
    cursor=conn.cursor()
    sql = "select content from article where title='" + caption +"'" 
    print(sql)
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    conn.commit()
    conn.close()
    print(result[0][0])
    return jsonify(content = result[0][0])


if __name__ == '__main__':

    # app.run('0.0.0.0',5000)
    WSGIServer(('0.0.0.0',5000),app).serve_forever()