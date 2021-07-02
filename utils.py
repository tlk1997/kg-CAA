import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "taoliankuan"), encrypted=False)
from logging import debug
from keybert import KeyBERT
import re
import jieba
from numpy.lib.twodim_base import tri
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer

def preprocess(document):
    string = '''！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.'''
    text =re.sub(string,"",document)
    document = re.sub(r'@[\w_-]+', '', document)
    document = re.sub(r'-', ' ', document)
    document = re.sub('https?://[^ ]+', '', document)
    document= ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", document).split())   
    document = re.sub(r'<[^>]*>', '', document)
    document = re.sub(r'\[(.+)\][(].+[)]', r'\1', document)
    document = document.lower()
    return document 


doc = """
11月上旬，在杭州市西湖区第五十届中小学生田径运动会初中组一百米赛跑中，“一名女生一骑绝尘，将对手远远甩在后面”的视频冲上了热搜。这名女生叫陆启睿，是杭州保俶塔实验学校九年级学生，网友们称她“风一样的女孩”。在这届运动会上，陆启睿毫无悬念地包揽包揽女子组100米、200米双项冠军。而100米12秒07、200米25秒11的成绩，也超过了我国女子一级运动员100米12.33秒、200米25.42秒的标准。
陆启睿的父亲接受当地媒体记者采访时表示，杭州保俶塔实验学校是个体育强校，很重视挖掘孩子们的体育天赋。小学四年级，因为跑步快，陆启睿正式进入学校田径队。训练比较辛苦，每天早上6点起床，6点半到学校运动场，周末只休息一天，寒来暑往一练就是5年。“她小学六年级时，就是全市200米第一名，但这次上热搜，引起全国关注感到意外。”
小陆和清华大学十项全能教练马汝平
小陆和清华大学十项全能教练马汝平杭州保俶塔实验学校田径队教练吕晓雨说，田径就是跑，练力量练素质，比较枯燥。其实就跟文化课一样， 需要持之以恒的过程。“陆启睿给所有教练的一个感觉就是很内敛，平时不说话，默默地训练，她几乎是不请假的。”
上周清华附中马约翰体育特长班邀请陆启睿前往北京参加试训。清华附中马约翰体育特长班创办于1986年。“马约翰班”的招生对象是有体育特长、有培养前途的中学生。据不完全统计，创办至今，已培养6名国际健将、30名运动健将、273名一级运动员，包括何姿、施廷懋、周吕鑫、李翔宇、王宇、郭凯等著名运动员。
陆启睿父亲介绍，试训时间为期3天，孩子跟着清华附中的田径队一起进行热身、拉伸运动，也测了100米、200米的成绩。除了看成绩，专业教练也会看孩子整体的身体素质，比如身体的比例、肌肉线条等等，还要看性格是否合适。三天试训下来，教练对女儿的基本水平还是比较满意的，已经被正式录取。“清华附中很重视文化学习，我们感受了一下，学习氛围很好，如果一切顺利，孩子升入高中参加更加系统的训练，未来可以考入清华大学。”
"""




model_name_or_path = "distilbert-base-nli-mean-tokens"


kw_model = KeyBERT(model_name_or_path)
keywords = kw_model.extract_keywords(doc)


model = AutoModel.from_pretrained("voidful/albert_chinese_tiny")
model.eval()
tokenizer = BertTokenizer.from_pretrained("voidful/albert_chinese_tiny")

def cos_sim(a, b):
    a = a.squeeze(0)
    b = b.squeeze(0)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)

    return a@b

def get_tags(text, tag_list, k=5):
    tag_list_embedding = tokenizer(tag_list.tolist(), return_tensors='pt', padding=True)
    output = model(**tag_list_embedding).pooler_output

    text = tokenizer(text, return_tensors='pt', padding=True)
    text_embedding = model(**text).pooler_output

    score = torch.tensor([cos_sim(text_embedding, o) for o in output])

    _, best_idx = torch.topk(score, k)
    best_idx = best_idx.tolist()

    return tag_list[best_idx].tolist()


cnt = 1


def add_nodes_article(tx,name, cm):
    tx.run("CREATE (n:article {name: $name, community:$cm, node_size:$node_size})",name=name, cm=cm, node_size=3)


def add_nodes_tag(tx,name, cm):
    tx.run("CREATE (n:tag {name: $name, community:$cm, node_size:$node_size})",name=name, cm=cm, node_size=6)

def add_rels(tx,name1,name2):
    tx.run("MATCH (a:article {name: $name1}),(b:tag {name: $name2}) MERGE (a)-[:BELONG]-(b)",name1=name1,name2=name2)

def get_tags_by_text(doc, top_n=3):
    doc = ' '.join(jieba.lcut(preprocess(doc)))
    keywords = kw_model.extract_keywords(doc, top_n=top_n
        # use_maxsum=True, nr_candidates=20, top_n=5
    )
    return [o[0] for o in keywords]


def add_friend(tx, name, friend_name):
    tx.run("MERGE (a:Person {name: $name}) "
        "MERGE (a)-[:BELONG]->(friend:Person {name: $friend_name})",
        name=name, friend_name=friend_name
    )


def update_single_node_size(name, cnt):
    def _update_size(tx,name,cm):
        tx.run("MATCH (n {name:$name}) set n.node_size=$node_size return n.name,n.node_size",name=name,node_size=cm)
    
    with driver.session() as session:
        session.write_transaction(_update_size, name, cnt)

tag_list = set()

node_dict = {}
if __name__ == "__main__":

    with driver.session() as session:
        for root, dirs, files in os.walk("./data", topdown=False):
            for name in files:
                if ".DS" in name: continue

                with open(os.path.join(root, name), "r", encoding="utf-8") as file:
                    lines = file.readlines()
                    title = lines[0].replace("\n", "")
                    doc = " ".join(lines[1:])
                    kwords = get_tags_by_text(doc)
                    for kword in kwords:
                        tag_list.add(kword)
                        if kword not in node_dict:
                            node_dict[kword] = 1
                            session.write_transaction(add_nodes_tag, kword, cnt)
                            cnt+=1

                        if title not in node_dict:
                            node_dict[title] = 1
                            session.write_transaction(add_nodes_article, title, 0)
                        
                        session.write_transaction(add_rels, title, kword)
    with open("tag_list.txt", "w") as file:
        for tag in tag_list:
            file.write(tag+"\n")
    