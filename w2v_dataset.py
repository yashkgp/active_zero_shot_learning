from data_parser import DataParser
import os
import numpy as np
import json
from scipy import sparse
import tqdm
DATA_DIR = 'data/superuser/'
OUTPUT_DIR = "output/"
if not os.path.exists(DATA_DIR+OUTPUT_DIR):
    os.mkdir(DATA_DIR+OUTPUT_DIR)
TAGS_FILE = 'Tags.xml'
POSTS_FILE = 'Posts.xml'
MIN_TAGS_COUNT = 10
MAX_QUES = -1
MIN_COUNT = 25
dp = DataParser()
dp.get_posts_and_tags(DATA_DIR+POSTS_FILE,target_filename=DATA_DIR+OUTPUT_DIR+"posts_json_total.json",max_ques=MAX_QUES)
with open(DATA_DIR+OUTPUT_DIR+'posts_json_total.json') as f:
    corpus = json.load(f)
size = len (corpus)
f=open(DATA_DIR+OUTPUT_DIR+"total_tags_w2v.txt","w")
for i in tqdm.tqdm(corpus):
    f.write(" ".join(i["Tags"])+"\n")
f.close()