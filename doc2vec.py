import json
from scipy import sparse
import  os
DATA_DIR = 'data/unix/'
OUTPUT_DIR = "output/"
data_dir = "data/unix/output/"
vector_size = 200
if not os.path.exists(DATA_DIR+OUTPUT_DIR):
    os.mkdir(DATA_DIR+OUTPUT_DIR)
with open(DATA_DIR+OUTPUT_DIR+'posts_json_50.0k.json') as f:
    corpus = json.load(f)

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument

from gensim.models import doc2vec

def label_sentences(corpus):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the complaint narrative.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label =  str(i)
        labeled.append(doc2vec.TaggedDocument(corpus[i]['Post'].split(), [label]))
    return labeled
data = label_sentences(corpus)
model_dbow = Doc2Vec(dm=0, vector_size=vector_size, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(data)])
for epoch in range(300):
    model_dbow.train(utils.shuffle([x for x in tqdm(data)]), total_examples=len(data), epochs=1)
    model_dbow.alpha -= 0.001
    model_dbow.min_alpha = model_dbow.alpha
def get_vectors(model, corpus_size, vectors_size):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors
from scipy import sparse
doc_vec = get_vectors(model_dbow,len(data),vector_size)
y = sparse.load_npz(data_dir+'tags_one_hot_sparse.npz')
y = y.todense()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( doc_vec, y, test_size=0.09, random_state=42)
np.save(data_dir+"y_train_doc2vec"+str(vector_size)+".npy",y_train)
np.save(data_dir+"X_train_doc2vec"+str(vector_size)+".npy",X_train)
np.save(data_dir+"X_test_doc2vec"+str(vector_size)+".npy",X_test)
np.save(data_dir+"y_test_doc2vec"+str(vector_size)+".npy",y_test)
