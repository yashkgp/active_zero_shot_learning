from gensim.models.keyedvectors import KeyedVectors
import numpy as np

'''
script to load the vectors and compute the cosine similarity between each word
'''
data_dr ="data/superuser/output/"
target_file = "data/superuser/output/"
word_vect = KeyedVectors.load_word2vec_format(data_dr+"out_50D.vec")

import json
with open("data/superuser/output/tags_json.json") as f:
    data = json.load(f)
num_tags = len(data)
print(num_tags)
simi_matrix = np.zeros((num_tags,num_tags))
counter = 0
for i in range(0,num_tags):
    for j in range(i,num_tags):
        if (i == j):
            simi_matrix[i][j] = 1
        else :
            try :
                val  = word_vect.similarity(data[i],data[j])
            except KeyError as e:
                val = 0
                counter+=1
                #print (data[i],data[j])
            simi_matrix[i][j] = val
            simi_matrix[j][i] = val
print (counter)
counter = 0
np.save(target_file + "similarity_w2v_trained_50D.npy",simi_matrix)
simi_matrix_wordnet = np.zeros((num_tags,num_tags))
from nltk.corpus import wordnet
for i in range(0,num_tags):
    for j in range(i,num_tags):
        if (i == j):
            simi_matrix_wordnet[i][j] = 1
        else :
            try :
                word1 = wordnet.synsets(data[i])[0]
                word2 = wordnet.synsets(data[j])[0]
                val = word1.wup_similarity(word2)
            except IndexError as e:
                val = 0
                counter+=1
                #print (data[i],data[j])
            simi_matrix_wordnet[i][j] =val
            simi_matrix_wordnet[j][i] = val
print(counter)
np.save(target_file+"similarity_wordnet.npy",simi_matrix_wordnet)
