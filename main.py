from optparse import OptionParser
import pickle

import matplotlib
matplotlib.use('agg')


from sklearn.model_selection import train_test_split
method ='tfidf'
vector_size=50
print(method)
parser = OptionParser()
parser.add_option("--data_dir", dest="data_dir")
parser.add_option("--start_seen", dest="start_seen_classes")
parser.add_option("--end_seen", dest="end_seen_classes")
parser.add_option("--plot_file",dest="plot_file_name")

(options, args) = parser.parse_args()

data_dir = options.data_dir
start_seen_classes = int(options.start_seen_classes)
end_seen_classes = int(options.end_seen_classes)
plot_file_name = options.plot_file_name


from select_classes import select_classes, select_classes_baseline
from Logreg import logreg_model
from RBM1 import train_RBM_and_compute_simiarity
from evaluate import compute_unseen_class_scores, compute_precision
from scipy import sparse
import numpy as np
import os


#y_data = sparse.load_npz(data_dir+'tags_one_hot_sparse.npz')
#y_train = y_data.todense()
#y_data = sparse.load_npz(data_dir+'test_tags_one_hot_sparse.npz')
#y_test = y_data.todense()
'''
tfdif based removal of redundant words just as in the baseline paper:
'''
if(method=='tfidf'):
	y_data = sparse.load_npz(data_dir+'tags_one_hot_sparse.npz')
	y_data = y_data.todense()
	X_data = sparse.load_npz(data_dir+'tfifdf_transformed.npz')
	X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42)

    #X_train = sparse.load_npz(data_dir+'tfifdf_transformed.npz')
    #X_test = sparse.load_npz(data_dir+'test_tfifdf_transformed.npz')
    # y_train = np.load(data_dir+"y_train.npy")
    # y_test = np.load(data_dir+"y_test.npy")
    # y_data = np.concatenate((y_train,y_test),axis=0)
    # X_train =sparse.load_npz(data_dir+"X_train.npz")
    # X_test =sparse.load_npz(data_dir+"X_test.npz")
'''
testing with doc2vec embeddings of questions(documents)
'''
else:
    y_test = np.load(data_dir+"y_test_doc2vec"+str(vector_size)+".npy")
    y_train = np.load(data_dir+"y_train_doc2vec"+str(vector_size)+".npy")
    X_train = np.load(data_dir+"X_train_doc2vec"+str(vector_size)+".npy")
    X_test = np.load(data_dir+"X_test_doc2vec"+str(vector_size)+".npy")
    y_data = np.concatenate((y_train,y_test),axis=0)

if os.path.exists(data_dir + 'similarity_w2v_trained_50D.npy'):
	similarity_matrix = np.load(data_dir + 'similarity_w2v_trained_50D.npy')
else:
	similarity_matrix = train_RBM_and_compute_simiarity(y_train,target_filename=data_dir + 'similarity_w2v.npy')

plot_x_ent = []
plot_y_ent = []

plot_x_deg = []
plot_y_deg = []

plot_x_topn = []
plot_y_topn = []

plot_x_page = []
plot_y_page = []

'''
can change the modes --- look into the select_classes.py file
'''

modes = ['top_n','pagerank_max','max-ent-uu']


#print(X_train.shape, y_train.shape)
#print(X_test.shape,y_test.shape)

for mode in modes:
	for num_seen_classes in range(start_seen_classes, end_seen_classes, 5):
		if mode == 'top_n' :
			selected_classes = select_classes_baseline(y_data, num_seen_classes, "top_n_baseline")
			
		else :
			selected_classes = select_classes(similarity_matrix, num_seen_classes, mode)

		to_remove = []
		print (select_classes)

		for class_idx in selected_classes:
			if np.sum(y_train[:, class_idx] < 0.5) < 5 or np.sum(y_train[:, class_idx] > 0.5) < 5:
				to_remove.append(class_idx)

		for class_idx in to_remove:
			selected_classes.remove(class_idx)
        
		models = logreg_model(X_train,selected_classes,y_train)
		y_pred  = np.zeros((y_test.shape[0],len(selected_classes)))

		i = 0
		for key in selected_classes:
			model = models[key]
			y_pred[:,i] = model.predict_proba(X_test)[:,1]
			i += 1

        # list the number of unseen classes
		unseen_classes = list(set(range(y_data.shape[1])) - set(selected_classes))
        # calculates precision@5
		score_unseen = compute_unseen_class_scores(y_pred,similarity_matrix,selected_classes,unseen_classes)

		precision = compute_precision(y_test[:,unseen_classes],score_unseen)

		print("Precision : %.6f" % (precision), num_seen_classes)
        
        ####### visualization part#####################################

		if mode == 'max-ent-uu':
			plot_x_ent.append(num_seen_classes)
			plot_y_ent.append(precision)
		elif mode  =='pagerank_max':
			plot_x_page.append(num_seen_classes)
			plot_y_page.append(precision)
		elif mode == 'max-deg-uu':
			plot_x_deg.append(num_seen_classes)
			plot_y_deg.append(precision)
		else :
			plot_x_topn.append(num_seen_classes)
			plot_y_topn.append(precision)




