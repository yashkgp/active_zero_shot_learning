import numpy as np

def select_classes(K, n, selection_strategy):
    if selection_strategy == 'max-deg-uu':
        n_classes = K.shape[0]
        seen = []
        unseen = list(range(n_classes))
        
        while len(seen) < n:
            K_UU = K[unseen, :][:, unseen]
            max_degree_node = np.argmax(np.sum(K_UU, 1))
            max_degree_node = unseen[max_degree_node]
            seen.append(max_degree_node)
            unseen.remove(max_degree_node)
        
        seen.sort()
        return seen

    elif selection_strategy == 'max-ent-uu':
        n_classes = K.shape[0]
        seen = []
        unseen = list(range(n_classes))
        
        while len(seen) < n:
            K_UU = K[unseen, :][:, unseen]
            P_UU = np.matmul( np.linalg.inv(np.diag(np.matmul(np.ones((1, K_UU.shape[0])), K_UU).flatten())), K_UU)
            max_entropy_node = np.argmax(-np.sum(P_UU*np.log(P_UU) , 1))
            max_entropy_node = unseen[max_entropy_node]
            seen.append(max_entropy_node)
            unseen.remove(max_entropy_node)
        
        seen.sort()
        return seen
    else:
        print("Error! This selection strategy is not defined.")

def select_classes_baseline(one_hot,n,selection_strategy="top_n"):
    if (selection_strategy=="top_n"):
        col_sum=one_hot.sum(axis=0)
        ind = np.argpartition(col_sum, -1*n)
        seen = []
        for i in range(1,n+1):
            seen.append(ind[-1*i])
        seen.sort()
        return seen
    elif  (selection_strategy=="top_n_baseline"):
        col_sum=one_hot.sum(axis=0)
        ind = np.argpartition(col_sum, -1*n)
        seen = []
        for i in range(1,n+1):
            seen.append(ind[0,-1*i])
        seen.sort()
        return seen
    else:
        print("Error! This selection strategy is not defined.")

if __name__ == "__main__":
    # K = np.asarray([[0,1,1], [0,0,0], [1,1,1]])
    # print(select_classes(K, 2, 'max-deg-uu'))

    K = np.asarray([[0,1,1], [1,0,1], [1,1,1]])
    print(select_classes(K, 2, 'max-ent-uu'))

