# coding=UTF-8
import gc
import copy
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import random
import os
import deepwalk
from sklearn.decomposition import PCA
import argparse
import networkx as nx
# import node2vec
from openne import graph, node2vec

def get_embedding(vectors: dict):
    matrix = np.zeros((
        len(vectors),
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix


def get_embedding_lap(vectors: dict):
    matrix = np.zeros((
        len(vectors),
        128
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

def processEmb(oldfile,newfile):
    f = open(oldfile)
    next(f)
    for line in f:
        f1 = open(newfile,'a+')
        f1.write(line)
    f1.close()
    f.close()

def clearEmb(newfile):
    f = open(newfile,'w')
    f.truncate()

def Net2edgelist(gene_disease_matrix_net):
    none_zero_position = np.where(np.triu(gene_disease_matrix_net) != 0)
    none_zero_row_index = np.mat(none_zero_position[0],dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1],dtype=int).T
    none_zero_position = np.hstack((none_zero_row_index,none_zero_col_index))
    none_zero_position = np.array(none_zero_position)
    name = 'gene_disease.txt'
    np.savetxt(name, none_zero_position,fmt="%d",delimiter=' ')

#获得gene_disease_emb
def Get_embedding_Matrix(gene_disease_matrix_net):
    Net2edgelist(gene_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("gene_disease.txt")
    _sdne=Get_sdne(graph1)
    _n2v=Get_n2v(graph1)
    _dw=Get_dw(graph1)
    _gf=Get_gf(graph1)
    _lap=Get_lap(graph1)
    _hope=Get_hope(graph1)
    return _dw

def Get_sdne(graph1):
    model = sdne.SDNE(graph1, [1000, 128])
    return get_embedding(model.vectors)

def Get_n2v(graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=10, dim=10)
    n2v_vectors = get_embedding(model.vectors)
    return n2v_vectors

def Get_dw(graph1):
    model = node2vec.Node2vec(graph=graph1, path_length=80, num_paths=10, dim=10, dw=True)
    n2v_vectors = get_embedding(model.vectors)
    return n2v_vectors

def Get_gf(graph1):
    model = gf.GraphFactorization(graph1)
    return get_embedding(model.vectors)

def Get_lap(graph1):
    model = lap.LaplacianEigenmaps(graph1)
    return get_embedding_lap(model.vectors)

def Get_hope(graph1):
    model = hope.HOPE(graph=graph1, d=128)
    return get_embedding(model.vectors)


def Calculate_metrics(predict_y_proba,test_feature_matrix, test_label_vector):
    clf = RandomForestClassifier(random_state=1, n_estimators=200, oob_score=True, n_jobs=-1)
    clf.fit(train_feature_matrix, train_label_vector)
    predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]
    predict_y = clf.predict(test_feature_matrix)
    AUPR = average_precision_score(test_label_vector, predict_y_proba)
    AUC = roc_auc_score(test_label_vector, predict_y_proba)
    MCC = matthews_corrcoef(test_label_vector, predict_y)
    ACC = accuracy_score(test_label_vector, predict_y, normalize=True)
    F1 = f1_score(test_label_vector, predict_y, average='binary')
    REC = recall_score(test_label_vector, predict_y, average='binary')
    PRE = precision_score(test_label_vector, predict_y, average='binary')
    metric = np.array((AUPR, AUC, PRE, REC, ACC, MCC, F1))
    print(metric)

    del train_feature_matrix
    del test_feature_matrix
    del train_label_vector
    del test_label_vector
    gc.collect()
    return metric

def constructNet(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix):
    disease_matrix = np.matrix(np.zeros((dis_chemical_matrix.shape[0], dis_chemical_matrix.shape[0]), dtype=np.int8))
    chemical_matrix = np.matrix(np.zeros((dis_chemical_matrix.shape[1], dis_chemical_matrix.shape[1]),dtype=np.int8))
    mat1 = np.hstack((gene_gene_matrix,gene_chemical_matrix,gene_dis_matrix))
    mat2 = np.hstack((gene_chemical_matrix.T,chemical_matrix,dis_chemical_matrix.T))
    mat3 = np.hstack((gene_dis_matrix.T,dis_chemical_matrix,disease_matrix))
    return np.vstack((mat1,mat2,mat3))

def cross_validation_experiment(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix,seed,ratio = 1):
    none_zero_position = np.where(gene_dis_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(gene_dis_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]
    random.seed(seed)
    zero_random_index = random.sample(range(len(zero_row_index)), ratio * len(none_zero_row_index))
    zero_row_index = zero_row_index[zero_random_index]
    zero_col_index = zero_col_index[zero_random_index]

    row_index = np.append(none_zero_row_index, zero_row_index)
    col_index = np.append(none_zero_col_index, zero_col_index)

    kf = KFold(n_splits=5, random_state=1, shuffle=True)

    metric = np.zeros((1,7), float)
    print("seed=%d, evaluating gene-disease...." % (seed))
    k_count=0

    for train, test in kf.split(row_index):

        train_gene_dis_matrix = np.copy(gene_dis_matrix)

        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]

        train_gene_dis_matrix[test_row, test_col] = 0
        gene_disease_matrix_net = constructNet(train_gene_dis_matrix, dis_chemical_matrix, gene_chemical_matrix,
                                               gene_gene_matrix)
        
        gene_disease_emb = Get_embedding_Matrix(np.mat(gene_disease_matrix_net))
        gene_len = gene_dis_matrix.shape[0]
        chem_len = gene_chemical_matrix.shape[1]
        gene_emb_matrix = np.array(gene_disease_emb[0:gene_len, 1:])
        dis_emb_matrix = np.array(gene_disease_emb[(gene_len + chem_len)::, 1:])

        train_feature_matrix = []
        train_label_vector = []

        for num in range(len(train_row)):
            feature_vector = np.append(gene_emb_matrix[train_row[num], :], dis_emb_matrix[train_col[num], :])
            train_feature_matrix.append(feature_vector)
            train_label_vector.append(gene_dis_matrix[train_row[num], train_col[num]])

        test_feature_matrix = []
        test_label_vector = []

        for num in range(len(test_row)):
            feature_vector = np.append(gene_emb_matrix[test_row[num], :], dis_emb_matrix[test_col[num], :])
            test_feature_matrix.append(feature_vector)
            test_label_vector.append(gene_dis_matrix[test_row[num], test_col[num]])

        train_feature_matrix = np.array(train_feature_matrix)
        train_label_vector = np.array(train_label_vector)
        test_feature_matrix = np.array(test_feature_matrix)
        test_label_vector = np.array(test_label_vector)
        
        
        clf = RandomForestClassifier(random_state=1, n_estimators=200, oob_score=True, n_jobs=-1)
        clf.fit(train_feature_matrix, train_label_vector)
        predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]
        predict_y_proba += metric
        
        metric += Calculate_metrics(predict_y_proba,test_feature_matrix, test_label_vector)
        # k_count+=1

    #print(metric / kf.n_splits)

    metric_avg = np.zeros((1,7),float)
    metric_avg[0, :] += metric / kf.n_splits
    metric = np.array(metric_avg)
    name = 'seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')
    return metric

if __name__=="__main__":
    gene_dis_matrix = np.loadtxt('gene-dis.csv', delimiter=',', dtype=int)
    dis_chemical_matrix = np.loadtxt('dis-chem.csv', delimiter=',', dtype=int)
    gene_chemical_matrix = np.loadtxt('gene-chem.csv', delimiter=',', dtype=int)
    gene_gene_matrix = np.loadtxt('gene-gene-network.csv', delimiter=',', dtype=int)

    result=np.zeros((1,7),float)
    average_result=np.zeros((1,7),float)
    circle_time=1

    for i in range(circle_time):
        result+=cross_validation_experiment(gene_dis_matrix,dis_chemical_matrix,gene_chemical_matrix,gene_gene_matrix,i,1)

    average_result=result/circle_time
    print(average_result)