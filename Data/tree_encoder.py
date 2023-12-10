import hashlib
import csv
import numpy as np


def hash_tree(adj_matrix):
    n = len(adj_matrix)
    hash_table = {}
    label = [0] * 1024
    for i in range(n):
        neighbors = tuple(sorted([j for j in range(n) if adj_matrix[i][j] == 1]))
        if neighbors not in hash_table:
            hash_table[neighbors] = len(hash_table)
        label[i] = hash_table[neighbors]
    return label


def get_tree_encoder(dataset):
    dataset_A = []
    h_tree = []
    with open('./Data/trees/' + dataset + '_tree.csv') as f:
        f_csv = csv.reader(f)
        result = list(f_csv)
        index = 0
        while(True):
            l = int(result[index][1])
            index += 1
            a = []
            A = []
            for j in range(index, index+l):
                for k in range(l):
                    a.append(int(result[j][k]))
                A.append(a)
                a = []
            index += l
            dataset_A.append(A)

            if index == len(result):
                break

    for A in dataset_A:
        A = np.array(A)
        h_tree.append(hash_tree(A))

    return h_tree