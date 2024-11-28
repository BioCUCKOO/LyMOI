import networkx as nx
import math
import torch
with open('all_positive_gene.txt', 'r')as f:
    dict_at = {}
    readlines = f.readlines()
    for line in readlines:
        gene = line.strip('\n')
        dict_at[gene] = 'yes'

G = nx.read_gpickle('Mynetwork.gpickle')
dict_num = {}

with open('Omics_feature.txt','r')as f:
    readlines = f.readlines()
    dict_feature = {}
    for line in readlines:
        uniprotid = line.strip('\n').split('\t')[0]
        dict_feature[uniprotid] = uniprotid


def sigmoid(x):
    return 1/(1 + math.exp(-x))

for node in dict_feature:
    num = 0
    if node in dict_at:
        num+=1
    if node in G:
        for neibor in list(G.neighbors(node)):
            if neibor in dict_at:
                num +=1
    num = torch.sigmoid(torch.log(torch.Tensor([num]))).item()
    dict_num[node] = num

with open('PPI_feature.txt','w')as f:
    for key in dict_num:
        f.write(key + '\t' + str(dict_num[key]) + '\n')



