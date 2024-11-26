import torch
import torch.nn.functional as F

dict  = {}
with open('cos_feature.txt','r')as f:
    readlines = f.readlines()
    for line in readlines:

        id = line.strip('\n').split('\t')[0]
        feature = line.strip('\n').split('\t')[1:]
        newfeature = [float(i) for i in feature]
        dict[id] = newfeature
dict1 = {}

with open('Input_GI.txt','r')as f:
    readlines = f.readlines()
    for line in readlines:
        id = line.strip('\n').split('\t')[0]
        id2 = line.strip('\n').split('\t')[1]
        if id in dict and id2 in dict:
            cosine_similarity = F.cosine_similarity(torch.Tensor(dict[id]), torch.Tensor(dict[id2]), dim=0)
            dict1[id+'\t'+id2+'\t'+str(cosine_similarity.tolist())] = ''

with open('cos_result.txt','w')as f:
    for key in dict1:
        f.write(key+'\n')