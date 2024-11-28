import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
def prepare_feature():

    with open('Omics_feature.txt', 'r') as f:
        one_feature, name_list = [], []
        for line in f:
            uniprotid, *features = line.strip().split('\t')
            features = list(map(float, features))
            one_feature.append(features)
            name_list.append(uniprotid)

    one_feature = torch.tensor(one_feature)

    with open('PPI_feature.txt', 'r') as f:
        ppi_feature_dict = {}
        for line in f:
            uniprotid, *ppi_features = line.strip().split('\t')
            ppi_features = list(map(float, ppi_features))
            ppi_feature_dict[uniprotid] = ppi_features

    combined_features = []
    for i, uniprotid in enumerate(name_list):
        if uniprotid in ppi_feature_dict:
            combined_feature = one_feature[i].tolist() + ppi_feature_dict[uniprotid]
        else:
            combined_feature = one_feature[i].tolist() + [0] * len(next(iter(ppi_feature_dict.values())))
        combined_features.append(combined_feature)

    combined_features = torch.tensor(combined_features)
    return combined_features, name_list

def prepare_label(name_list):
    with open('all_positive_gene.txt', 'r') as f:
        dict_at1 = {}
        for line in f:
            gene = line.strip('\n')
            dict_at1[gene] = 'yes'

    with open('single_positive_gene.txt', 'r') as f:
        dict_at2 = {}
        for line in f:
            gene = line.strip('\n')
            dict_at2[gene] = 'yes'

    labels = []
    for gene in name_list:
        label1 = 1 if gene in dict_at1 else 0
        label2 = 1 if gene in dict_at2 else 0
        labels.append([label1, label2])

    return torch.tensor(labels, dtype=torch.float32)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.sigmoid = nn.Sigmoid()

        self.first = nn.Linear(dim_in, 100)
        self.hidden1 = nn.Linear(100, 100)
        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 100)
        self.out = nn.Linear(100, dim_out)

    def forward(self, X):
        output = F.relu(self.first(X))
        output = F.relu(self.hidden1(output))
        output = F.relu(self.hidden2(output))
        output = F.relu(self.hidden3(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output

def load_checkpoint(path: str):
    debug = info = print

    state = torch.load(path)
    loaded_state_dict = state['state_dict']

    model = GCN(50,2)
    model_state_dict = model.state_dict()

    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            info(f'Warning: Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            info(f'Warning: Pretrained parameter "{param_name}" '
                 f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                 f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model

def getlist(filename):
    list1 = []
    list2 = []
    with open(filename, 'r')as f:
        readlines = f.readlines()
        for line in readlines:
            one = line.strip('\n').split('\t')[0]
            two = line.strip('\n').split('\t')[1]
            list1.append(float(one))
            list2.append(float(two))
    return list1, list2

def get_auc_list(list1, list2, filename, score):
    with open(filename, 'w')as f:
        f.write('label\tscore\tname\n')
        for i in range(len(list2)):
            if list2[i][0]<score:
                continue
            f.write(str(list1[i][0]) + '\t' + str(list2[i][0]) + '\t' + name_list[i] + '\n')

if __name__ == "__main__":
    X_final, name_list = prepare_feature()
    Pre_label = prepare_label(name_list)
    model1 = load_checkpoint('model/dsf')
    Pre_value = model1(X_final)

    for i in range(Pre_value.size()[1]):
        value = Pre_value.index_select(1, torch.tensor([i]))
        label = Pre_label.index_select(1, torch.tensor([i]))

    pre_list1, pre_list2 = label.detach().numpy().tolist(),value.detach().numpy().tolist()
    fpr5, tpr5, threshold5 = roc_curve(pre_list1, pre_list2)

    for i in range(len(fpr5)):
        if 1 - fpr5[i] < 0.99:
            score = threshold5[i-1]
            break

    get_auc_list(label.detach().numpy().tolist(), value.detach().numpy().tolist(), 'result.txt',score)

