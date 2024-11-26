import torch
import torch.nn as nn
import torch.nn.functional as F
class GCN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.first_linear_dim = dim_in
        self.ffn_hidden_size = 100
        self.output_size = dim_out
        self.sigmoid = nn.Sigmoid()
        self.first = nn.Linear(self.first_linear_dim, self.ffn_hidden_size)
        self.hidden1 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden2 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
        self.hidden3 = nn.Linear(self.ffn_hidden_size, self.ffn_hidden_size)
    def forward(self, X):
        output = F.relu(self.first(X))
        output = F.relu(self.hidden1(output))
        output = F.relu(self.hidden2(output))
        output = F.relu(self.hidden3(output))
        return output

with open('single_positive_gene.txt', 'r')as f:
    dict_at1 = {}
    readlines = f.readlines()
    for line in readlines:
        gene = line.strip('\n')
        dict_at1[gene] = 'yes'

with open('PPI_feature.txt', 'r')as f:
    dict_PPI = {}
    readlines = f.readlines()
    for line in readlines:
        gene = line.strip('\n').split('\t')[0]
        shuzhi = line.strip('\n').split('\t')[1]
        dict_PPI[gene] = float(shuzhi)

def initialize_weights(model: nn.Module) -> None:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

def load_checkpoint(path: str):
    debug = info = print

    state = torch.load(path)
    loaded_state_dict = state['state_dict']

    model = GCN(50,2)
    model_state_dict = model.state_dict()

    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name == 'out.weight' or param_name == 'out.bias':
            continue
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
def prepare_feature():
    with open('Omics_feature.txt','r')as f:
        one_feature = []
        name_list = []
        readlines = f.readlines()
        for line in readlines:
            uniprotid = line.strip('\n').split('\t')[0]
            feature1 = [float(i) for i in line.strip('\n').split('\t')[1:]]
            if uniprotid in dict_PPI:
                feature1.append(dict_PPI[uniprotid])
            else:
                feature1.append(0.0)
            one_feature.append(feature1)
            name_list.append(uniprotid)

    one_feature = torch.Tensor(one_feature)
    X_final = one_feature
    X_label_list = []
    for name in name_list:
        X_label = []
        if name in dict_at1:
            X_label.append(1)
        else:
            X_label.append(0)

        X_label_list.append(X_label)
    X_label = torch.Tensor(X_label_list)

    return X_final,X_label,name_list

X_final,X_label,name_list = prepare_feature()

model1 = load_checkpoint('model/model4')
y_pred = model1(X_final)
pre_label = X_label.tolist()
pre_value = y_pred.tolist()

with open('cos_feature.txt', 'w')as f:
    for i in range(len(pre_label)):
        f.write(name_list[i])
        for j in pre_value[i]:
            f.write('\t'+str(j))
        f.write('\n')
