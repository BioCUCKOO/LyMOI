import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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

def train(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        y_pred = model(x)
        loss = loss_function(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    pred_values, true_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            pred_values.append(y_pred)
            true_labels.append(y)
    pred_values = torch.cat(pred_values).cpu().numpy()
    true_labels = torch.cat(true_labels).cpu().numpy()
    return pred_values, true_labels

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def load_checkpoint(path: str):
    state = torch.load(path)
    loaded_state_dict = state['state_dict']
    model = GCN(50, 2)
    model_state_dict = model.state_dict()

    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if param_name in ['out.weight', 'out.bias', 'first.weight', 'first.bias']:
            continue
        if param_name not in model_state_dict:
            continue
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            continue
        pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    return model


class MyDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X[indices]
        self.y = y[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_folds(num_folds, data_size):
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    fold_size = data_size // num_folds
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]
    return folds

def plot_roc_curve(fpr, tpr, auc_value, fold_num):
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC fold {fold_num+1} (AUC = {auc_value:.4f})')

def save_checkpoint(path: str, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, path)

def cross_validate(X, y, model_class, num_folds=10, epochs=50, lr=0.0001):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    aucs = []
    all_true_labels = []
    all_pred_values = []
    all_names = []  # 用于保存基因名字

    fold_num = 1
    for train_index, test_index in skf.split(X, y[:, 1]):
        print(f"Fold {fold_num}/{num_folds}")

        train_dataset = MyDataset(X, y, train_index)
        test_dataset = MyDataset(X, y, test_index)
        train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=20)

        model = load_checkpoint('teacher_model')
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            loss = train(model, train_loader, loss_function, optimizer)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        save_checkpoint('model/model' + str(num_folds), model)

        pred_values, true_labels = evaluate(model, test_loader)
        auc = roc_auc_score(true_labels, pred_values)
        aucs.append(auc)
        print(f"Best AUC for Fold {fold_num}: {auc:.4f}")

        all_true_labels.append(true_labels)
        all_pred_values.append(pred_values)
        all_names.extend([name_list[i] for i in test_index])  # 添加基因名字

        fold_num += 1

    all_true_labels = np.vstack(all_true_labels)
    all_pred_values = np.vstack(all_pred_values)

    overall_auc = roc_auc_score(all_true_labels, all_pred_values)
    print(f"Cross-validation completed!")
    print(f"Mean AUC: {np.mean(aucs):.4f}, Std AUC: {np.std(aucs):.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")

    # 将基因名字、真实标签和预测标签一起保存
    results = np.column_stack((all_names, all_true_labels[:, 1], all_pred_values[:, 1]))
    np.savetxt('result/' + str(num_folds) + 'folds_predictions.txt', results, delimiter='\t',
               header="Gene_Name\tTrue_Label\tPred_Label", fmt="%s")

    return overall_auc


if __name__ == "__main__":
    X_final, name_list = prepare_feature()

    y = prepare_label(name_list)

    overall_auc = cross_validate(X_final, y, GCN, num_folds=4)

    print(f"Final Overall AUC: {overall_auc:.4f}")
