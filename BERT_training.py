from torch.optim import Adam
from torch.utils.data import DataLoader
import dgl
import torch
import sys
sys.path.append("../")
from units import BERTDataset
from BERT import *
import tqdm
import pandas as pd
import numpy as np
import os
import codecs
from scipy import io as sio
from subword_nmt.apply_bpe import BPE
from sklearn import model_selection
from rdkit.Chem import AllChem


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax((abs(tpr - fpr)))
    # max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, precision]


class DGL_AttentiveFP(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers = 2, num_timesteps = 2, graph_feat_size = 200, predictor_dim=None):
        super(DGL_AttentiveFP, self).__init__()
        from dgllife.model.gnn import AttentiveFPGNN
        from dgllife.model.readout import AttentiveFPReadout
        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps)

    def forward(self, bg):
        with bg.local_scope():
            bg = bg.to(device)
            node_feats = bg.ndata.pop('h')
            edge_feats = bg.edata.pop('e')
            node_feats = self.gnn(bg, node_feats, edge_feats)
            graph_feats = self.readout(bg, node_feats, False)

        return graph_feats

class GB_model(nn.Module):
    def __init__(self, bertconfig, G_dim, B_dim):
        super(GB_model, self).__init__()

        self.graphencoder = DGL_AttentiveFP(node_feat_size=39,edge_feat_size=11,num_layers=3,num_timesteps=2,graph_feat_size=128)
        self.Smilesencoder = BertModel(config=bertconfig)

        self.transform1 = nn.Linear(G_dim+B_dim, 128)
        self.transform2 = nn.Tanh()
        self.transform3 = nn.Dropout(p=0.1)
        self.transform4 = nn.Linear(128, 2)
    def forward(self, bg,  input_ids, positional_enc):
            graph_embedding = self.graphencoder(bg)
            m_embeddding, smile_embedding =self.Smilesencoder(input_ids, positional_enc)
            final = torch.cat((graph_embedding,smile_embedding), dim=1).to(device)
            embedding = self.transform3(self.transform2(self.transform1(final)))
            logit = self.transform4(embedding)

            return final, graph_embedding, smile_embedding, logit

def init_positional_encoding(hidden_dim, max_seq_len):
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / hidden_dim) for i in range(hidden_dim)]
        if pos != 0 else np.zeros(hidden_dim) for pos in range(max_seq_len)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    position_enc = position_enc / (denominator + 1e-8) #对position_enc做一个归一化
    position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
    return position_enc

def main(config):

    train_dataset = BERTDataset(corpus_path=config["train_corpus_path"],
                                corpus_label=config["train_corpus_label"],
                                 word2idx_path=config["word2idx_path"],
                                seq_len=config["max_seq_len"],
                                hidden_dim=config["hidden_size"],
                                )
    # 初始化训练dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config["batch_size"],
                                  collate_fn=lambda x: x)


    # 初始化超参数
    bertconfig = BertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],  # 隐藏层维度也就是字向量维度
        num_hidden_layers=config["num_hidden_layers"],  # transformer block 的个数
        num_attention_heads=config["num_attention_heads"],  # 注意力机制"头"的个数
        intermediate_size=config["intermediate_size"],  # feedforward层线性映射的维度
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.5
    )
    # 初始化bert模型
    model = GB_model(bertconfig,config["G_dim"], config["hidden_size"])
    model.to(device)
    # 初始化positional encoding
    positional_enc = init_positional_encoding(hidden_dim=bertconfig.hidden_size,
                                                   max_seq_len=config["max_seq_len"])
    positional_enc = torch.unsqueeze(positional_enc, dim=0).to(device)

    optim_parameters = list(model.parameters())
    optimizer = torch.optim.Adam(optim_parameters, lr=config["lr"])
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(train_dataset.weight).to(device))
    for epoch in range(config["train_epoches"]):
        model.train()

        for step, data in enumerate(train_dataloader):
            # print('IDX of data_iter:', i)
            bert_input = [i["bert_input"] for i in data]
            graph_input = [i["graph_input"] for i in data]
            tlabel = [i["tlabel"] for i in data]
            g_train = dgl.batch(graph_input).to(device)
            bert_input = torch.nn.utils.rnn.pad_sequence(bert_input, batch_first=True).to(device)

            pos_embedding = positional_enc[:, :bert_input.size()[-1], :].to(device)

            final, Gembedding, Bembedding, logit = model(g_train, bert_input,pos_embedding)

            loss = loss_func(logit, torch.tensor(tlabel).long().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1:05d} | Loss: {loss:.4f}")

    model.to("cpu")
    torch.save(model.state_dict(), f"./models/pretain2.pt")



if __name__ == '__main__':

    # 配置bert模型参数
    config = {}
    config["train_corpus_path"] = "./data/pretrain_smiles.txt"
    config["train_corpus_label"] = "./data/pretrain_label.mat"
    config["word2idx_path"] = "ident_base_r2m.pickle"
    config["vocab_size"] = 1525
    config["batch_size"] = 300
    config["max_seq_len"] = 100

    config["lr"] = 5e-4
    config["train_epoches"] =300
    config["G_dim"] = 128
    config["hidden_size"] = 150
    config["num_hidden_layers"] = 6
    config["num_attention_heads"] = 6
    config["intermediate_size"] = 1525

    main(config)