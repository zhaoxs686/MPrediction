from torch.utils.data import Dataset
import tqdm
import json
import torch
import dgl
import random
import numpy as np
import pickle
from sklearn.utils import shuffle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
import pickle
import os
from feature import mol2alt_sentence
from scipy import io as sio

str2num = {'<pad>':0 ,'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'S': 6, 'Cl': 7, 'P': 8, 'Br':  9,
         'B': 10,'I': 11,'Si':12,'Se':13,'<unk>':14,'<mask>':15,'<global>':16}

num2str =  {i:j for j,i in str2num.items()}

class BERTDataset(Dataset):
    def __init__(self, corpus_path, corpus_label, word2idx_path, seq_len, hidden_dim=300):
        # hidden dimension for positional encoding
        self.hidden_dim = hidden_dim

        # define path of dicts
        self.word2idx_path = word2idx_path

        # define max length
        self.seq_len = seq_len

        # directory of corpus dataset
        self.corpus_path = corpus_path
        self.corpus_label = corpus_label
        # define special symbols
        self.pad_index = 0
        self.unk_index = 1
        self.cls_index = 2
        self.sep_index = 3
        self.mask_index = 4

        # 加载语料
        ident = open(word2idx_path, 'rb')
        self.ident_dict = pickle.load(ident)

        with open(corpus_path, "r", encoding="utf-8") as f:

            # 将数据集全部加载到内存
            self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset")]
            self.corpus_lines = len(self.lines)

        label = sio.loadmat(self.corpus_label)
        label = label['label']
        self.tlabel = np.array(label, dtype=np.int32)
        num_p = len(np.where(label == 1)[0])
        num_n = len(np.where(label == 0)[0])
        self.weight = torch.tensor([num_p * 1.0 / num_n, 1.0])

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, d_structure = self.get_corpus_line(item)
        char_tokens_ = t1
        t1_random = self.tokenize_char(char_tokens_)
        t1 = [self.cls_index] + t1_random + [self.sep_index]

        bert_input = t1[:self.seq_len]

        output = {
            "bert_input": torch.tensor(bert_input),
            "graph_input": d_structure,
            "tlabel": torch.tensor(self.tlabel[item]),
        }
        return output

    # Structure model
    def get_d_structure(self, SMILES):
        from dgllife.utils import smiles_to_bigraph, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
        drug_node_featurizer = AttentiveFPAtomFeaturizer()
        drug_bond_featurizer = AttentiveFPBondFeaturizer(self_loop=True)
        from functools import partial
        fc = partial(smiles_to_bigraph, add_self_loop=True)
        D_Structure = fc(smiles=SMILES, node_featurizer=drug_node_featurizer, edge_featurizer=drug_bond_featurizer)

        return D_Structure


    def tokenize_char(self, segments):
        return [self.ident_dict.get(char, self.unk_index) for char in segments]


    def standardizeAndcanonical(self, smi):
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        # standardize
        mol = Chem.MolFromSmiles(smi)
        mol2 = lfc.choose(mol)
        smi2 = Chem.MolToSmiles(mol2)
        return smi2

    def get_corpus_line(self, item):

        smiles = self.lines[item]
        if smiles[-1] == "\n":
            smiles = smiles[:-1]
        if smiles[-1] == " ":
            smiles = smiles[:-1]
        smiles = self.standardizeAndcanonical(smiles)
        d_structure = self.get_d_structure(smiles)
        t = Chem.MolFromSmiles(smiles)
        sentence_1 = mol2alt_sentence(t, 1)

        return sentence_1, d_structure