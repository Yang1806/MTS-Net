import torch.utils.data
import time
import dgl
from scipy import sparse as sp
import networkx as nx
import hashlib
import numpy as np
import torch
from dgl import DGLGraph
from rdkit import Chem
import csv
import random
from rdkit.Chem import AllChem

from Data.scaffold import scaffold_split
from Data.tree_encoder import get_tree_encoder
from nets.molecules_graph.pubchemfp import GetPubChemFPs



atom_type_max = 100
atom_f_dim = 132
atom_features_define = {
    'atom_symbol': list(range(32)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ], }


def get_bond_features(bond):
    bond_feats = []
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_feats = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_feats = 2
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_feats = 3
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_feats = 4
    else:
        bond_feats = 5
    return bond_feats


def onek_encoding_unk(key,length):
    if key == 34:
        key = 17
    if key == 52:
        key = 18
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1

    return encoding


def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
           onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
           onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0]
    return feature


def split_data(data, graph_labels, graph_lists, type, seed):
    if type == 'random':
        random.seed(seed)
        random.shuffle(data)
        random.seed(seed)
        random.shuffle(graph_labels)
        random.seed(seed)
        random.shuffle(graph_lists)
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))
        train_val_size = train_size + val_size

        train_data = data[:train_size]
        val_data = data[train_size:train_val_size]
        test_data = data[train_val_size:]

        train_labels = graph_labels[:train_size]
        val_labels = graph_labels[train_size:train_val_size]
        test_labels = graph_labels[train_val_size:]

        train_lists = graph_lists[:train_size]
        val_lists = graph_lists[train_size:train_val_size]
        test_lists = graph_lists[train_val_size:]


        return train_data, val_data, test_data, train_labels, val_labels, test_labels, train_lists, val_lists, test_lists
    elif type == 'scaffold':
        return scaffold_split(data, seed)
    else:
        raise ValueError('Split_type is Error.')


def load_data(seed, name):
    data = []
    graph_lists = []
    graph_labels = []

    h = get_tree_encoder(name)

    with open('./Data/molecules/' + name + '.csv') as f:
        f_csv = csv.reader(f)
        next(f_csv)
        for index, row in enumerate(f_csv):
            G = DGLGraph()
            smile = row[0]
            h_tree = h[index]
            mol = Chem.MolFromSmiles(smile)

            G.add_nodes(mol.GetNumAtoms())

            node_features = []
            edge_features = []

            for i in range(mol.GetNumAtoms()):
                atom_i = mol.GetAtomWithIdx(i)
                atom_i_features = get_atom_feature(atom_i)
                node_features.append(atom_i_features)

                for j in range(mol.GetNumAtoms()):
                    bond_ij = mol.GetBondBetweenAtoms(i, j)
                    if bond_ij is not None:
                        G.add_edges(i, j)
                        bond_features_ij = get_bond_features(bond_ij)
                        edge_features.append(bond_features_ij)

            G.ndata['feat'] = torch.from_numpy(np.array(node_features))
            G.edata['feat'] = torch.from_numpy(np.array(edge_features))


            # label = torch.tensor(float(row[1]))
            # label = torch.tensor([float(x) if x != '' else None for x in row[1:]])
            label = torch.tensor([float(x) if x != '' else -1 for x in row[1:]])


            graph_labels.append(label)

            bond = []
            for i in range(mol.GetNumAtoms()):
                bond1 = []
                for j in range(mol.GetNumAtoms()):
                    bond_ij = mol.GetBondBetweenAtoms(i, j)
                    if bond_ij is not None:
                        bond_features_ij = get_bond_features(bond_ij)
                        bond1.append(bond_features_ij)
                    else:
                        bond1.append(0)
                bond.append(bond1)
            bond = torch.tensor(bond)

            set = {
                'num_atom': mol.GetNumAtoms(),
                'atom_type': G.ndata['feat'],
                'bond_type': bond
            }
            data.append(set)

            fp = []
            fp_list = []
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_pubcfp = GetPubChemFPs(mol)

            fp.extend(fp_maccs)
            fp.extend(fp_pubcfp)
            fp_list.append(fp)
            pad1 = [0]*len(fp)
            for _ in range(mol.GetNumAtoms()-1):
                fp_list.append(pad1)

            h_tree_list = []
            h_tree_list.append(h_tree)
            pad2 = [0]*len(h_tree)
            for _ in range(mol.GetNumAtoms()-1):
                h_tree_list.append(pad2)

            fp_list = torch.Tensor(fp_list)
            G.ndata['fpfeat'] = torch.from_numpy(np.array(fp_list))
            h_tree_list = torch.Tensor(h_tree_list)
            G.ndata['treefeat'] = torch.from_numpy(np.array(h_tree_list))

            graph_lists.append(G)

    train_data, val_data, test_data, train_labels, val_labels, test_labels, train_lists, val_lists, test_lists = split_data(data, graph_labels, graph_lists, 'random', seed)
    train = MoleculeDGL(train_data, train_labels, train_lists, './Data/molecules', 'train', len(train_data))
    val = MoleculeDGL(val_data, val_labels, val_lists, './Data/molecules', 'val', len(val_data))
    test = MoleculeDGL(test_data, test_labels, test_lists, './Data/molecules', 'test', len(test_data))
    return train, val, test


class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data, label, list, data_dir, split, num_graphs=None):
        self.data_dir = data_dir
        self.split = split
        self.num_graphs = num_graphs
        self.n_samples = num_graphs
        self.data = data
        self.graph_labels = label
        self.graph_lists = list

        
        """
        Data is a list of Molecule dict objects with following attributes
        
          molecule = Data[idx]
        ; molecule['num_atom'] : nb of atoms, an integer (N)
        ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
        ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
        ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
        """
        
        # self.graph_lists = []
        # self.graph_labels = []
        # self.n_samples = len(self.Data)
        # self._prepare()
    
    def _prepare(self):
        print("preparing %d graphs for the %s set..." % (self.num_graphs, self.split.upper()))
        
        for molecule in self.data:
            node_features = molecule['atom_type'].long()
            
            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])
            g.ndata['feat'] = node_features
            
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features
            
            self.graph_lists.append(g)
            self.graph_labels.append(molecule['logP_SA_cycle_normalized'])
        
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.graph_lists[idx], self.graph_labels[idx]
    
    


def self_loop(g):

    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


def make_full_graph(g):


    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))
    full_g.ndata['feat'] = g.ndata['feat']
    full_g.edata['feat'] = torch.zeros(full_g.number_of_edges()).long()
    
    try:
        full_g.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
    except:
        pass

    try:
        full_g.ndata['wl_pos_enc'] = g.ndata['wl_pos_enc']
    except:
        pass    
    
    return full_g



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    if EigVec.shape[0] < pos_enc_dim+1:
        EigVec = np.pad(EigVec, ((0, 0), (0, pos_enc_dim + 1 - EigVec.shape[0])), 'constant')
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()
    
    return g

def wl_positional_encoding(g):
    """
        WL-based absolute positional embedding 
        adapted from 
        
        "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
        Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
        https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1


    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1
        
    g.ndata['wl_pos_enc'] = torch.LongTensor(list(node_color_dict.values()))
    return g


class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, name, seed):
        start = time.time()
        print("Loading dataset %s..." % (name))
        self.seed = seed
        self.name = name
        train, val, test = load_data(seed, name)
        self.train = train
        self.val = val
        self.test = test
        # self.num_atom_type = 32
        self.num_bond_type = 5
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        if labels[0].shape[0] == 1:
            labels = torch.tensor(np.array(labels)).unsqueeze(1)
        else:
            labels = torch.stack(labels)
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _make_full_graph(self):
        
        # function for converting graphs to full graphs
        # this function will be called only if full_graph flag is True
        self.train.graph_lists = [make_full_graph(g) for g in self.train.graph_lists]
        self.val.graph_lists = [make_full_graph(g) for g in self.val.graph_lists]
        self.test.graph_lists = [make_full_graph(g) for g in self.test.graph_lists]
    
    
    def _add_laplacian_positional_encodings(self, pos_enc_dim):
        
        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):
        
        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]
