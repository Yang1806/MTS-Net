"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import copy

import torch
import torch.nn as nn
import math
import numpy as np

from train.metrics import compute_score
from Data.get_shortest_path import floyd, shortest_path

def train_epoch(model, optimizer, device, data_loader, epoch, metric_f, dataset_type, task_num):
    model.train()
    epoch_loss = 0
    epoch_train_metric = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_f = batch_graphs.ndata['fpfeat'].to(device)
        batch_t = batch_graphs.ndata['treefeat'].to(device)

        # get adjacency matrix
        adj_matrices = batch_graphs.adjacency_matrix()
        a = adj_matrices._indices()
        nodes = batch_graphs.batch_num_nodes()
        edges = batch_graphs.batch_num_edges()
        adj = []
        num_edges = 0
        num_nodes = 0
        for i in range(len(nodes)):
            adj_i = np.zeros((nodes[i], nodes[i]))
            for j in range(edges[i]):
                adj_i[a[0][j+num_edges] - num_nodes][a[1][j+num_edges] - num_nodes] = 1
                adj_i[a[1][j+num_edges] - num_nodes][a[0][j+num_edges] - num_nodes] = 1
            num_edges += edges[i]
            num_nodes += nodes[i]
            adj.append(adj_i)

        # get shortest path
        path = []
        for i in range(len(adj)):
            next_node = floyd(adj[i])
            path_i = []
            for j in range(len(adj[i])):
                path_ij = []
                for k in range(len(adj[i])):
                    path_ijk = shortest_path(next_node, j, k)
                    path_ij.append(path_ijk)
                path_i.append(path_ij)
            path.append(path_i)

        old_path = copy.deepcopy(path)
        n = 0
        s = 0
        for i in range(len(path)):
            for j in range(len(path[i])):
                for k in range(len(path[i][j])):
                    for l in range(len(path[i][j][k])):
                        if path[i][j][k][l] != -1:
                            path[i][j][k][l] += s
            s += int(nodes[n])
            n += 1

        # get atom embedding on shortest path
        for i in range(len(path)):
            for j in range(len(path[i])):
                for k in range(len(path[i][j])):
                    for l in range(len(path[i][j][k])):
                        if path[i][j][k][l] != -1:
                            path[i][j][k][l] = batch_x[path[i][j][k][l]]

        # get tree encoding
        tree = []
        for i in range(batch_x.shape[0]):
            if sum(batch_t[i]) != 0:
                tree.append(batch_t[i])
        tree = torch.stack(tree)

        # get fingerprint encoding
        fp = []
        for i in range(batch_x.shape[0]):
            if sum(batch_f[i]) != 0:
                fp.append(batch_f[i])
        fp = torch.stack(fp)

        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()
        try:
            batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        try:
            batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
        except:
            batch_wl_pos_enc = None



        batch_scores = model.forward(batch_graphs, batch_x, batch_e, fp, tree, path, batch_lap_pos_enc, batch_wl_pos_enc)

        # mask = torch.Tensor([[x is not None for x in tb] for tb in batch_targets])
        # target = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch_targets])

        mask = torch.Tensor([[x != -1 for x in tb] for tb in batch_targets])
        target = torch.Tensor([[0 if x == -1 else x for x in tb] for tb in batch_targets])


        weight = torch.ones(target.shape)

        mask = mask.to(device)
        target = target.to(device)
        weight = weight.to(device)

        loss = model.loss(batch_scores, target, dataset_type, weight, mask)
        loss = loss.sum() / mask.sum()

        # loss = model.loss(batch_scores, batch_targets, dataset_type)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        batch_scores = batch_scores.cpu().detach().numpy().tolist()
        batch_targets = batch_targets.cpu().detach().numpy().tolist()

        epoch_train_metric += compute_score(batch_scores, batch_targets, metric_f, task_num)

        # nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_metric /= (iter + 1)
    
    return epoch_loss, epoch_train_metric, optimizer

def evaluate_network(model, device, data_loader, epoch, metric_f, dataset_type, task_num):
    model.eval()
    epoch_test_loss = 0
    epoch_test_metric = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_f = batch_graphs.ndata['fpfeat'].to(device)
            batch_t = batch_graphs.ndata['treefeat'].to(device)

            tree = []
            for i in range(batch_x.shape[0]):
                if sum(batch_t[i]) != 0:
                    tree.append(batch_t[i])
            tree = torch.stack(tree)

            fp = []
            for i in range(batch_x.shape[0]):
                if sum(batch_f[i]) != 0:
                    fp.append(batch_f[i])
            fp = torch.stack(fp)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata['lap_pos_enc'].to(device)
            except:
                batch_lap_pos_enc = None
            
            try:
                batch_wl_pos_enc = batch_graphs.ndata['wl_pos_enc'].to(device)
            except:
                batch_wl_pos_enc = None
                
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, fp, tree, epoch, batch_lap_pos_enc, batch_wl_pos_enc)

            mask = torch.Tensor([[x is not None for x in tb] for tb in batch_targets])
            target = torch.Tensor([[0 if x is None else x for x in tb] for tb in batch_targets])
            weight = torch.ones(target.shape)

            mask = mask.to(device)
            target = target.to(device)
            weight = weight.to(device)

            loss = model.loss(batch_scores, target, dataset_type, weight, mask)
            loss = loss.sum() / mask.sum()

            epoch_test_loss += loss.detach().item()

            batch_scores = batch_scores.cpu().detach().numpy().tolist()
            batch_targets = batch_targets.cpu().detach().numpy().tolist()

            epoch_test_metric += compute_score(batch_scores, batch_targets, metric_f, task_num)

            # nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_metric /= (iter + 1)
        
    return epoch_test_loss, epoch_test_metric

