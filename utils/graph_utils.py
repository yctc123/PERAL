import statistics
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    'Helper function.'

    g_nx = nx.MultiDiGraph()        
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    for rel, adj in enumerate(graph):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    graphs_pos, g_labels_pos, r_labels_pos, graphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)

    return (batched_graph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, r_labels_neg), g_labels_neg


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    g_dgl_neg = send_graph_to_device(g_dgl_neg, device)


    return ((g_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, r_labels_neg), targets_neg)


def collate_dgl_train(samples):
    graphs_pos, g_labels_pos, r_labels_pos, paths_pos,\
    graphs_negs, g_labels_negs, r_labels_negs, paths_negs = map(list, zip(*samples))

    batched_graph_pos = dgl.batch(graphs_pos)
    batched_bi_graph_pos = dgl.batch([g.bi_graph for g in graphs_pos])

    batched_graph_cor = dgl.batch(graphs_pos)             
    batched_bi_graph_cor = dgl.batch([g.bi_graph for g in graphs_pos])

    batched_paths_pos = torch.stack(paths_pos, dim=0)

    graphs_neg_flat = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg_flat = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg_flat = [item for sublist in r_labels_negs for item in sublist]
    paths_neg_flat = [item for sublist in paths_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg_flat)
    batched_bi_graph_neg = dgl.batch([g.bi_graph for g in graphs_neg_flat])

    batched_paths_neg = torch.stack(paths_neg_flat, dim=0)

    return (batched_graph_pos, batched_bi_graph_pos, r_labels_pos, batched_paths_pos), g_labels_pos,\
           (batched_graph_neg, batched_bi_graph_neg, r_labels_neg_flat, batched_paths_neg), g_labels_neg_flat,\
           (batched_graph_cor, batched_bi_graph_cor, r_labels_pos, batched_paths_pos)

def move_batch_to_device_dgl_train(batch, device):
    (data_pos, targets_pos, data_neg, targets_neg, data_cor) = batch

    g_dgl_pos, bi_graph_pos, r_labels_pos, paths_pos = data_pos
    g_dgl_neg, bi_graph_neg, r_labels_neg, paths_neg = data_neg
    g_dgl_cor, bi_graph_cor, r_labels_cor, paths_cor = data_cor

    bi_graph_pos = send_graph_to_device(bi_graph_pos, device)
    bi_graph_neg = send_graph_to_device(bi_graph_neg, device)
    bi_graph_cor = send_graph_to_device(bi_graph_cor, device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    g_dgl_neg = send_graph_to_device(g_dgl_neg, device)
    g_dgl_cor = send_graph_to_device(g_dgl_cor, device)

    targets_pos = torch.LongTensor(targets_pos).to(device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device)
    paths_pos = paths_pos.to(device)

    targets_neg = torch.LongTensor(targets_neg).to(device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device)
    paths_neg = paths_neg.to(device)

    paths_cor = paths_cor.to(device)

    return ((g_dgl_pos, bi_graph_pos, r_labels_pos, paths_pos), targets_pos,
            (g_dgl_neg, bi_graph_neg, r_labels_neg, paths_neg), targets_neg,
            (g_dgl_cor, bi_graph_cor, r_labels_pos, paths_cor))


def collate_dgl_train_cor(samples):


    (graphs_pos, g_labels_pos, r_labels_pos, paths_pos,
     graphs_negs, g_labels_negs, r_labels_negs, paths_negs,
     graphs_cont1, g_labels_cont1, r_labels_cont1, paths_cont1,
     graphs_cont2, g_labels_cont2, r_labels_cont2, paths_cont2) = map(list, zip(*samples))

    batched_graph_pos = dgl.batch(graphs_pos)
    batched_bi_graph_pos = dgl.batch([g.bi_graph for g in graphs_pos])
    batched_paths_pos = torch.stack(paths_pos, dim=0)                             

    graphs_neg_flat = [item for sublist in graphs_negs for item in sublist]
    g_labels_neg_flat = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg_flat = [item for sublist in r_labels_negs for item in sublist]
    paths_neg_flat = [item for sublist in paths_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg_flat)
    batched_bi_graph_neg = dgl.batch([g.bi_graph for g in graphs_neg_flat])
    batched_paths_neg = torch.stack(paths_neg_flat, dim=0)                                

    batched_graph_cont1 = dgl.batch(graphs_cont1)
    batched_bi_graph_cont1 = dgl.batch([g.bi_graph for g in graphs_cont1])
    batched_paths_cont1 = torch.stack(paths_cont1, dim=0)                         

    batched_graph_cont2 = dgl.batch(graphs_cont2)
    batched_bi_graph_cont2 = dgl.batch([g.bi_graph for g in graphs_cont2])
    batched_paths_cont2 = torch.stack(paths_cont2, dim=0)                         

    return ((batched_graph_pos, batched_bi_graph_pos, r_labels_pos, batched_paths_pos), g_labels_pos,
            (batched_graph_neg, batched_bi_graph_neg, r_labels_neg_flat, batched_paths_neg), g_labels_neg_flat,
            (batched_graph_cont1, batched_bi_graph_cont1, r_labels_pos, batched_paths_cont1),
            (batched_graph_cont2, batched_bi_graph_cont2, r_labels_pos, batched_paths_cont2))


def move_batch_to_device_dgl_train_cor(batch, device):


    (data_pos, targets_pos,
     data_neg, targets_neg,
     data_cont1, data_cont2) = batch

    g_dgl_pos, bi_graph_pos, r_labels_pos, paths_pos = data_pos
    g_dgl_neg, bi_graph_neg, r_labels_neg, paths_neg = data_neg
    g_dgl_cont1, bi_graph_cont1, r_labels_cont1, paths_cont1 = data_cont1
    g_dgl_cont2, bi_graph_cont2, r_labels_cont2, paths_cont2 = data_cont2

    bi_graph_pos = send_graph_to_device(bi_graph_pos, device)
    bi_graph_neg = send_graph_to_device(bi_graph_neg, device)
    bi_graph_cont1 = send_graph_to_device(bi_graph_cont1, device)
    bi_graph_cont2 = send_graph_to_device(bi_graph_cont2, device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    g_dgl_neg = send_graph_to_device(g_dgl_neg, device)
    g_dgl_cont1 = send_graph_to_device(g_dgl_cont1, device)
    g_dgl_cont2 = send_graph_to_device(g_dgl_cont2, device)

    targets_pos = torch.LongTensor(targets_pos).to(device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device)
    paths_pos = paths_pos.to(device)

    targets_neg = torch.LongTensor(targets_neg).to(device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device)
    paths_neg = paths_neg.to(device)

    paths_cont1 = paths_cont1.to(device)
    paths_cont2 = paths_cont2.to(device)

    return ((g_dgl_pos, bi_graph_pos, r_labels_pos, paths_pos), targets_pos,
            (g_dgl_neg, bi_graph_neg, r_labels_neg, paths_neg), targets_neg,
            (g_dgl_cont1, bi_graph_cont1, r_labels_pos, paths_cont1),
            (g_dgl_cont2, bi_graph_cont2, r_labels_pos, paths_cont2))


def send_graph_to_device(g, device):

    g = g.to(device)
    return g


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())
