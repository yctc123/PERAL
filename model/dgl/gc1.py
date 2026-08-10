from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from .discriminator import Discriminator
from .batch_gru import BatchGRU
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id, ent2rels):                                                              
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.ent2rels = ent2rels
        self.gnn = RGCN(params)                                              

        self.rel_emb = nn.Embedding(self.params.num_rels + 1, self.params.inp_dim, sparse=False, padding_idx=self.params.num_rels)

        self.ent_padding = nn.Parameter(torch.FloatTensor(1, self.params.sem_dim).uniform_(-1, 1))
        if self.params.init_nei_rels == 'both':
            self.w_rel2ent = nn.Linear(2 * self.params.inp_dim, self.params.sem_dim)
        elif self.params.init_nei_rels == 'out' or 'in':
            self.w_rel2ent = nn.Linear(self.params.inp_dim, self.params.sem_dim)

        self.sigmoid = nn.Sigmoid()
        self.nei_rels_dropout = nn.Dropout(self.params.nei_rels_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.q_proj = nn.Linear(self.params.emb_dim,self.params.emb_dim)
        self.k_proj = nn.Linear(self.params.emb_dim, self.params.emb_dim)
        self.v_proj = nn.Linear(self.params.emb_dim, self.params.emb_dim)
        self.no_path_emb = nn.Parameter(torch.zeros(1, self.params.emb_dim))
        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

        if self.params.comp_hrt:
            self.fc_layer = nn.Linear(2 * self.params.num_gcn_layers * self.params.emb_dim, 1)

        if self.params.nei_rel_path:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + 2 * self.params.emb_dim, 1)

        if self.params.comp_ht == 'mlp':
            self.fc_comp = nn.Linear(2 * self.params.emb_dim, self.params.emb_dim)

        if self.params.nei_rel_path:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim, self.params.num_gcn_layers * self.params.emb_dim + self.params.emb_dim)
        else:
            self.disc = Discriminator(self.params.num_gcn_layers * self.params.emb_dim , self.params.num_gcn_layers * self.params.emb_dim)

        self.rnn = torch.nn.GRU(self.params.emb_dim, self.params.emb_dim, batch_first=True)

        self.batch_gru = BatchGRU(self.params.num_gcn_layers * self.params.emb_dim )

        self.W_o = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim * 2, self.params.num_gcn_layers * self.params.emb_dim)

    def init_ent_emb_matrix(self, g):
        """Helper function."""
        out_nei_rels = g.ndata['out_nei_rels']
        in_nei_rels = g.ndata['in_nei_rels']
        target_rels = g.ndata['r_label']
        out_nei_rels_emb = self.rel_emb(out_nei_rels)
        in_nei_rels_emb = self.rel_emb(in_nei_rels)
        target_rels_emb = self.rel_emb(target_rels).unsqueeze(2)
        out_atts = self.softmax(self.nei_rels_dropout(torch.matmul(out_nei_rels_emb, target_rels_emb).squeeze(2)))
        in_atts = self.softmax(self.nei_rels_dropout(torch.matmul(in_nei_rels_emb, target_rels_emb).squeeze(2)))
        out_sem_feats = torch.matmul(out_atts.unsqueeze(1), out_nei_rels_emb).squeeze(1)
        in_sem_feats = torch.matmul(in_atts.unsqueeze(1), in_nei_rels_emb).squeeze(1)
        if self.params.init_nei_rels == 'both':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(torch.cat([out_sem_feats, in_sem_feats], dim=1)))
        elif self.params.init_nei_rels == 'out':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(out_sem_feats))
        elif self.params.init_nei_rels == 'in':
            ent_sem_feats = self.sigmoid(self.w_rel2ent(in_sem_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_sem_feats], dim=1)                     
    def comp_ht_emb(self, head_embs, tail_embs):
        if self.params.comp_ht == 'mult':
            ht_embs = head_embs * tail_embs
        elif self.params.comp_ht == 'mlp':
            ht_embs = self.fc_comp(torch.cat([head_embs, tail_embs], dim=1))
        elif self.params.comp_ht == 'sum':
            ht_embs = head_embs + tail_embs
        else:
            raise KeyError(f'composition operator of head and relation embedding {self.comp_ht} not recognized.')

        return ht_embs

    def comp_hrt_emb(self, head_embs, tail_embs, rel_embs):
        rel_embs = rel_embs.repeat(1, self.params.num_gcn_layers)
        if self.params.comp_hrt == 'TransE':
            hrt_embs = head_embs + rel_embs - tail_embs
        elif self.params.comp_hrt == 'DistMult':
            hrt_embs = head_embs * rel_embs * tail_embs
        else: raise KeyError(f'composition operator of (h, r, t) embedding {self.comp_hrt} not recognized.')

        return hrt_embs


    """Helper function."""


    def nei_rel_path(self, g, rel_labels, r_emb_out,precomputed_paths):
        'Helper function.'
        device = rel_labels.device
        batch_size = rel_labels.shape[0]
        MAX_PATHS = precomputed_paths.shape[1]    
        PATH_LEN = precomputed_paths.shape[2]     
        PAD_ID = self.params.num_rels

        path_embs = F.embedding(precomputed_paths.view(-1, PATH_LEN), r_emb_out)

        pad_mask = (precomputed_paths.view(-1, PATH_LEN) == PAD_ID).unsqueeze(-1)
        path_embs = path_embs.masked_fill(pad_mask, 0.0)

        _, hn = self.rnn(path_embs)

        path_reprs = hn.squeeze(0).view(batch_size, MAX_PATHS, -1)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_embs = g.ndata['repr'][tail_ids]
        target_rel_embs = F.embedding(rel_labels, r_emb_out, padding_idx=-1).squeeze(1)

        query_input =   torch.cat([ target_rel_embs], dim=1)                                      

        query = self.q_proj(query_input).unsqueeze(1)
        keys = self.k_proj(path_reprs)
        values = self.v_proj(path_reprs)

        scores = torch.matmul(query, keys.transpose(-2, -1)) / np.sqrt(path_reprs.size(-1))
        atts = F.softmax(scores.squeeze(1), dim=1).unsqueeze(1)             

        path_output = torch.matmul(atts, values).squeeze(1)

        has_path_mask = (precomputed_paths != PAD_ID).any(dim=2).any(dim=1).float().unsqueeze(-1)

        final_output = path_output * has_path_mask
        if hasattr(self, 'no_path_emb'):
            final_output = final_output + self.no_path_emb * (1.0 - has_path_mask)

        return final_output


    def get_logits(self, s_G, s_g_pos, s_g_cor):
        ret = self.disc(s_G, s_g_pos, s_g_cor)
        return ret

    def forward(self, data, is_return_emb=False, cor_graph=False):                                                      
        g, rel_labels, precomputed_paths = data

        if self.params.init_nei_rels == 'no':
            g.ndata['init'] = g.ndata['feat'].clone()
        else:
            self.init_ent_emb_matrix(g)      

        if cor_graph:
            g.ndata['init'] = g.ndata['init'][torch.randperm(g.ndata['feat'].shape[0])]

        r = self.rel_emb.weight.clone()

        g.ndata['h'], r_emb_out = self.gnn(g, r)

        graph_sizes = g.batch_num_nodes()
        out_dim = self.params.num_gcn_layers * self.params.emb_dim
        g.ndata['repr'] = F.relu(self.batch_gru(g.ndata['repr'].view(-1, out_dim), graph_sizes))
        node_hiddens = F.relu(self.W_o(g.ndata['repr']))                      
        g.ndata['repr'] = self.dropout(node_hiddens)                      
        g_out = mean_nodes(g, 'repr').view(-1, out_dim)

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.params.add_ht_emb:
            g_rep = torch.cat([g_out,
                               head_embs.view(-1, out_dim),
                               tail_embs.view(-1, out_dim),
                               F.embedding(rel_labels, r_emb_out, padding_idx=-1).squeeze(1)], dim=1)                       
        else:
            g_rep = torch.cat([g_out, self.rel_emb(rel_labels)], dim=1)

        if self.params.comp_hrt:
            edge_embs = self.comp_hrt_emb(head_embs.view(-1, out_dim), tail_embs.view(-1, out_dim), F.embedding(rel_labels, r_emb_out, padding_idx=-1))
            g_rep = torch.cat([g_out, edge_embs], dim=1)

        if self.params.nei_rel_path:
            g_p = self.nei_rel_path(g, rel_labels, r_emb_out, precomputed_paths)
            g_rep = torch.cat([g_rep, g_p], dim=1)
            s_g = torch.cat([g_out, g_p], dim=1)
        else:
            s_g = g_out
        output = self.fc_layer(g_rep).squeeze(-1)

        self.r_emb_out = r_emb_out
        if not is_return_emb:
            return output
        else:
            s_G = s_g.mean(0)
            return output, s_G, s_g
