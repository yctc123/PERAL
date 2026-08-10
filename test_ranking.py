import os
import random
import argparse
import logging
import json
import hashlib
import time

import scipy.sparse as ssp
from tqdm import tqdm
import torch
import numpy as np
import dgl


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            if triplet[1] in saved_relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], saved_relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    num_rels = len(id2relation)
    num_ents = len(entity2id)
    h2r = {}
    h2r_len = {}
    t2r = {}
    t2r_len = {}
    for triplet in triplets['graph']:
        h, t, r = triplet
        if h not in h2r:
            h2r_len[h] = 1
            h2r[h] = [r]
        else:
            h2r_len[h] += 1
            h2r[h].append(r)

        if t not in t2r:
            t2r[t] = [r]
            t2r_len[t]  = 1
        else:
            t2r[t].append(r)
            t2r_len[t] += 1


    h_nei_rels_len = int(np.percentile(list(h2r_len.values()), 75))
    t_nei_rels_len = int(np.percentile(list(t2r_len.values()), 75))
    print("Average number of relations each node: ", "head: ", h_nei_rels_len, 'tail: ', t_nei_rels_len)


    m_h2r = np.ones([num_ents, h_nei_rels_len]) * num_rels
    for ent, rels in h2r.items():
        if len(rels) > h_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), h_nei_rels_len)]
            m_h2r[ent] = rels
        else:
            rels = np.array(rels)
            m_h2r[ent][: rels.shape[0]] = rels

    m_t2r = np.ones([num_ents, t_nei_rels_len]) * num_rels
    for ent, rels in t2r.items():
        if len(rels) > t_nei_rels_len:
            rels = np.array(rels)[np.random.choice(np.arange(len(rels)), t_nei_rels_len)]
            m_t2r[ent] = rels
        else:
            rels = np.array(rels)
            m_t2r[ent][: rels.shape[0]] = rels


    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t
    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, m_h2r, m_t2r


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, id2relation, params, node_features, kge_entity2id, m_h2r, m_t2r):
    global model_, adj_list_, dgl_adj_list_, id2entity_, id2relation_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_
    model_, adj_list_, dgl_adj_list_, id2entity_, id2relation_, params_, node_features_, kge_entity2id_, m_h2r_, m_t2r_ = model, adj_list, dgl_adj_list, id2entity, id2relation, params, node_features, kge_entity2id, m_h2r, m_t2r


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(ruleN_pred_path, entity2id, saved_relation2id):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split('\n')[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {'head': [[], 10000], 'tail': [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = entity2id[pred_data[3 * i][0]], saved_relation2id[pred_data[3 * i][1]], entity2id[pred_data[3 * i][2]]
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet['head'][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet['head'][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet['tail'][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet['tail'][1] = j

            neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
            neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


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


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited                  

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])                             
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)
    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing_subgraph_nodes


from scipy.sparse import coo_matrix
def ssp_multigraph_to_dgl(graph, n_feats=None):
    num_nodes = graph[0].shape[0]

    src_list = []
    dst_list = []
    etype_list = []

    for rel, adj in enumerate(graph):
        if not isinstance(adj, coo_matrix):
            adj = adj.tocoo()
        src_list.append(adj.row)
        dst_list.append(adj.col)
        etype_list.append(np.full(adj.nnz, rel, dtype=np.int64))

    src = np.concatenate(src_list) if src_list else np.array([], dtype=np.int64)
    dst = np.concatenate(dst_list) if dst_list else np.array([], dtype=np.int64)
    etypes = np.concatenate(etype_list) if etype_list else np.array([], dtype=np.int64)

    g_dgl = dgl.graph((src, dst), num_nodes=num_nodes)

    g_dgl.edata['type'] = torch.from_numpy(etypes)

    if n_feats is not None:
        if not isinstance(n_feats, torch.Tensor):
            n_feats = torch.tensor(n_feats, dtype=torch.float32)
        g_dgl.ndata['feat'] = n_feats

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1
    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

    head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
    tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1        
    n_ids[tail_id] = 2        
    subgraph.ndata['id'] = torch.FloatTensor(n_ids)

    return subgraph


_llm_entity_text_ = None
_llm_relation_text_ = None
_llm_path_cache_data_ = None
_llm_tokenizer_ = None
_llm_local_model_ = None
_llm_local_failed_ = False


def _default_llm_desc_dir(dataset):
    base_dir = os.path.join(os.path.dirname(__file__), 'subgraph_extraction')
    candidates = [
        os.path.join(base_dir, dataset),
        os.path.join(base_dir, dataset.lower()),
        os.path.join(base_dir, dataset.upper()),
    ]
    if dataset.lower().startswith('fb'):
        candidates.append(os.path.join(base_dir, 'Fb237'))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def _load_text_map(file_path):
    mapping = {}
    if not file_path or not os.path.isfile(file_path):
        return mapping
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def _ensure_llm_resources():
    global _llm_entity_text_, _llm_relation_text_, _llm_path_cache_data_
    if _llm_entity_text_ is not None and _llm_relation_text_ is not None:
        return
    desc_dir = getattr(params_, 'llm_path_desc_dir', None) or _default_llm_desc_dir(getattr(params_, 'dataset', ''))
    _llm_entity_text_ = _load_text_map(os.path.join(desc_dir, 'entity2text.txt')) if desc_dir else {}
    _llm_relation_text_ = _load_text_map(os.path.join(desc_dir, 'relation2text.txt')) if desc_dir else {}
    _llm_path_cache_data_ = {}
    cache_path = getattr(params_, 'llm_path_cache', None)
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'key' in item and 'ranked_indices' in item:
                    if 'importance_scores' in item:
                        _llm_path_cache_data_[item['key']] = {
                            'ranked_indices': item['ranked_indices'],
                            'importance_scores': item.get('importance_scores', {})
                        }
                    else:
                        _llm_path_cache_data_[item['key']] = item['ranked_indices']


def _entity_text(global_id):
    name = id2entity_.get(int(global_id), str(global_id))
    desc = _llm_entity_text_.get(name, '') if _llm_entity_text_ is not None else ''
    return f"{name} ({desc})" if desc and desc != name else name


def _relation_text(rel_id):
    rel_id = int(rel_id)
    name = id2relation_.get(rel_id, str(rel_id))
    desc = _llm_relation_text_.get(name, '') if _llm_relation_text_ is not None else ''
    return f"{name} ({desc})" if desc and desc != name else name


def _path_to_llm_item(idx, path):
    rel_seq = [int(r) for r in path['rel_seq'] if int(r) != model_.params.num_rels]
    parent_ids = path['global_ent_seq']
    steps = []
    for step_idx, rel_id in enumerate(rel_seq):
        src = _entity_text(parent_ids[step_idx]) if step_idx < len(parent_ids) else ''
        dst = _entity_text(parent_ids[step_idx + 1]) if step_idx + 1 < len(parent_ids) else ''
        steps.append(f"{src} -- {_relation_text(rel_id)} --> {dst}")
    return {'index': idx, 'length': len(rel_seq), 'path': ' ; '.join(steps)}


def _llm_cache_key(head_global, tail_global, target_rel, candidates):
    payload = {
        'head': int(head_global),
        'tail': int(tail_global),
        'rel': int(target_rel),
        'paths': [
            {
                'rel_seq': [int(r) for r in p['rel_seq']],
                'global_ent_seq': [int(e) for e in p['global_ent_seq']]
            }
            for p in candidates
        ]
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()


def _build_path_rank_messages(head_global, tail_global, target_rel, candidates):
    path_items = [_path_to_llm_item(i, path) for i, path in enumerate(candidates)]
    original_indices = [item['index'] for item in path_items]
    user_payload = {
        'query': {
            'head': _entity_text(head_global),
            'relation': _relation_text(target_rel),
            'tail': _entity_text(tail_global)
        },
        'candidate_paths': path_items
    }
    return [
        {
            'role': 'system',
            'content': (
                'You are a knowledge-graph path ranker. Return exactly one JSON object '
                'with only ranked_indices and importance_scores. ranked_indices must be '
                'a permutation of the provided candidate indices. importance_scores must '
                'map every candidate index to a number between 0 and 1, where a larger '
                'value means stronger support for the query relation.'
            )
        },
        {
            'role': 'user',
            'content': (
                'Rank the candidate paths by how strongly they semantically support the '
                'relation in the query triple. Both two-hop and three-hop paths may be '
                'useful. Do not invent paths or indices. Valid indices are: '
                f'{original_indices}. If the evidence is ambiguous, preserve the original '
                'order and assign the same score, such as 0.5, to every candidate. '
                'Return JSON only.\n' + json.dumps(user_payload, ensure_ascii=False)
            )
        }
    ]

def _extract_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError('No JSON object found', text, 0)


def _resolve_llm_torch_dtype():
    dtype_name = str(getattr(params_, 'llm_path_torch_dtype', 'auto') or 'auto').lower()
    if dtype_name == 'auto':
        return 'auto'
    if dtype_name in ('float16', 'fp16', 'half'):
        return torch.float16
    if dtype_name in ('bfloat16', 'bf16'):
        return torch.bfloat16
    if dtype_name in ('float32', 'fp32'):
        return torch.float32
    return 'auto'


def _ensure_local_qwen_model():
    global _llm_tokenizer_, _llm_local_model_, _llm_local_failed_
    if _llm_tokenizer_ is not None and _llm_local_model_ is not None:
        return True
    if _llm_local_failed_:
        return False
    model_path = getattr(params_, 'llm_path_local_model', None)
    if not model_path:
        logger.warning('llm_path_local_model is not set; keep original path order.')
        _llm_local_failed_ = True
        return False
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _llm_tokenizer_ = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        kwargs = {
            'device_map': getattr(params_, 'llm_path_device_map', 'auto'),
            'trust_remote_code': True
        }
        dtype_value = _resolve_llm_torch_dtype()
        kwargs['torch_dtype'] = dtype_value
        _llm_local_model_ = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        _llm_local_model_.eval()
        return True
    except Exception as exc:
        logger.warning('Loading local Qwen model failed; keep original path order. Error: %s', exc)
        _llm_tokenizer_ = None
        _llm_local_model_ = None
        _llm_local_failed_ = True
        return False


def _call_local_qwen_path_ranker(head_global, tail_global, target_rel, candidates):
    if not _ensure_local_qwen_model():
        return None
    messages = _build_path_rank_messages(head_global, tail_global, target_rel, candidates)
    try:
        prompt = _llm_tokenizer_.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _llm_tokenizer_([prompt], return_tensors='pt')
        first_param = next(_llm_local_model_.parameters())
        inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = _llm_local_model_.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=_llm_tokenizer_.eos_token_id
            )
        generated = outputs[:, inputs['input_ids'].shape[1]:]
        content = _llm_tokenizer_.batch_decode(generated, skip_special_tokens=True)[0]
        logger.debug('Raw Qwen output: %s', content)
        parsed = _extract_json_object(content)
        return parsed
    except Exception as exc:
        logger.warning('Local Qwen path ranking failed; keep original path order. Error: %s', exc)
        return None


def _normalize_llm_rank_result(rank_result, num_candidates):
    if rank_result is None:
        return None, {}

    if isinstance(rank_result, list):
        ranked_indices = rank_result
        raw_scores = {}
    elif isinstance(rank_result, dict):
        ranked_indices = rank_result.get('ranked_indices')
        raw_scores = rank_result.get('importance_scores', {})
    else:
        return None, {}

    if not ranked_indices:
        return None, {}

    scores = {}
    if isinstance(raw_scores, dict):
        iterable = raw_scores.items()
        for idx, score in iterable:
            try:
                idx = int(idx)
                score = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_candidates:
                scores[idx] = max(0.0, min(1.0, score))
    elif isinstance(raw_scores, list):
        for pos, item in enumerate(raw_scores):
            try:
                if isinstance(item, dict):
                    idx = int(item.get('index', pos))
                    score = float(item.get('score', item.get('importance', item.get('importance_score'))))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    idx = int(item[0])
                    score = float(item[1])
                else:
                    idx = pos
                    score = float(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_candidates:
                scores[idx] = max(0.0, min(1.0, score))

    return ranked_indices, scores


def _serialize_importance_scores(importance_scores):
    return {str(int(idx)): float(score) for idx, score in importance_scores.items() if score is not None}


def _rank_result_has_importance_scores(rank_result, num_candidates):
    _, scores = _normalize_llm_rank_result(rank_result, num_candidates)
    return len(scores) == num_candidates


def _path_to_llm_item_with_score(idx, path, importance_scores):
    item = _path_to_llm_item(idx, path)
    item['importance_score'] = importance_scores.get(int(idx))
    return item


def _rank_candidate_paths_with_llm(parent_ids, target_rel, candidates):
    if not getattr(params_, 'use_llm_path_filter', False) or len(candidates) <= 1:
        return candidates
    _ensure_llm_resources()
    if not _llm_entity_text_ or not _llm_relation_text_:
        logger.warning('LLM path description files are missing or empty; keep original path order. dir=%s', getattr(params_, 'llm_path_desc_dir', None))
        return candidates
    for path in candidates:
        path['global_ent_seq'] = [int(parent_ids[int(local_id)]) for local_id in path['ent_seq']]
    head_global = int(parent_ids[0])
    tail_global = int(parent_ids[1])
    cache_key = _llm_cache_key(head_global, tail_global, target_rel, candidates)
    rank_result = _llm_path_cache_data_.get(cache_key) if _llm_path_cache_data_ is not None else None
    is_cache_hit = rank_result is not None
    if rank_result is not None and not _rank_result_has_importance_scores(rank_result, len(candidates)):
        logger.info('Cached LLM path ranking has no complete importance_scores; refresh from LLM. key=%s', cache_key)
        rank_result = None
        is_cache_hit = False
    if rank_result is None:
        rank_result = _call_local_qwen_path_ranker(head_global, tail_global, target_rel, candidates)

    ranked_indices, importance_scores = _normalize_llm_rank_result(rank_result, len(candidates))

    if ranked_indices is not None:
        cache_path = getattr(params_, 'llm_path_cache', None)
        if cache_path and not is_cache_hit:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            cache_item = {'key': cache_key, 'ranked_indices': ranked_indices}
            if importance_scores:
                cache_item['importance_scores'] = _serialize_importance_scores(importance_scores)
            with open(cache_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
            _llm_path_cache_data_[cache_key] = {
                'ranked_indices': ranked_indices,
                'importance_scores': _serialize_importance_scores(importance_scores)
            }

    if not ranked_indices:
        return candidates

    used = set()
    ranked = []
    final_order = []
    for idx in ranked_indices:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in used:
            candidates[idx]['llm_importance_score'] = importance_scores.get(idx)
            ranked.append(candidates[idx])
            used.add(idx)
            final_order.append(idx)
    for idx, path in enumerate(candidates):
        if idx not in used:
            path['llm_importance_score'] = importance_scores.get(idx)
            ranked.append(path)
            final_order.append(idx)
    logger.info('LLM path importance scores: %s', _serialize_importance_scores(importance_scores))
    return ranked
def _find_paths_between_roots_eval(subgraph, num_rels, max_paths, rel, parent_ids=None, use_llm_for_paths=False):
    """Collect one-hop to three-hop paths and optionally rank multi-hop paths with the LLM."""
    PAD_ID = num_rels
    u_list = (subgraph.ndata['id'] == 1).nonzero(as_tuple=True)[0]
    v_list = (subgraph.ndata['id'] == 2).nonzero(as_tuple=True)[0]

    if len(u_list) == 0 or len(v_list) == 0:
        return [[PAD_ID] * 3] * max_paths, [[-1] * 2] * max_paths

    u, v = u_list[0].item(), v_list[0].item()
    etype = subgraph.edata['type']
    seen_paths = set()
    direct_paths = []
    candidate_paths = []
    max_candidates = max(0, int(getattr(params_, 'llm_path_max_candidates', 50)))

    def ensure_tensor(eids):
        if isinstance(eids, torch.Tensor):
            return eids.view(-1)
        return torch.tensor([eids])

    def get_edge_rels(src, dst):
        if not subgraph.has_edges_between(src, dst):
            return []
        return [etype[eid].item() for eid in ensure_tensor(subgraph.edge_ids(src, dst))]

    def candidate_pool_full():
        return max_candidates > 0 and len(candidate_paths) >= max_candidates

    def make_path(rel_seq, ent_seq, mid_seq):
        return {
            'rel_seq': [int(x) for x in rel_seq],
            'ent_seq': [int(x) for x in ent_seq],
            'mid_seq': [int(x) for x in mid_seq]
        }

    def is_new_path(path):
        key = (tuple(path['rel_seq']), tuple(path['ent_seq']))
        if key in seen_paths:
            return False
        seen_paths.add(key)
        return True

    def add_candidate(path):
        if candidate_pool_full():
            return False
        if is_new_path(path):
            candidate_paths.append(path)
            return True
        return False

                                                  
    if subgraph.has_edges_between(u, v):
        eids = ensure_tensor(subgraph.edge_ids(u, v))
        for eid in eids:
            if eid == rel:
                continue
            path = make_path([etype[eid].item(), PAD_ID, PAD_ID], [u, v], [-1, -1])
            if is_new_path(path):
                direct_paths.append(path)

                                                              
    for n1 in subgraph.successors(u):
        if candidate_pool_full():
            break
        n1 = n1.item()
        if n1 == u or n1 == v:
            continue
        r1s, r2s = get_edge_rels(u, n1), get_edge_rels(n1, v)
        for r1 in r1s:
            for r2 in r2s:
                path = make_path([r1, r2, PAD_ID], [u, n1, v], [n1, -1])
                add_candidate(path)
                if candidate_pool_full():
                    break
            if candidate_pool_full():
                break

                                                                    
    for n1 in subgraph.successors(u):
        if candidate_pool_full():
            break
        n1 = n1.item()
        if n1 == u or n1 == v:
            continue
        for n2 in subgraph.successors(n1):
            if candidate_pool_full():
                break
            n2 = n2.item()
            if n2 == u or n2 == v or n2 == n1:
                continue
            r1s, r2s, r3s = get_edge_rels(u, n1), get_edge_rels(n1, n2), get_edge_rels(n2, v)
            for r1 in r1s:
                for r2 in r2s:
                    for r3 in r3s:
                        path = make_path([r1, r2, r3], [u, n1, n2, v], [n1, n2])
                        add_candidate(path)
                        if candidate_pool_full():
                            break
                    if candidate_pool_full():
                        break
                if candidate_pool_full():
                    break

    if use_llm_for_paths and parent_ids is not None and max_paths > len(direct_paths):
        candidate_paths = _rank_candidate_paths_with_llm(parent_ids, rel, candidate_paths)

    final_paths = direct_paths + candidate_paths
    path_relations = [path['rel_seq'] for path in final_paths[:max_paths]]
    path_entities = [path['mid_seq'] for path in final_paths[:max_paths]]

    while len(path_relations) < max_paths:
        path_relations.append([PAD_ID] * 3)
        path_entities.append([-1] * 2)

    return path_relations[:max_paths], path_entities[:max_paths]

def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value,
                  id2entity, m_h2r, m_t2r,
                  node_features=None, kge_entity2id=None, llm_path_rank_index=None):
    subgraphs = []
    r_labels = []
    nodes_num = []
    ht_nei_rels = []
    all_paths_rel = []
    bi_graphs = []

    num_rels = model_.params.num_rels
    max_p = model_.params.max_paths if hasattr(model_, 'params') else 3

    for link_idx, link in enumerate(all_links):
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling(
            (head, tail), rel, adj_list, h=params_.hop,
            enclosing_sub_graph=params_.enclosing_sub_graph,
            max_node_label_value=max_node_label_value
        )

        subgraph = dgl_adj_list.subgraph(nodes)
        parent_nid = subgraph.ndata[dgl.NID]

        subgraph.ndata['out_nei_rels'] = torch.LongTensor(m_h2r[parent_nid])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(m_t2r[parent_nid])
        subgraph.ndata['r_label'] = torch.full((subgraph.number_of_nodes(),), rel, dtype=torch.long)

        subgraph.edata['label'] = torch.full((subgraph.number_of_edges(),), rel, dtype=torch.long)

        try:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            found_rels = subgraph.edata['type'][edges_btw_roots].view(-1)
            need_add_edge = not torch.any(found_rels == rel).item()
        except:
            need_add_edge = True

        if need_add_edge:
            subgraph.add_edges(torch.tensor([0]), torch.tensor([1]))
            subgraph.edata['type'][-1] = rel
            subgraph.edata['label'][-1] = rel

        n_feats = node_features[[kge_entity2id[id2entity[n]] for n in nodes]] if kge_entity2id else None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        use_llm = llm_path_rank_index is not None and link_idx == int(llm_path_rank_index)
        path_rels, path_ents = _find_paths_between_roots_eval(
            subgraph, num_rels, max_p, rel,
            parent_ids=parent_nid.tolist(), use_llm_for_paths=use_llm
        )

        p_e_src, p_e_dst = [], []
        e_p_src, e_p_dst = [], []
        g_p_src, g_p_dst = [], []
        p_g_src, p_g_dst = [], []

        u_idx = (subgraph.ndata['id'] == 1).nonzero().item()
        v_idx = (subgraph.ndata['id'] == 2).nonzero().item()

        for p_idx in range(max_p):
            if path_rels[p_idx][0] != num_rels:
                g_p_src.append(0); g_p_dst.append(p_idx)
                p_g_src.append(p_idx); p_g_dst.append(0)

                current_path_ents = [u_idx, v_idx] + [m for m in path_ents[p_idx] if m != -1]
                for e_idx in current_path_ents:
                    p_e_src.append(p_idx); p_e_dst.append(e_idx)
                    e_p_src.append(e_idx); e_p_dst.append(p_idx)

        bi_g = dgl.heterograph({
            ('path', 'interact', 'entity'): (torch.tensor(p_e_src, dtype=torch.long),
                                            torch.tensor(p_e_dst, dtype=torch.long)),
            ('entity', 'interact', 'path'): (torch.tensor(e_p_src, dtype=torch.long),
                                            torch.tensor(e_p_dst, dtype=torch.long)),
            ('global', 'link', 'path'): (torch.tensor(g_p_src, dtype=torch.long),
                                        torch.tensor(g_p_dst, dtype=torch.long)),
            ('path', 'link', 'global'): (torch.tensor(p_g_src, dtype=torch.long),
                                        torch.tensor(p_g_dst, dtype=torch.long))
        }, num_nodes_dict={'entity': subgraph.number_of_nodes(), 'path': max_p, 'global': 1})

        subgraphs.append(subgraph)
        bi_graphs.append(bi_g)
        all_paths_rel.append(path_rels)
        r_labels.append(rel)
        nodes_num.append(subgraph.number_of_nodes())
        ht_nei_rels.append((subgraph.ndata['out_nei_rels'][u_idx].tolist(), subgraph.ndata['out_nei_rels'][v_idx].tolist()))

    return (dgl.batch(subgraphs), dgl.batch(bi_graphs),
            torch.LongTensor(r_labels), torch.LongTensor(all_paths_rel)), np.array(nodes_num), ht_nei_rels, None


def get_rank(neg_links):
    device = next(model_.parameters()).device

    head_neg_links = neg_links['head'][0]
    head_target_id = int(neg_links['head'][1])
    if head_target_id != 10000:
        data, head_nodes_num, head_nei_rels, _ = get_subgraphs(
            head_neg_links, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
            llm_path_rank_index=head_target_id
        )
        g, bi_g, r_l, p = data
        data = (g.to(device), bi_g.to(device), r_l.to(device), p.to(device))
        head_scores = model_(data).view(-1).cpu().detach().numpy()
        head_rank = int(np.argwhere(np.argsort(head_scores)[::-1] == head_target_id)[0][0] + 1)
    else:
        head_scores = np.array([])
        head_rank = 10000
        head_nodes_num = np.array([])
        head_nei_rels = []

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = int(neg_links['tail'][1])
    if tail_target_id != 10000:
        data, tail_nodes_num, tail_nei_rels, _ = get_subgraphs(
            tail_neg_links, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
            llm_path_rank_index=tail_target_id
        )
        g, bi_g, r_l, p = data
        data = (g.to(device), bi_g.to(device), r_l.to(device), p.to(device))
        tail_scores = model_(data).view(-1).cpu().detach().numpy()
        tail_rank = int(np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id)[0][0] + 1)
    else:
        tail_scores = np.array([])
        tail_rank = 10000
        tail_nodes_num = np.array([])
        tail_nei_rels = []

    return head_scores, head_rank, tail_scores, tail_rank, head_nodes_num, tail_nodes_num, head_nei_rels, tail_nei_rels

def _prepare_model_inputs_for_benchmark(neg_links):
    device = next(model_.parameters()).device

    head_data = None
    head_neg_links = neg_links['head'][0]
    head_target_id = int(neg_links['head'][1])
    if head_target_id != 10000:
        data, _, _, _ = get_subgraphs(
            head_neg_links, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
        )
        g, bi_g, r_l, p = data
        head_data = (g.to(device), bi_g.to(device), r_l.to(device), p.to(device))

    tail_data = None
    tail_neg_links = neg_links['tail'][0]
    tail_target_id = int(neg_links['tail'][1])
    if tail_target_id != 10000:
        data, _, _, _ = get_subgraphs(
            tail_neg_links, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
        )
        g, bi_g, r_l, p = data
        tail_data = (g.to(device), bi_g.to(device), r_l.to(device), p.to(device))

    return head_data, tail_data


def _benchmark_single_triple_forward(triple_links):
    sample_count = min(max(1, int(getattr(params_, 'benchmark_num_samples', 30))), len(triple_links))
    bench_links = triple_links[:sample_count]
    device = next(model_.parameters()).device
    prepared_inputs = []

    logger.info('Benchmark preparing single-triple model inputs: %d triples', sample_count)
    for link in tqdm(bench_links, total=len(bench_links), desc='prepare-single-triple-forward-data'):
        one_link = np.array([link])
        data, _, _, _ = get_subgraphs(
            one_link, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
        )
        g, bi_g, r_l, p = data
        prepared_inputs.append((g.to(device), bi_g.to(device), r_l.to(device), p.to(device)))
    warmup = min(max(0, int(getattr(params_, 'benchmark_warmup', 3))), max(0, sample_count - 1))
    model_.eval()
    if warmup > 0:
        logger.info('Benchmark warmup single triples: %d', warmup)
        with torch.no_grad():
            for i in range(warmup):
                data = prepared_inputs[i]
                _ = model_(data)
        _sync_for_timing()

    times_ms = []
    timed_inputs = prepared_inputs[warmup:]
    logger.info('Benchmark single-triple forward timing: %d triples', len(timed_inputs))
    with torch.no_grad():
        for data in tqdm(timed_inputs, total=len(timed_inputs), desc='single-triple-forward-benchmark'):
            _sync_for_timing()
            t0 = time.perf_counter()
            _ = model_(data)
            _sync_for_timing()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms, dtype=np.float64)
    stats = {
        'benchmark': True,
        'num_samples': int(len(times_arr)),
        'avg_ms': float(times_arr.mean()) if len(times_arr) > 0 else 0.0,
        'std_ms': float(times_arr.std(ddof=1)) if len(times_arr) > 1 else 0.0,
        'min_ms': float(times_arr.min()) if len(times_arr) > 0 else 0.0,
        'max_ms': float(times_arr.max()) if len(times_arr) > 0 else 0.0,
    }
    logger.info(
        'Single-triple forward result: triples=%d | avg=%.3f ms | std=%.3f ms | min=%.3f ms | max=%.3f ms',
        stats['num_samples'], stats['avg_ms'], stats['std_ms'], stats['min_ms'], stats['max_ms']
    )
    return stats

def _benchmark_subgraph_extraction(triple_links, run_idx=0):
    sample_count = min(max(1, int(getattr(params_, 'benchmark_subgraph_num_samples', 30))), len(triple_links))
    bench_links = triple_links[:sample_count]
    times_ms = []
    graph_nodes = []
    graph_edges = []
    path_nodes = []
    path_edges = []

    logger.info(
        'Run %d/%d benchmark single-triple subgraph extraction: %d triples',
        run_idx + 1, params_.num_eval_runs, sample_count
    )
    for link in tqdm(bench_links, total=len(bench_links), desc='single-triple-subgraph-extraction'):
        one_link = np.array([link])
        t0 = time.perf_counter()
        data, nodes_num, _, _ = get_subgraphs(
            one_link, adj_list_, dgl_adj_list_, model_.params.max_label_value,
            id2entity_, m_h2r_, m_t2r_, node_features_, kge_entity2id_,
        )
        t1 = time.perf_counter()

        g, bi_g, _, _ = data
        times_ms.append((t1 - t0) * 1000.0)
        graph_nodes.append(int(nodes_num[0]))
        graph_edges.append(g.num_edges())
        path_nodes.append(bi_g.num_nodes())
        path_edges.append(bi_g.num_edges())

    times_arr = np.array(times_ms, dtype=np.float64)
    stats = {
        'benchmark_subgraph': True,
        'num_samples': int(len(times_arr)),
        'avg_ms': float(times_arr.mean()) if len(times_arr) > 0 else 0.0,
        'std_ms': float(times_arr.std(ddof=1)) if len(times_arr) > 1 else 0.0,
        'min_ms': float(times_arr.min()) if len(times_arr) > 0 else 0.0,
        'max_ms': float(times_arr.max()) if len(times_arr) > 0 else 0.0,
        'avg_graph_nodes': float(np.mean(graph_nodes)) if graph_nodes else 0.0,
        'avg_graph_edges': float(np.mean(graph_edges)) if graph_edges else 0.0,
        'avg_path_nodes': float(np.mean(path_nodes)) if path_nodes else 0.0,
        'avg_path_edges': float(np.mean(path_edges)) if path_edges else 0.0,
    }
    logger.info(
        'Run %d/%d single-triple subgraph extraction result: triples=%d | avg=%.3f ms | std=%.3f ms | min=%.3f ms | max=%.3f ms | avg_graph_nodes=%.2f | avg_graph_edges=%.2f | avg_path_nodes=%.2f | avg_path_edges=%.2f',
        run_idx + 1, params_.num_eval_runs,
        stats['num_samples'], stats['avg_ms'], stats['std_ms'], stats['min_ms'], stats['max_ms'],
        stats['avg_graph_nodes'], stats['avg_graph_edges'], stats['avg_path_nodes'], stats['avg_path_edges']
    )
    return stats
def save_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, all_h_nodes_num, all_t_nodes_num, all_h_nei_rels, all_t_nei_rels):

    with open(os.path.join('./data', params.dataset, 'grail_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score, nodes_num, nei_rels in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)], all_h_nodes_num[50 * i:50 * (i + 1)], all_h_nei_rels[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score), str(nodes_num)] + [' head nei rels: '] + [str(rel) for rel in nei_rels[0]] + [' tail nei rels: '] + [str(rel) for rel in nei_rels[1]]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score, nodes_num, nei_rels in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)], all_t_nodes_num[50 * i:50 * (i + 1)], all_t_nei_rels[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score), str(nodes_num)] + [' head nei rels: '] + [str(rel) for rel in nei_rels[0]] + [' tail nei rels: '] + [str(rel) for rel in nei_rels[1]]) + '\n')


def save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation):

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_head_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(neg_triplet['head'][0], all_head_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(head_score)]) + '\n')

    with open(os.path.join('./data', params.dataset, 'grail_ruleN_ranking_tail_predictions.txt'), "w") as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(neg_triplet['tail'][0], all_tail_scores[50 * i:50 * (i + 1)]):
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o], str(tail_score)]) + '\n')


def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def _safe_divide(a, b):
    return a / b if b > 0 else 0.0


def _compute_metrics_from_ranks(ranks):
    is_hit1 = [x for x in ranks if x <= 1]
    is_hit5 = [x for x in ranks if x <= 5]
    is_hit10 = [x for x in ranks if x <= 10]
    return {
        'mrr': float(np.mean(1 / np.array(ranks))) if ranks else 0.0,
        'hits_1': _safe_divide(len(is_hit1), len(ranks)),
        'hits_5': _safe_divide(len(is_hit5), len(ranks)),
        'hits_10': _safe_divide(len(is_hit10), len(ranks)),
    }


def _sync_for_timing():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_once(params, run_idx=0):
    s_t = time.time()
    model = torch.load(params.model_path, map_location='cpu', weights_only=False)

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, m_h2r, m_t2r = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)
    node_features, kge_entity2id = get_kge_embeddings(params.dataset, params.kge_model) if params.use_kge_embeddings else (None, None)

    intialize_worker(model, adj_list, dgl_adj_list, id2entity, id2relation, params, node_features, kge_entity2id, m_h2r, m_t2r)
    if params.benchmark_subgraph_extraction:
        stats = _benchmark_subgraph_extraction(triplets['links'], run_idx=run_idx)
        logger.info(f'Run {run_idx + 1}/{params.num_eval_runs} Time used: {time.time() - s_t}')
        return stats

    if params.benchmark_inference:
        stats = _benchmark_single_triple_forward(triplets['links'])
        logger.info(f'Run {run_idx + 1}/{params.num_eval_runs} Time used: {time.time() - s_t}')
        return stats

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
        save_to_file(neg_triplets, id2entity, id2relation)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)
    elif params.mode == 'ruleN':
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(params.ruleN_pred_path, entity2id, relation2id)
    else:
        raise ValueError(f'Unknown mode: {params.mode}')

    ranks = []
    all_head_scores = []
    all_tail_scores = []
    nodes_num = []
    all_h_nei_rels = []
    all_t_nei_rels = []
    all_h_nodes_num = []
    all_t_nodes_num = []

    with torch.no_grad():
        for link in tqdm(neg_triplets, total=len(neg_triplets)):
            head_scores, head_rank, tail_scores, tail_rank, h_nodes_num, t_nodes_num, h_nei_rels, t_nei_rels = get_rank(link)

            ranks.append(head_rank)
            ranks.append(tail_rank)

            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

            nodes_num.append(h_nodes_num.tolist()[0])
            nodes_num.append(t_nodes_num.tolist()[0])

            all_h_nodes_num += h_nodes_num.tolist()
            all_t_nodes_num += t_nodes_num.tolist()

            all_h_nei_rels += h_nei_rels
            all_t_nei_rels += t_nei_rels

    if params.mode == 'ruleN':
        save_score_to_file_from_ruleN(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation)
    else:
        save_score_to_file(neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, all_h_nodes_num, all_t_nodes_num, all_h_nei_rels, all_t_nei_rels)

    metrics_dict = _compute_metrics_from_ranks(ranks)

    logger.info(
        f'Run {run_idx + 1}/{params.num_eval_runs} Total: '
        f"MRR | Hits@1 | Hits@5 | Hits@10 : {metrics_dict['mrr']} | {metrics_dict['hits_1']} | {metrics_dict['hits_5']} | {metrics_dict['hits_10']}"
    )
    logger.info(f'Run {run_idx + 1}/{params.num_eval_runs} Time used: {time.time() - s_t}')

    result_npy = os.path.join(
        './experiments',
        params.experiment_name,
        'result-' + time.strftime('%Y%m%d%H', time.localtime(time.time())) + f'-run{run_idx + 1:02d}'
    )
    result = {'ranks': ranks, 'nodes_num': nodes_num, 'head_scores': all_head_scores, 'tail_scores': all_tail_scores}
    np.save(result_npy, result)

    return metrics_dict


def main(params):
    run_metrics = []
    base_seed = int(getattr(params, 'eval_seed', 2))

    for run_idx in range(params.num_eval_runs):
        curr_seed = base_seed + run_idx
        random.seed(curr_seed)
        np.random.seed(curr_seed)
        torch.manual_seed(curr_seed)

        logger.info(f'========== Eval run {run_idx + 1}/{params.num_eval_runs} | seed={curr_seed} ==========')
        metrics_dict = evaluate_once(params, run_idx=run_idx)
        run_metrics.append(metrics_dict)

    if params.benchmark_inference or params.benchmark_subgraph_extraction:
        avg_vals = np.array([m['avg_ms'] for m in run_metrics], dtype=np.float64)
        std_vals = np.array([m['std_ms'] for m in run_metrics], dtype=np.float64)
        n_vals = np.array([m['num_samples'] for m in run_metrics], dtype=np.int64)

        ddof = 1 if params.num_eval_runs > 1 else 0
        logger.info('========== Final Benchmark (Mean +- Std over %d runs) ==========', params.num_eval_runs)
        logger.info('Samples per run: %s', n_vals.tolist())
        logger.info('Per-sample latency mean: %.3f +- %.3f ms', float(avg_vals.mean()), float(avg_vals.std(ddof=ddof)))
        logger.info('Per-run latency std mean: %.3f ms', float(std_vals.mean()))
        return

    mrr_vals = np.array([m['mrr'] for m in run_metrics], dtype=np.float64)
    h1_vals = np.array([m['hits_1'] for m in run_metrics], dtype=np.float64)
    h5_vals = np.array([m['hits_5'] for m in run_metrics], dtype=np.float64)
    h10_vals = np.array([m['hits_10'] for m in run_metrics], dtype=np.float64)

    ddof = 1 if params.num_eval_runs > 1 else 0
    logger.info('========== Final (Mean +- Std over %d runs) ==========', params.num_eval_runs)
    logger.info('MRR: %.6f +- %.6f', float(mrr_vals.mean()), float(mrr_vals.std(ddof=ddof)))
    logger.info('Hits@1: %.6f +- %.6f', float(h1_vals.mean()), float(h1_vals.std(ddof=ddof)))
    logger.info('Hits@5: %.6f +- %.6f', float(h5_vals.mean()), float(h5_vals.std(ddof=ddof)))
    logger.info('Hits@10: %.6f +- %.6f', float(h10_vals.mean()), float(h10_vals.std(ddof=ddof)))


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=False,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument("--max_paths", type=int, default=3,
                        help="max_paths")
    parser.add_argument("--num_eval_runs", type=int, default=4,
                        help='number of repeated evaluation runs for mean/std')
    parser.add_argument("--eval_seed", type=int, default=12345,
                        help='base random seed for repeated evaluation')
    parser.add_argument("--benchmark_inference", action='store_true',
                        help='benchmark per-sample inference latency on a subset of samples')
    parser.add_argument("--benchmark_num_samples", type=int, default=30,
                        help='number of single triples used for inference latency benchmark')
    parser.add_argument("--benchmark_warmup", type=int, default=3,
                        help='number of warmup samples before timing')
    parser.add_argument("--benchmark_subgraph_extraction", action='store_true',
                        help='benchmark single-triple subgraph extraction time')
    parser.add_argument("--benchmark_subgraph_num_samples", type=int, default=30,
                        help='number of single triples used for subgraph extraction benchmark')
    parser.add_argument('--use_llm_path_filter', action='store_true',
                        help='rank multi-hop paths with a local language model')
    parser.add_argument('--no_llm_path_filter', action='store_true',
                        help='disable language-model path ranking')
    parser.add_argument('--llm_path_desc_dir', type=str, default=None,
                        help='directory containing entity2text.txt and relation2text.txt')
    parser.add_argument('--llm_path_cache', type=str, default=None,
                        help='JSONL cache for path-ranking results')
    parser.add_argument('--llm_path_local_model', type=str, default=None,
                        help='local Hugging Face model directory')
    parser.add_argument('--llm_path_device_map', type=str, default='auto')
    parser.add_argument('--llm_path_torch_dtype', type=str, default='auto',
                        choices=['auto', 'float16', 'fp16', 'bfloat16', 'bf16', 'float32', 'fp32'])
    parser.add_argument('--llm_path_max_candidates', type=int, default=50)
    params = parser.parse_args()

    if params.no_llm_path_filter:
        params.use_llm_path_filter = False

    params.device = 'cpu'

    params.file_paths = {
        'graph': os.path.join('./data', params.dataset, 'train.txt'),
        'links': os.path.join('./data', params.dataset, 'test.txt')
    }

    params.ruleN_pred_path = os.path.join('./data', params.dataset, 'pos_predictions.txt')
    params.model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)


