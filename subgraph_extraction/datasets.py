from torch.utils.data import Dataset
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
import hashlib
import urllib.error
import urllib.request
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file
from .graph_sampler import *


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None, is_ent2rels=None):

    testing = 'test' in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation, h2r, m_h2r, t2r, m_t2r = process_files(params.file_paths, saved_relation2id, sort_data=params.sort_data)

                                                                                                   

    data_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        '''
        adj_list,
        edges,
        num_neg_samples_per_link=1,
        max_size=1000000,
        constrained_neg_prob=0
        '''
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}
                                 
                                                                                                                        
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=params.constrained_neg_prob)               

    if testing:
        directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)
    

def get_kge_embeddings(dataset, kge_model):

    path = './experiments/kge_baselines/{}_{}'.format(kge_model, dataset)
    node_features = np.load(os.path.join(path, 'entity_embedding.npy'))
    with open(os.path.join(path, 'id2entity.json')) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths,max_paths, included_relations=None, add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='', kge_model='', file_name='', is_ret_nodes_num=False, use_llm_path_filter=False, llm_path_desc_dir=None, llm_path_cache=None, llm_path_model='gpt-4o-mini', llm_path_backend='openai', llm_path_local_model=None, llm_path_device_map='auto', llm_path_torch_dtype='auto', llm_path_max_candidates=50):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=20, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.db_name_pos = db_name_pos
        if self.db_name_pos == 'train_pos':
            db_name_pos_contrast1 = 'train_pos_contrast1' 
            self.db_pos_contrast1 = self.main_env.open_db(db_name_pos_contrast1.encode())
            db_name_pos_contrast2 = 'train_pos_contrast2'
            self.db_pos_contrast2 = self.main_env.open_db(db_name_pos_contrast2.encode())
        self.node_features, self.kge_entity2id = get_kge_embeddings(dataset, kge_model) if use_kge_embeddings else (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.is_ret_nodes_num = is_ret_nodes_num
        self.max_paths = max_paths
        self.dataset = dataset
        self.use_llm_path_filter = use_llm_path_filter
        self.llm_path_model = llm_path_model
        self.llm_path_backend = llm_path_backend
        self.llm_path_local_model = llm_path_local_model
        self.llm_path_device_map = llm_path_device_map
        self.llm_path_torch_dtype = llm_path_torch_dtype
        self.llm_path_max_candidates = max(0, int(llm_path_max_candidates))
        self._llm_tokenizer = None
        self._llm_local_model = None
        self.llm_path_desc_dir = llm_path_desc_dir or self._default_llm_desc_dir(dataset)
        self.llm_path_cache = llm_path_cache
        self._llm_entity_text = None
        self._llm_relation_text = None
        self._llm_path_cache_data = None
        ssp_graph, __, __, __, id2entity, id2relation, h2r, m_h2r, t2r, m_t2r = process_files(raw_data_paths, included_relations, add_traspose_rels)
        self.num_rels = len(ssp_graph)

                                                                        
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

                                                                                                             
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation
        self.m_h2r = m_h2r
        self.m_t2r = m_t2r

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')                                
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

                                               
                                                                                                                                                                                                 

                                               
                                                                                                                                                                                        

                                               
                                                                                                                                                                                                                 

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        if self.db_name_pos == 'train_pos':
                                                                       
                                                                                                                       
                         
            db_name_pos_contrast1 = 'train_pos_contrast1'
                                                                                          
            with self.main_env.begin(db=self.db_pos_contrast1) as txn:
                self.num_graphs_pos_contrast1 = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        
                        
            db_name_pos_contrast2 = 'train_pos_contrast2'
                                                                                          
            with self.main_env.begin(db=self.db_pos_contrast2) as txn:
                self.num_graphs_pos_contrast2 = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
                                                                                                                                                             
                                                                                                              
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos, use_llm_for_paths=True)
            paths_pos = subgraph_pos.extracted_paths         
                                                                 
                                                                                       
            nei_rels_pos = [[0, 1], [0, 1]]
                                             
                                                                       
                                                                
                                                                                                               
                                                                                                  
                                                            
                                                                                 
                                     
                                                                             
                                                   
                                                  
        if self.db_name_pos == 'train_pos':
                       
            with self.main_env.begin(db=self.db_pos_contrast1) as txn:
                str_id = '{:08}'.format(index).encode('ascii')
                nodes_cont1, r_label_cont1, g_label_cont1, n_labels_cont1 = deserialize(txn.get(str_id)).values()
                subgraph_cont1 = self._prepare_subgraphs(nodes_cont1, r_label_cont1, n_labels_cont1, use_llm_for_paths=False)
                paths_cont1 = subgraph_cont1.extracted_paths
                                 
                g_label_cont1 = g_label_pos
                r_label_cont1 = r_label_pos
        
                       
            with self.main_env.begin(db=self.db_pos_contrast2) as txn:
                str_id = '{:08}'.format(index).encode('ascii')
                nodes_cont2, r_label_cont2, g_label_cont2, n_labels_cont2 = deserialize(txn.get(str_id)).values()
                subgraph_cont2 = self._prepare_subgraphs(nodes_cont2, r_label_cont2, n_labels_cont2, use_llm_for_paths=False)
                paths_cont2 = subgraph_cont2.extracted_paths
                g_label_cont2 = g_label_pos
                r_label_cont2 = r_label_pos
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        nei_rels_negs = []
        paths_negs = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                                                                                                     
                                                            
                sub_neg = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg, use_llm_for_paths=False)
                subgraphs_neg.append(sub_neg)
                paths_negs.append(sub_neg.extracted_paths)        
                                                                     
                                                                                           
                nei_rels_neg = [[0, 1], [0, 1]]
                nei_rels_negs.append(nei_rels_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)
        
                                                                  
                                                                                                
                              
                               
        if self.db_name_pos == 'train_pos':
                                                                        
                                                                        
                                                                        
            return (subgraph_pos, g_label_pos, r_label_pos, paths_pos,
            subgraphs_neg, g_labels_neg, r_labels_neg, paths_negs,
            subgraph_cont1, g_label_cont1, r_label_cont1, paths_cont1,
            subgraph_cont2, g_label_cont2, r_label_cont2, paths_cont2)
        else:
            return subgraph_pos, g_label_pos, r_label_pos, paths_pos,\
                   subgraphs_neg, g_labels_neg, r_labels_neg, paths_negs

    def __len__(self):
        return self.num_graphs_pos

                                                                                     
                                                                                                                       
                                                                                                                                    
                                                                                                                   
                                                                                                              
                                                           
                                                                                                                                            
                     
                                                       
                                                                                                                                                                                                                                                                                         
                                                  
                                                                                                                                                                                                                                                                                                                                                                                           
                                       
                                          
                                                                                       
                                                                                        

                                                                                          
                                                                                                            
                                                                                             
                                                                                     

                                                          
                                                                                 
                                       
                                                                                                    
                                                                                                   

                         
    
                                                                                   
             
                  
                                
                            
                          
             
                                         
                                         
    
                                
                                                             
                                   
                                          
                                          
            
                                                   
                                                   
                                      
    
                   
                       
                                                                                           
                                                                                           
                                                                                            
                                                                                           
           
        
                              
                            
                                
                                     
                         
           
        
                                                                        
                   
                                                                                   
             
                              
                                
                                
             
                         
                                                          
                                                          
                                                          
                                                          
    
                                                             
                                                       
                                   
                                               
                                          
                                          
                
                                          
                                          
            
                                           
                                           
                                  
                                      
            
                                      
                                  
    
                              
                       
                   
                                                                                           
                                                                                           
                                                                                            
                                                                                            
            
                          
                                                                                          
                                                                                          
                                                                                           
                                                                                          
           
        
                        
                            
                                
                                     
                         
           
        
                                                                        
                   
    def _default_llm_desc_dir(self, dataset):
        if not dataset:
            return None
        base_dir = os.path.dirname(__file__)
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

    def _load_text_map(self, file_path):
        mapping = {}
        if not file_path or not os.path.isfile(file_path):
            return mapping
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.rstrip('\n').split('\t', 1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
        return mapping

    def _ensure_llm_resources(self):
        if self._llm_entity_text is not None and self._llm_relation_text is not None:
            return

        self._llm_entity_text = {}
        self._llm_relation_text = {}
        if self.llm_path_desc_dir:
            self._llm_entity_text = self._load_text_map(os.path.join(self.llm_path_desc_dir, 'entity2text.txt'))
            self._llm_relation_text = self._load_text_map(os.path.join(self.llm_path_desc_dir, 'relation2text.txt'))

        self._llm_path_cache_data = {}
        if self.llm_path_cache and os.path.isfile(self.llm_path_cache):
            with open(self.llm_path_cache, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'key' in item and 'ranked_indices' in item:
                        if 'importance_scores' in item:
                            self._llm_path_cache_data[item['key']] = {
                                'ranked_indices': item['ranked_indices'],
                                'importance_scores': item.get('importance_scores', {})
                            }
                        else:
                            self._llm_path_cache_data[item['key']] = item['ranked_indices']

    def _entity_text(self, global_id):
        name = self.id2entity.get(int(global_id), str(global_id))
        desc = self._llm_entity_text.get(name, '') if self._llm_entity_text is not None else ''
        return f"{name} ({desc})" if desc and desc != name else name

    def _relation_text(self, rel_id):
        rel_id = int(rel_id)
        name = self.id2relation.get(rel_id, str(rel_id))
        desc = self._llm_relation_text.get(name, '') if self._llm_relation_text is not None else ''
        return f"{name} ({desc})" if desc and desc != name else name

    def _path_to_llm_item(self, idx, path):
        parent_ids = path['global_ent_seq']
        rel_seq = [int(r) for r in path['rel_seq'] if int(r) != self.num_rels]
        steps = []
        for step_idx, rel_id in enumerate(rel_seq):
            src = self._entity_text(parent_ids[step_idx]) if step_idx < len(parent_ids) else ''
            dst = self._entity_text(parent_ids[step_idx + 1]) if step_idx + 1 < len(parent_ids) else ''
            steps.append(f"{src} -- {self._relation_text(rel_id)} --> {dst}")
        return {
            'index': idx,
            'length': len(rel_seq),
            'path': ' ; '.join(steps)
        }

    def _llm_cache_key(self, head_global, tail_global, target_rel, candidates):
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

    def _build_path_rank_messages(self, head_global, tail_global, target_rel, candidates):
        path_items = [self._path_to_llm_item(i, path) for i, path in enumerate(candidates)]
        original_indices = [item['index'] for item in path_items]
        user_payload = {
            'query': {
                'head': self._entity_text(head_global),
                'relation': self._relation_text(target_rel),
                'tail': self._entity_text(tail_global)
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
    def _extract_json_object(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise json.JSONDecodeError('No JSON object found', text, 0)

    def _call_openai_path_ranker(self, head_global, tail_global, target_rel, candidates):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            logging.warning('OPENAI_API_KEY is not set; keep original path order.')
            return None

        request_body = {
            'model': self.llm_path_model,
            'temperature': 0,
            'response_format': {'type': 'json_object'},
            'messages': self._build_path_rank_messages(head_global, tail_global, target_rel, candidates)
        }
        req = urllib.request.Request(
            'https://api.openai.com/v1/chat/completions',
            data=json.dumps(request_body).encode('utf-8'),
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            content = data['choices'][0]['message']['content']
            parsed = self._extract_json_object(content)
            return parsed
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            logging.warning('OpenAI path ranking failed; keep original path order. Error: %s', exc)
            return None

    def _resolve_torch_dtype(self):
        import torch
        dtype_name = str(self.llm_path_torch_dtype or 'auto').lower()
        if dtype_name == 'auto':
            return 'auto'
        if dtype_name in ('float16', 'fp16', 'half'):
            return torch.float16
        if dtype_name in ('bfloat16', 'bf16'):
            return torch.bfloat16
        if dtype_name in ('float32', 'fp32'):
            return torch.float32
        logging.warning('Unknown llm_path_torch_dtype=%s; use auto.', self.llm_path_torch_dtype)
        return 'auto'

    def _ensure_local_qwen_model(self):
        if self._llm_tokenizer is not None and self._llm_local_model is not None:
            return True
        if not self.llm_path_local_model:
            logging.warning('llm_path_local_model is not set; keep original path order.')
            return False
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                self.llm_path_local_model,
                trust_remote_code=True
            )
            self._llm_local_model = AutoModelForCausalLM.from_pretrained(
                self.llm_path_local_model,
                torch_dtype=self._resolve_torch_dtype(),
                device_map=self.llm_path_device_map,
                trust_remote_code=True
            )
            self._llm_local_model.eval()
            return True
        except Exception as exc:
            logging.warning('Loading local Qwen model failed; keep original path order. Error: %s', exc)
            self._llm_tokenizer = None
            self._llm_local_model = None
            return False

    def _call_local_qwen_path_ranker(self, head_global, tail_global, target_rel, candidates):
        if not self._ensure_local_qwen_model():
            return None
        import torch
        messages = self._build_path_rank_messages(head_global, tail_global, target_rel, candidates)
        try:
            prompt = self._llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self._llm_tokenizer([prompt], return_tensors='pt')
            first_param = next(self._llm_local_model.parameters())
            inputs = {k: v.to(first_param.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._llm_local_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self._llm_tokenizer.eos_token_id
                )
            generated = outputs[:, inputs['input_ids'].shape[1]:]
            content = self._llm_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            logging.warning('Raw Qwen output: %s', content)
            parsed = self._extract_json_object(content)
            return parsed
        except Exception as exc:
            logging.warning('Local Qwen path ranking failed; keep original path order. Error: %s', exc)
            return None

    def _call_llm_path_ranker(self, head_global, tail_global, target_rel, candidates):
        backend = str(self.llm_path_backend or 'openai').lower()
        if backend in ('local_qwen', 'qwen', 'transformers'):
            return self._call_local_qwen_path_ranker(head_global, tail_global, target_rel, candidates)
        if backend == 'openai':
            return self._call_openai_path_ranker(head_global, tail_global, target_rel, candidates)
        logging.warning('Unknown llm_path_backend=%s; keep original path order.', self.llm_path_backend)
        return None
    def _normalize_llm_rank_result(self, rank_result, num_candidates):
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
            for idx, score in raw_scores.items():
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

    def _serialize_importance_scores(self, importance_scores):
        return {str(int(idx)): float(score) for idx, score in importance_scores.items() if score is not None}

    def _rank_candidate_paths_with_llm(self, subgraph, target_rel, candidates):
        if not self.use_llm_path_filter or len(candidates) <= 1:
            return candidates
        self._ensure_llm_resources()
        if not self._llm_entity_text or not self._llm_relation_text:
            logging.warning('LLM path description files are missing or empty; keep original path order. dir=%s', self.llm_path_desc_dir)
            return candidates

        parent_ids = subgraph.ndata['parent_id'].long().tolist()
        head_global = parent_ids[0]
        tail_global = parent_ids[1]
        for path in candidates:
            path['global_ent_seq'] = [parent_ids[int(local_id)] for local_id in path['ent_seq']]

        cache_key = self._llm_cache_key(head_global, tail_global, target_rel, candidates)
        rank_result = self._llm_path_cache_data.get(cache_key) if self._llm_path_cache_data is not None else None
        is_cache_hit = rank_result is not None
        if rank_result is None:
            rank_result = self._call_llm_path_ranker(head_global, tail_global, target_rel, candidates)

        ranked_indices, importance_scores = self._normalize_llm_rank_result(rank_result, len(candidates))
        if ranked_indices is not None and self.llm_path_cache and not is_cache_hit:
            cache_dir = os.path.dirname(self.llm_path_cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            cache_item = {'key': cache_key, 'ranked_indices': ranked_indices}
            if importance_scores:
                cache_item['importance_scores'] = self._serialize_importance_scores(importance_scores)
            with open(self.llm_path_cache, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
            self._llm_path_cache_data[cache_key] = {
                'ranked_indices': ranked_indices,
                'importance_scores': self._serialize_importance_scores(importance_scores)
            }

        if not ranked_indices:
            return candidates

        used = set()
        ranked = []
        for idx in ranked_indices:
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(candidates) and idx not in used:
                candidates[idx]['llm_importance_score'] = importance_scores.get(idx)
                ranked.append(candidates[idx])
                used.add(idx)
        for idx, path in enumerate(candidates):
            if idx not in used:
                path['llm_importance_score'] = importance_scores.get(idx)
                ranked.append(path)
        logging.info('LLM path importance scores: %s', self._serialize_importance_scores(importance_scores))
        return ranked
    def _build_bipartite_graph(self, num_entities, paths_ent_indices, max_paths):
                       
                     
        p_e_src, p_e_dst = [], []                  
        e_p_src, e_p_dst = [], []                  
                         
        g_p_src, g_p_dst = [], []                  
        p_g_src, p_g_dst = [], []                  
    
        for p_idx, ent_set in enumerate(paths_ent_indices):
                             
            if p_idx >= max_paths:
                break
            for e_idx in ent_set:
                e_idx = int(e_idx)
                if e_idx < 0 or e_idx >= num_entities:
                    continue
                                  
                p_e_src.append(p_idx); p_e_dst.append(e_idx)
                e_p_src.append(e_idx); e_p_dst.append(p_idx)
            
                                               
            g_p_src.append(0); g_p_dst.append(p_idx)
            p_g_src.append(p_idx); p_g_dst.append(0)
    
        def to_id_tensor(vals):
                                               
            return torch.tensor(vals, dtype=torch.int64)

                     
                                   
        data_dict = {
            ('path', 'interact', 'entity'): (to_id_tensor(p_e_src), to_id_tensor(p_e_dst)),
            ('entity', 'interact', 'path'): (to_id_tensor(e_p_src), to_id_tensor(e_p_dst)),
            ('global', 'link', 'path'): (to_id_tensor(g_p_src), to_id_tensor(g_p_dst)),
            ('path', 'link', 'global'): (to_id_tensor(p_g_src), to_id_tensor(p_g_dst))
        }
        
        num_nodes_dict = {'path': max_paths, 'entity': num_entities, 'global': 1}
        return dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    def _prepare_subgraphs(self, nodes, r_label, n_labels, use_llm_for_paths=True):
                                   
        subgraph: dgl.DGLGraph = self.graph.subgraph(nodes)
        subgraph.edata['type'] = self.graph.edata['type'][subgraph.edata[dgl.EID]]
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        
                    
        has_rel = subgraph.has_edges_between(0, 1)
        if has_rel:
            edges_btw_roots = subgraph.edge_ids(0, 1)
            found_rels = subgraph.edata['type'][edges_btw_roots].view(-1)
            rel_link = torch.any(found_rels == r_label).item()
        else:
            rel_link = False
    
        if not has_rel or not rel_link:
            subgraph.add_edges([0], [1])
            subgraph.edata['type'][-1] = torch.tensor(r_label, dtype=torch.long)
            subgraph.edata['label'][-1] = torch.tensor(r_label, dtype=torch.long)
    
                          
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features_new(subgraph, n_labels, r_label, n_feats)
    
                       
        subgraph.ndata['parent_id'] = self.graph.subgraph(nodes).ndata[dgl.NID]
        subgraph.ndata['out_nei_rels'] = torch.LongTensor(self.m_h2r[subgraph.ndata['parent_id']])
        subgraph.ndata['in_nei_rels'] = torch.LongTensor(self.m_t2r[subgraph.ndata['parent_id']])
    
                                              
                                                                   
        path_data = self._find_paths_with_details(subgraph, self.max_paths, r_label, use_llm_for_paths=use_llm_for_paths)
        
                                         
        subgraph.extracted_paths = torch.LongTensor(path_data['rel_seqs'])
        
                                  
                                                     
        subgraph.bi_graph = self._build_bipartite_graph(
            subgraph.number_of_nodes(), 
            path_data['ent_indices'], 
            self.max_paths
        )
    
        return subgraph
    def _find_paths_with_details(self, subgraph, max_paths, target_rel, use_llm_for_paths=True):
        'Helper function.'
        PAD_REL = self.num_rels
        u_list = (subgraph.ndata['id'] == 1).nonzero(as_tuple=True)[0]
        v_list = (subgraph.ndata['id'] == 2).nonzero(as_tuple=True)[0]

        res = {
            'rel_seqs': [],
            'ent_indices': []
        }
        seen_paths = set()

        if len(u_list) == 0 or len(v_list) == 0:
            res['rel_seqs'] = [[PAD_REL] * 3] * max_paths
            res['ent_indices'] = []
            return res

        u, v = u_list[0].item(), v_list[0].item()
        etype = subgraph.edata['type']

        def get_rels(src, dst):
            if not subgraph.has_edges_between(src, dst):
                return []
            eids = subgraph.edge_ids(src, dst)
            if isinstance(eids, torch.Tensor):
                eids = eids.view(-1).tolist()
            else:
                eids = [eids]
            return [etype[eid].item() for eid in eids]

        def make_path(rel_seq, ent_seq):
            return {
                'rel_seq': rel_seq,
                'ent_seq': [int(x) for x in ent_seq],
                'ent_set': set(int(x) for x in ent_seq)
            }

        def is_new_path(path):
            key = (tuple(path['rel_seq']), tuple(path['ent_seq']))
            if key in seen_paths:
                return False
            seen_paths.add(key)
            return True

        direct_paths = []
        candidate_paths = []
        max_candidates = self.llm_path_max_candidates

        def candidate_pool_full():
            return max_candidates > 0 and len(candidate_paths) >= max_candidates

        def add_candidate(path):
            if candidate_pool_full():
                return False
            if is_new_path(path):
                candidate_paths.append(path)
                return True
            return False

                                                   
        for r in get_rels(u, v):
            path = make_path([r, PAD_REL, PAD_REL], [u, v])
            if is_new_path(path):
                direct_paths.append(path)

                                                                             
        for n1 in subgraph.successors(u):
            if candidate_pool_full():
                break
            n1 = n1.item()
            if n1 == u or n1 == v:
                continue
            r1s, r2s = get_rels(u, n1), get_rels(n1, v)
            for r1 in r1s:
                for r2 in r2s:
                    path = make_path([r1, r2, PAD_REL], [u, n1, v])
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
                r1s, r2s, r3s = get_rels(u, n1), get_rels(n1, n2), get_rels(n2, v)
                for r1 in r1s:
                    for r2 in r2s:
                        for r3 in r3s:
                            path = make_path([r1, r2, r3], [u, n1, n2, v])
                            add_candidate(path)
                            if candidate_pool_full():
                                break
                        if candidate_pool_full():
                            break
                    if candidate_pool_full():
                        break

        remaining_slots = max_paths - len(direct_paths)
        if remaining_slots > 0 and use_llm_for_paths:
            candidate_paths = self._rank_candidate_paths_with_llm(subgraph, target_rel, candidate_paths)
        final_paths = direct_paths + candidate_paths

        for path in final_paths[:max_paths]:
            res['rel_seqs'].append(path['rel_seq'])
            res['ent_indices'].append(path['ent_set'])

        num_found = len(res['rel_seqs'])
        if num_found < max_paths:
            res['rel_seqs'] += [[PAD_REL] * 3] * (max_paths - num_found)

        return res
                                                              
              
                                          
             
                                            
                                                                        
                                                                        
        
                 
                                                                   
                                                                          
           
        
                                                  
                                                           
                                     
                        
    
                                                   
                                        
    
                             
                                 
                                                                    
                                                
                                                                              
                                 
                                                        
    
                         
                               
                        
                                                         
                                                           
                                               
    
                         
                                              
                                               
                                
                                                 
                                                             
                                
                                    
                                                                     
                                                                   
                                                               
    
                         
                                              
                                               
                                
                                                 
                                                    
                                    
                                                                 
                                                                                        
                                    
                                        
                                            
                                                                             
                                                                      
                                                                           
    
                  
                                          
                                   
                                                                          
        
                    
                                                                        
                                                                   
                                                  
        
                       
                                                                        
                                                                        
        
                                                  
                                                           
            
                                                   
                                        
                    
    
                                                             
                                  
                                                
                                      
                                                 
                                           
                                                    
    
                                 
                                              
                                                           
                              
                                                                   
                                                   
    
                                       
                                    
                        
                                                   
                                     
                                
                                                 
                
                                                                                             
                                                                       
                                                                       
                    
                                           
                                               
                                                                                              
                                                               
                                                           
                                                   
    
                                             
                                    
                                                   
                                     
                                
                                                 
                
                                                         
                                          
                                    
                                                                 
                    
                                                                
                                                                 
                                                           
                        
                                                                           
                                                                            
                                                                           
                        
                                               
                                                   
                                                       
                                                                                                                   
                                                                       
                                                                   
                                                               
                                                       
                                                   
    
                        
                       
                                                           
                                    
                                                                            
            
                                  
    def _prepare_features(self, subgraph, n_labels, n_feats=None):
                                                                        
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[1]                                                          
        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, r_label, n_feats=None):
                                                                        
                          
                                                                            
                       
        n_nodes = subgraph.number_of_nodes() 
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))                                                                                          
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1                                                                          
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1        
                                                                                              
                                                
                                                                      
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)
                                                                       
                                                        
                                   
                               
                                                      
        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1        
        n_ids[tail_id] = 2        
        
                                                                        
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)
        
                                                                            
        subgraph.ndata['r_label'] = torch.LongTensor(np.ones(n_nodes) * r_label)
        self.n_feat_dim = n_feats.shape[1]                                                          

        
        return subgraph



