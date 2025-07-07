import os
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict, OrderedDict
from sklearn.metrics import f1_score, auc, roc_auc_score, precision_recall_curve
import random
import copy
import json
import sys

# This data_url is for standard HGB datasets, not used for your custom 'ours' dataset
# but kept for completeness if this script is used for other datasets.
data_url = {
    'amazon': 'https://cloud.tsinghua.edu.cn/d/10974f42a5ab46b99b88/files/?p=%2Famazon_ini.zip&dl=1',
    'LastFM': 'https://cloud.tsinghua.edu.cn/d/10974f42a5ab46b99b88/files/?p=%2FLastFM_ini.zip&dl=1',
    'PubMed': 'https://cloud.tsinghua.edu.cn/d/10974f42a5ab46b99b88/files/?p=%2FPubMed_ini.zip&dl=1'
}

def download_and_extract(path): # Not used for your custom dataset if files are already present
    dataset_name = path.split('/')[-1]
    if dataset_name not in data_url:
        print(f"Info: Dataset {dataset_name} not in standard download list.")
        return
    prefix = os.path.join(*path.split('/')[:-1])
    os.makedirs(prefix, exist_ok=True)
    print(f"Attempting download for {dataset_name}...")
    os.system("wget \"{}\" -O {}".format(data_url[dataset_name], path+'.zip'))
    os.system("unzip -o {} -d {}".format(path+'.zip', prefix)) # -o to overwrite


class data_loader:
    def __init__(self, path, edge_types_to_evaluate=[]):
        self.path = path
        if not os.path.exists(path):
            if os.path.exists(path + '.zip'):
                print(f"Unzipping {path + '.zip'}")
                os.system("unzip -o {} -d {}".format(path + '.zip', os.path.join(*path.split('/')[:-1])))
                if os.path.exists(path + '_ini'): # Handle cases where unzip creates an _ini folder
                    os.system("mv {} {}".format(path + '_ini', path))
            else:
                dataset_name_for_url = path.split('/')[-1]
                if dataset_name_for_url in data_url:
                    print(f"Dataset {dataset_name_for_url} not found locally, attempting download.")
                    download_and_extract(path)
                    if os.path.exists(path + '_ini'):
                        os.system("mv {} {}".format(path + '_ini', path))
                else:
                    print(f"FATAL ERROR: Dataset path {path} not found, and not a known downloadable dataset.")
                    sys.exit(1)
        if not os.path.exists(path): 
            print(f"FATAL ERROR: Dataset path {path} still not found after attempting setup. Exiting.")
            sys.exit(1)

        print(f"INFO data_loader: Initializing for dataset at {self.path}")
        self.splited = False 
        self.nodes = self.load_nodes() 
        
        # Load the two separate link files
        self.links_init = self.load_links('link.dat') 
        self.raw_links_test_from_file = self.load_links('link.dat.test')

        # --- START: NEW CODE BLOCK TO MERGE GRAPHS ---
        print("INFO data_loader: Merging train and test links to create a full graph for message passing.")
        self.links_full_graph = copy.deepcopy(self.links_init)
        for r_id, test_matrix in self.raw_links_test_from_file['data'].items():
            if r_id in self.links_full_graph['data']:
                # Add the test links to the corresponding matrix in the full graph
                self.links_full_graph['data'][r_id] += test_matrix
            else:
                # If the relation type only exists in the test set, add it
                self.links_full_graph['data'][r_id] = test_matrix
                if r_id in self.raw_links_test_from_file['meta']:
                    self.links_full_graph['meta'][r_id] = self.raw_links_test_from_file['meta'][r_id]
        # --- END: NEW CODE BLOCK ---

        # The rest of the logic proceeds as before, but the splits happen on a copy
        self.train_pos, self.valid_pos = self.get_train_valid_pos(copy.deepcopy(self.links_init))
        self.train_neg = self.generate_negative_samples(self.train_pos, "training")
        self.valid_neg = self.generate_negative_samples(self.valid_pos, "validation")
        
        # The test set is still loaded from its original, separate file for evaluation
        self.links_test = self.raw_links_test_from_file
        self.test_types = list(self.links_test['data'].keys()) if not edge_types_to_evaluate else edge_types_to_evaluate
        print(f"INFO data_loader: Final test_types for evaluation: {self.test_types}")
        
        self.test_neg = {}
        if self.links_test['data']:
            test_pos_for_neg_sampling = defaultdict(lambda: [[], []])
            for r_id, sp_matrix in self.links_test['data'].items():
                if sp_matrix.nnz > 0:
                    rows, cols = sp_matrix.nonzero()
                    test_pos_for_neg_sampling[r_id][0].extend(rows)
                    test_pos_for_neg_sampling[r_id][1].extend(cols)
            if test_pos_for_neg_sampling:
                self.test_neg = self.generate_negative_samples(test_pos_for_neg_sampling, "testing")

        self.types = self.load_types('node.dat') 
        # self.links now correctly represents ONLY the training graph links for the loss function
        self.links = {'data': self.get_training_links_matrices(), 'meta': self.links_init['meta']}
        self.gen_transpose_links() 
        self.nonzero = False 

        print(f"INFO data_loader: Initialization complete.")
        print(f"  Total nodes: {self.nodes['total']}")
        print(f"  Node types and counts: {self.nodes['count']}")
        print(f"  Relation types in training graph (self.links['meta']): {list(self.links['meta'].keys()) if self.links and 'meta' in self.links else 'None'}")
        print(f"  Number of training positive links per type: { {r: len(p[0]) for r,p in self.train_pos.items()} if self.train_pos else 'None' }")
        print(f"  Number of training negative links per type: { {r: len(p[0]) for r,p in self.train_neg.items()} if self.train_neg else 'None' }")
        print(f"  Number of validation positive links per type: { {r: len(p[0]) for r,p in self.valid_pos.items()} if self.valid_pos else 'None' }")
        print(f"  Number of validation negative links per type: { {r: len(p[0]) for r,p in self.valid_neg.items()} if self.valid_neg else 'None' }")
        print(f"  Relation types in test set (self.links_test['meta']): {list(self.links_test['meta'].keys()) if self.links_test and 'meta' in self.links_test else 'None'}")
        print(f"  Number of positive test links per type (self.links_test['count']): {self.links_test['count'] if self.links_test else 'None'}")
        print(f"  Number of negative test links per type: { {r: len(p[0]) for r,p in self.test_neg.items()} if self.test_neg else 'None' }")

    def get_training_links_matrices(self):
        training_links_data = {} 
        for r_id, pairs in self.train_pos.items():
            if pairs and len(pairs[0]) > 0:
                # Assuming weights are 1.0 if not explicitly carried from original file to train_pos
                # If train_pos needs to store weights, get_train_valid_pos should be modified.
                weights = [1.0] * len(pairs[0]) 
                training_links_data[r_id] = sp.csr_matrix((weights, (pairs[0], pairs[1])),
                                                           shape=(self.nodes['total'], self.nodes['total']))
            else:
                training_links_data[r_id] = sp.csr_matrix((self.nodes['total'], self.nodes['total']), dtype=np.float32)
        
        if hasattr(self, 'links_init') and 'meta' in self.links_init:
            for r_id in self.links_init['meta'].keys():
                if r_id not in training_links_data:
                    training_links_data[r_id] = sp.csr_matrix((self.nodes['total'], self.nodes['total']), dtype=np.float32)
        return training_links_data

    def get_train_valid_pos(self, links_for_splitting, train_ratio=0.9):
        if self.splited:
            return getattr(self,'train_pos',{}), getattr(self,'valid_pos',{})

        train_pos_links = defaultdict(lambda: [[], []]) 
        valid_pos_links = defaultdict(lambda: [[], []])
        
        print(f"Splitting links from link.dat into train/valid for relations: {list(links_for_splitting['data'].keys())}")

        for r_id, sp_matrix in links_for_splitting['data'].items():
            if sp_matrix.nnz == 0: continue

            current_links_coo = sp_matrix.tocoo()
            links_by_head = defaultdict(list)
            link_weights = current_links_coo.data if hasattr(current_links_coo, 'data') and current_links_coo.data is not None and current_links_coo.data.size == current_links_coo.row.size else [1.0]*len(current_links_coo.row)

            for h_id, t_id, weight in zip(current_links_coo.row, current_links_coo.col, link_weights):
                links_by_head[h_id].append({'tail': t_id, 'weight': weight}) # Store weight too
            
            for h_id, t_id_details_list in links_by_head.items():
                if not t_id_details_list: continue
                random.shuffle(t_id_details_list)
                if t_id_details_list:
                    first_link_detail = t_id_details_list.pop(0)
                    train_pos_links[r_id][0].append(h_id)
                    train_pos_links[r_id][1].append(first_link_detail['tail'])
                    # train_pos_links[r_id] could also store weights if needed by GNN directly
                for link_detail in t_id_details_list:
                    if random.random() < train_ratio:
                        train_pos_links[r_id][0].append(h_id)
                        train_pos_links[r_id][1].append(link_detail['tail'])
                    else:
                        valid_pos_links[r_id][0].append(h_id)
                        valid_pos_links[r_id][1].append(link_detail['tail'])
        self.splited = True
        return dict(train_pos_links), dict(valid_pos_links)

    def generate_negative_samples(self, positive_samples_dict, set_name_for_log="unknown"):
        neg_samples = defaultdict(lambda: [[], []])
        print(f"Generating negative samples for '{set_name_for_log}' set...")
        all_known_true_links = defaultdict(set)
        source_link_sets_for_filtering = [self.links_init['data']] 
        if hasattr(self, 'raw_links_test_from_file') and isinstance(self.raw_links_test_from_file, dict) and self.raw_links_test_from_file.get('data'):
            source_link_sets_for_filtering.append(self.raw_links_test_from_file['data'])
        for link_set in source_link_sets_for_filtering:
            if isinstance(link_set, dict):
                for r_id, sp_matrix in link_set.items():
                    if isinstance(sp_matrix, sp.spmatrix):
                        r, c = sp_matrix.nonzero()
                        for h, t in zip(r,c): all_known_true_links[r_id].add((h,t))
        for r_id_pos, pos_pairs_current_set in positive_samples_dict.items():
             if isinstance(pos_pairs_current_set, (list, np.ndarray)) and len(pos_pairs_current_set) == 2:
                 for i in range(len(pos_pairs_current_set[0])):
                     all_known_true_links[r_id_pos].add((pos_pairs_current_set[0][i], pos_pairs_current_set[1][i]))

        for r_id, pos_pairs in positive_samples_dict.items():
            if not pos_pairs or not pos_pairs[0]: continue
            if r_id not in self.links_init['meta']:
                print(f"  Warning (generate_negative_samples for '{set_name_for_log}'): r_id {r_id} not in self.links_init['meta']. Skipping."); continue
            _ , target_node_type_id = self.links_init['meta'][r_id] 
            if target_node_type_id not in self.nodes['shift'] or target_node_type_id not in self.nodes['count']:
                print(f"  Warning (GSN for '{set_name_for_log}', r_id {r_id}): Target type {target_node_type_id} info missing. Skipping."); continue
            target_type_global_start_id = self.nodes['shift'][target_node_type_id]
            target_type_node_count = self.nodes['count'][target_node_type_id]
            if target_type_node_count == 0: continue
            num_positive_samples_for_rid = len(pos_pairs[0])
            current_neg_heads, current_neg_tails = [], []
            for i in range(num_positive_samples_for_rid): 
                head_node_global_id = pos_pairs[0][i]
                sampled_neg_tail_global_id = -1; num_attempts = 0
                max_sampling_attempts = min(target_type_node_count * 2, 200) 
                while num_attempts < max_sampling_attempts:
                    random_local_tail_id = random.randrange(0, target_type_node_count)
                    sampled_neg_tail_global_id = target_type_global_start_id + random_local_tail_id
                    if (head_node_global_id, sampled_neg_tail_global_id) not in all_known_true_links[r_id]: break 
                    num_attempts += 1
                if sampled_neg_tail_global_id == -1 and target_type_node_count > 0 : 
                     sampled_neg_tail_global_id = target_type_global_start_id + random.randrange(0, target_type_node_count)
                if sampled_neg_tail_global_id != -1:
                    current_neg_heads.append(head_node_global_id); current_neg_tails.append(sampled_neg_tail_global_id)
            neg_samples[r_id] = [current_neg_heads, current_neg_tails]
            if len(current_neg_heads) > 0 : print(f"  Generated {len(neg_samples[r_id][0])} negative samples for relation {r_id} in '{set_name_for_log}' set.")
        return dict(neg_samples)

    def get_train_neg(self): return getattr(self, 'train_neg', defaultdict(lambda: [[], []]))
    def get_valid_neg(self): return getattr(self, 'valid_neg', defaultdict(lambda: [[], []]))
        
    def get_test_neigh(self): 
        random.seed(1) 
        final_test_pairs = defaultdict(lambda: [[], []]); final_test_labels = defaultdict(list)
        if not self.test_types: print("get_test_neigh: No self.test_types. Returning empty."); return {}, {}
        
        all_true_pos_links = defaultdict(set) 
        source_link_sets = [self.links_init['data']] 
        if hasattr(self, 'raw_links_test_from_file') and isinstance(self.raw_links_test_from_file,dict) and self.raw_links_test_from_file.get('data'):
            source_link_sets.append(self.raw_links_test_from_file['data'])
        for link_set in source_link_sets:
            if isinstance(link_set, dict):
                for r_id, sp_matrix in link_set.items():
                    if isinstance(sp_matrix, sp.spmatrix):
                        r, c = sp_matrix.nonzero(); [all_true_pos_links[r_id].add((h,t)) for h,t in zip(r,c)]

        for r_id in self.test_types:
            # self.links_test['data'] is {r_id: [[h],[t]]} from valid_pos
            if r_id not in self.links_test['data'] or not self.links_test['data'][r_id][0]:
                print(f"  get_test_neigh: No positive test links for r_id {r_id}. Skipping."); continue
            pos_test_heads, pos_test_tails = self.links_test['data'][r_id][0], self.links_test['data'][r_id][1]
            if r_id not in self.links_test['meta']:
                 print(f"  Warning (get_test_neigh): Meta info for r_id {r_id} missing. Skipping."); continue
            _ , target_node_type_id = self.links_test['meta'][r_id]
            target_type_global_start_id = self.nodes['shift'][target_node_type_id]
            target_type_node_count = self.nodes['count'][target_node_type_id]

            for h_id, t_id in zip(pos_test_heads, pos_test_tails):
                final_test_pairs[r_id][0].append(h_id); final_test_pairs[r_id][1].append(t_id)
                final_test_labels[r_id].append(1)
                if target_type_node_count == 0: continue
                neg_t_id_global = -1; attempts = 0; max_attempts = min(100, target_type_node_count * 2)
                while attempts < max_attempts:
                    rand_local_t_id = random.randrange(0, target_type_node_count)
                    neg_t_id_global = target_type_global_start_id + rand_local_t_id
                    if (h_id, neg_t_id_global) not in all_true_pos_links[r_id]: break
                    attempts += 1
                if neg_t_id_global != -1 :
                    final_test_pairs[r_id][0].append(h_id); final_test_pairs[r_id][1].append(neg_t_id_global)
                    final_test_labels[r_id].append(0)
        return dict(final_test_pairs), dict(final_test_labels)

    def get_test_neigh_w_random(self):
        print("INFO: get_test_neigh_w_random() is currently aliased to get_test_neigh().")
        return self.get_test_neigh()

    def get_test_neigh_full_random(self):
        print("WARNING: get_test_neigh_full_random() not fully implemented for typical LP eval. Returning empty.")
        return {}, {} # Needs careful review if this specific sampling is required
        
    def gen_transpose_links(self):
        self.links['data_trans'] = {} # Use regular dict
        if hasattr(self, 'links') and 'data' in self.links and isinstance(self.links['data'], dict):
            print(f"INFO data_loader: Generating transposed link matrices for {len(self.links['data'])} relation types in self.links['data'] (training graph)...")
            for r_id, sp_matrix in self.links['data'].items(): 
                if isinstance(sp_matrix, sp.spmatrix):
                    self.links['data_trans'][r_id] = sp_matrix.transpose().tocsr()
                else: # Should be a sparse matrix after get_training_links_matrices
                    self.links['data_trans'][r_id] = sp.csr_matrix((self.nodes['total'], self.nodes['total']), dtype=np.float32)
            # Ensure all original relation types have an entry in data_trans
            if hasattr(self, 'links_init') and 'meta' in self.links_init:
                for r_id_orig in self.links_init['meta'].keys():
                    if r_id_orig not in self.links['data_trans']:
                        self.links['data_trans'][r_id_orig] = sp.csr_matrix((self.nodes['total'], self.nodes['total']), dtype=np.float32)
        else:
            print("WARNING data_loader (gen_transpose_links): self.links['data'] not setup. No transposed links generated.")

    def load_links(self, name_of_link_file):
        links = {'total': 0, 'count': Counter(), 'meta': {}, 'data': {}} # Use regular dict
        link_file_path = os.path.join(self.path, name_of_link_file)

        if not hasattr(self, 'nodes') or not self.nodes or self.nodes.get('total',0) == 0:
             print(f"FATAL ERROR in load_links: self.nodes not properly loaded before {name_of_link_file}.")
             return links 
        if not os.path.exists(link_file_path):
            print(f"INFO data_loader: Link file {link_file_path} not found. Returning empty links for this part.")
            return links 
        
        print(f"INFO data_loader: Loading links from {link_file_path}...")
        line_count_for_file = 0
        temp_links_data_tuples = defaultdict(list) 
        with open(link_file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                th = line.strip().split('\t')
                if len(th) < 4: 
                    if name_of_link_file == 'link.dat.test' and line.strip() == '' and line_idx == 0: break 
                    print(f"Warning: Line {line_idx+1} in {name_of_link_file} has {len(th)} columns (expected 4): '{line.strip()}'. Skipping.")
                    continue
                try: h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(th[2]), float(th[3]) 
                except ValueError: print(f"Warning: ValueError parsing line {line_idx+1} in {name_of_link_file}. Skipping."); continue
                if not (0 <= h_id < self.nodes['total'] and 0 <= t_id < self.nodes['total']):
                    print(f"Warning: Node ID out of bounds in {name_of_link_file}, line {line_idx+1}. Skipping."); continue
                if r_id not in links['meta']:
                    h_type, t_type = self.get_node_type(h_id), self.get_node_type(t_id)
                    if h_type == -1 or t_type == -1: print(f"Warning: Could not determine node types for link in {name_of_link_file}. Skipping."); continue
                    links['meta'][r_id] = (h_type, t_type)
                temp_links_data_tuples[r_id].append((h_id, t_id, link_weight))
                links['count'][r_id] += 1; links['total'] += 1; line_count_for_file +=1
        print(f"  Read {line_count_for_file} link entries from {name_of_link_file}.")
        
        final_data_matrices = {} 
        for r_id_in_meta in links['meta'].keys(): 
            list_of_triples = temp_links_data_tuples.get(r_id_in_meta, [])
            if list_of_triples: final_data_matrices[r_id_in_meta] = self.list_to_sp_mat(list_of_triples, self.nodes['total'])
            else: final_data_matrices[r_id_in_meta] = sp.csr_matrix((self.nodes['total'], self.nodes['total']), dtype=np.float32)
        links['data'] = final_data_matrices
        return links

    def load_nodes(self):
        print(f"INFO data_loader: Loading nodes from {os.path.join(self.path, 'node.dat')}...")
        nodes = {'total': 0, 'count': Counter(), 'shift': {}, 'type_map': {}}
        
        # First pass: count nodes per type to calculate total and shifts
        with open(os.path.join(self.path, 'node.dat'), 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if len(line) < 3: continue
                try:
                    node_type = int(line[2])
                    nodes['count'][node_type] += 1
                    nodes['total'] += 1
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed line in node.dat: {line}")
                    continue
        
        current_shift = 0
        for t_id in sorted(nodes['count'].keys()):
            nodes['shift'][t_id] = current_shift
            current_shift += nodes['count'][t_id]

        # Second pass: create the type_map with correct local IDs
        with open(os.path.join(self.path, 'node.dat'), 'r') as f:
            local_id_counters = Counter()
            for line in f:
                line = line.strip().split('\t')
                if len(line) < 3: continue
                try:
                    node_id, node_name, node_type = int(line[0]), line[1], int(line[2])
                    local_id = local_id_counters[node_type]
                    nodes['type_map'][node_id] = [node_type, local_id]
                    local_id_counters[node_type] += 1
                except (ValueError, IndexError):
                    # Warning already printed in first pass
                    continue

        print(f"INFO data_loader: Nodes loaded. Total: {nodes['total']}, Counts: {nodes['count']}")
        return nodes

    def load_types(self, name_of_node_file):
        types_info = {'types': [], 'total': 0, 'data': {}}
        node_file_path = os.path.join(self.path, name_of_node_file)
        if not os.path.exists(node_file_path): print(f"ERROR in load_types: {node_file_path} not found."); return types_info
        print(f"INFO data_loader: Loading node type assignments from {node_file_path} (for self.types)...")
        unique_type_ids = set()
        with open(node_file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                th = line.strip().split('\t')
                if len(th) < 3: continue
                try: node_id_global, _, node_type_id = int(th[0]), th[1], int(th[2])
                except ValueError: continue
                types_info['data'][node_id_global] = node_type_id
                unique_type_ids.add(node_type_id); types_info['total'] += 1
        types_info['types'] = sorted(list(unique_type_ids))
        print(f"INFO data_loader: Finished loading types. Mapped: {types_info['total']}, Unique types: {types_info['types']}")
        return types_info

    def list_to_sp_mat(self, list_of_link_triples, num_total_nodes_shape):
        if not list_of_link_triples: return sp.csr_matrix((num_total_nodes_shape, num_total_nodes_shape), dtype=np.float32)
        data_values = [x[2] for x in list_of_link_triples]; row_indices = [x[0] for x in list_of_link_triples]; col_indices = [x[1] for x in list_of_link_triples]
        return sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(num_total_nodes_shape, num_total_nodes_shape)).tocsr()

    @staticmethod
    def evaluate(edge_list_preds, confidence, true_labels):
        if not isinstance(edge_list_preds, (list, np.ndarray)) or len(edge_list_preds) != 2 or \
           not isinstance(edge_list_preds[0], (list, np.ndarray)) or not isinstance(edge_list_preds[1], (list, np.ndarray)) :
            return {'roc_auc': 0.0, 'MRR': 0.0}
        if not (len(edge_list_preds[0]) == len(confidence) == len(true_labels)): return {'roc_auc': 0.0, 'MRR': 0.0}
        if len(true_labels) == 0: return {'roc_auc': 0.0, 'MRR': 0.0}
        confidence_arr, true_labels_arr = np.array(confidence), np.array(true_labels)
        roc_auc_val = 0.0
        try:
            if len(np.unique(true_labels_arr)) < 2: roc_auc_val = 0.0
            else: roc_auc_val = roc_auc_score(true_labels_arr, confidence_arr)
        except ValueError: roc_auc_val = 0.0
        mrr_list = []; preds_by_head = defaultdict(list)
        for i in range(len(edge_list_preds[0])):
            preds_by_head[edge_list_preds[0][i]].append({'conf': confidence_arr[i], 'label': true_labels_arr[i]})
        for h_id, h_preds_list in preds_by_head.items():
            if not any(p_item['label'] == 1 for p_item in h_preds_list): continue
            h_preds_sorted = sorted(h_preds_list, key=lambda x_item: x_item['conf'], reverse=True)
            for rank_idx, pred_item in enumerate(h_preds_sorted):
                if pred_item['label'] == 1: mrr_list.append(1.0 / (rank_idx + 1)); break 
        mrr_val = np.mean(mrr_list) if mrr_list else 0.0
        return {'roc_auc': roc_auc_val, 'MRR': mrr_val}

    def get_node_type(self, node_id_global):
        for type_id_int in sorted(self.nodes['shift'].keys()):
            start_id = self.nodes['shift'][type_id_int]
            num_nodes_of_this_type = self.nodes['count'].get(type_id_int, 0)
            if start_id <= node_id_global < start_id + num_nodes_of_this_type: return type_id_int
        return -1

    def get_edge_type(self, info_tuple_or_id):
        if isinstance(info_tuple_or_id, int): return info_tuple_or_id
        if isinstance(info_tuple_or_id, list) and len(info_tuple_or_id)==1 and isinstance(info_tuple_or_id[0],int): return info_tuple_or_id[0]
        if isinstance(info_tuple_or_id, (tuple, list)) and len(info_tuple_or_id) == 2:
            target_meta = tuple(info_tuple_or_id)
            for r_id, meta_tuple_stored in self.links_init['meta'].items(): # Check against initial full meta
                if meta_tuple_stored == target_meta: return r_id
            reversed_meta = (info_tuple_or_id[1], info_tuple_or_id[0])
            for r_id, meta_tuple_stored in self.links_init['meta'].items():
                if meta_tuple_stored == reversed_meta: return -r_id - 1 
            return None
        return None

    def get_edge_info(self, edge_id_int):
        return self.links_init['meta'].get(edge_id_int, None) # Use links_init

    # --- Metapath related functions (get_meta_path, get_nonzero, dfs, get_full_meta_path) ---
    # These were present in your original script. Included for completeness.
    # Ensure they work correctly with the modified data structures if you use them.
    def get_meta_path(self, meta_path_def=[]):
        ini = sp.eye(self.nodes['total'], dtype=np.float32).tocsr()
        for item in meta_path_def:
            edge_type_id = self.get_edge_type(item)
            if edge_type_id is None: return sp.csr_matrix((self.nodes['total'], self.nodes['total']))
            matrix_to_multiply = None
            if edge_type_id >= 0:
                matrix_to_multiply = self.links['data'].get(edge_type_id) # Use training graph links
            else:
                original_id = -edge_type_id - 1
                matrix_to_multiply = self.links['data_trans'].get(original_id)
            if matrix_to_multiply is None: return sp.csr_matrix((self.nodes['total'], self.nodes['total']))
            ini = ini.dot(matrix_to_multiply)
        return ini

    def get_nonzero(self):
        if self.nonzero: return
        self.nonzero = True; self.re_cache = defaultdict(lambda: defaultdict(list))
        for r_id, sp_matrix in self.links['data'].items(): # Based on training graph links
            csr_mat = sp_matrix.tocsr()
            for i in range(csr_mat.shape[0]): self.re_cache[r_id][i] = csr_mat[i].nonzero()[1].tolist()
        for r_id, sp_matrix_trans in self.links['data_trans'].items():
            csr_mat_trans = sp_matrix_trans.tocsr()
            self.re_cache[-r_id-1] = defaultdict(list)
            for i in range(csr_mat_trans.shape[0]): self.re_cache[-r_id-1][i] = csr_mat_trans[i].nonzero()[1].tolist()

    def dfs(self, current_path_nodes, meta_path_edge_types, found_paths_dict):
        if not meta_path_edge_types:
            found_paths_dict[current_path_nodes[0]].append(list(current_path_nodes)); return
        last_node = current_path_nodes[-1]; edge_type = meta_path_edge_types[0]
        if last_node in self.re_cache.get(edge_type, {}):
            for neighbor in self.re_cache[edge_type][last_node]:
                self.dfs(current_path_nodes + [neighbor], meta_path_edge_types[1:], found_paths_dict)

    def get_full_meta_path(self, meta_path_def=[], symmetric=False):
        if not meta_path_def: return {}
        if not self.nonzero: self.get_nonzero()
        int_meta_path = [self.get_edge_type(item) for item in meta_path_def]
        if any(et_id is None for et_id in int_meta_path): return {}
        
        paths_found = defaultdict(list); first_et = int_meta_path[0]
        start_node_type = -1
        meta_ref = self.links_init['meta'] # Use original full graph meta for type definition
        if first_et >= 0:
            if first_et not in meta_ref: return {}
            start_node_type = meta_ref[first_et][0]
        else:
            orig_et = -first_et - 1
            if orig_et not in meta_ref: return {}
            start_node_type = meta_ref[orig_et][1]
        
        if start_node_type == -1 or start_node_type not in self.nodes['shift']: return {}
        start_idx, end_idx = self.nodes['shift'][start_node_type], self.nodes['shift'][start_node_type] + self.nodes['count'][start_node_type]
        for i in range(start_idx, end_idx): self.dfs([i], int_meta_path, paths_found)
        if symmetric and len(int_meta_path) > 1: print("Warning: Symmetric metapath > 1 not fully implemented as per original HGB logic without further review.")
        return dict(paths_found)

    def sample_link_prediction_batch(self, relation_id, batch_size=256, split='train'):
        """
        Sample a batch of (u, v, label) pairs for link prediction for a given relation type.
        - relation_id: int, the relation type (e.g., 0 for cell-drug, 1 for drug-cell)
        - batch_size: int, total number of samples (half positive, half negative)
        - split: 'train' or 'valid'
        Returns: list of (u, v, label)
        """
        if split == 'train':
            pos = self.train_pos.get(relation_id, [[], []])
            neg = self.train_neg.get(relation_id, [[], []])
        elif split == 'valid':
            pos = self.valid_pos.get(relation_id, [[], []])
            neg = self.valid_neg.get(relation_id, [[], []])
        else:
            raise ValueError("split must be 'train' or 'valid'")
        n_pos = min(len(pos[0]), batch_size // 2)
        n_neg = min(len(neg[0]), batch_size - n_pos)
        pos_indices = np.random.choice(len(pos[0]), n_pos, replace=False) if n_pos > 0 else []
        neg_indices = np.random.choice(len(neg[0]), n_neg, replace=False) if n_neg > 0 else []
        batch = []
        for idx in pos_indices:
            batch.append((pos[0][idx], pos[1][idx], 1))
        for idx in neg_indices:
            batch.append((neg[0][idx], neg[1][idx], 0))
        np.random.shuffle(batch)
        return batch