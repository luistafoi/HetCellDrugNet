import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter, defaultdict 
import sys
import os
import json
import torch as th

class input_data(object):
    def __init__(self, args, dl, temp_dir):
        self.args = args
        self.dl = dl
        self.temp_dir = temp_dir
        
        print("--- Inside data_generator.input_data.__init__ ---")
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        self.node_n, self.node_shift = node_n, node_shift
        
        # This part is unchanged
        info_dat_abs_path = os.path.join(self.dl.path, 'info.dat')
        print(f"  Attempting to load info.dat from: {info_dat_abs_path}")
        try:
            with open(info_dat_abs_path, 'r', encoding='utf-8') as f:
                data_info_content = json.load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load or parse {info_dat_abs_path}: {e}")
            sys.exit(1)

        self.node_type2name, self.node_name2type = dict(), dict()
        try:
            # First, try the strict HGB format which has a "node type" key
            node_type_mapping = data_info_content["node.dat"]["node type"]
        except KeyError:
            # If that fails, it means we have the simpler format.
            print("  > Info: 'node type' key not found in info.dat, using simpler format.")
            node_type_mapping = data_info_content["node.dat"]
        except Exception as e:
            print(f"FATAL ERROR: Could not parse info.dat. Unexpected structure. Error: {e}")
            sys.exit(1)

        for type_id_str, type_info_list in node_type_mapping.items():
            type_id_int = int(type_id_str)
            type_name = type_info_list[0]
            self.node_type2name[type_id_int] = type_name
            self.node_name2type[type_name] = type_id_int
            
        print(f"  self.node_type2name map: {self.node_type2name}")
        print(f"  self.node_name2type map: {self.node_name2type}")
        
        # This part is also unchanged
        num_node_types_actual = len(self.dl.nodes['count'])
        self.standand_node_L = [20] * num_node_types_actual
        self.top_k = {i: 5 for i in range(num_node_types_actual)}
        self.neigh_L = sum(self.standand_node_L) if self.standand_node_L else 50
        self.while_max_count = 1e5
        
        # --- START: THE SINGLE MOST IMPORTANT FIX ---
        # Initialize an empty neighbor list for ALL nodes in the graph
        neigh_list = {nt: [[] for _ in range(node_n[nt])] for nt in node_n}

        print("\n  Populating neighbor lists using the FULL merged graph...")
        # Use the new 'links_full_graph' object we created in the data_loader
        for r_id, sp_matrix in dl.links_full_graph['data'].items():
            h_type, t_type = dl.links_full_graph['meta'][r_id]
            rows, cols = sp_matrix.nonzero()
            
            # Populate the lists using correct local IDs from the type_map
            for r_global, c_global in zip(rows, cols):
                if r_global not in dl.nodes['type_map'] or c_global not in dl.nodes['type_map']:
                    continue

                h_id_local = dl.nodes['type_map'][r_global][1]
                t_id_local = dl.nodes['type_map'][c_global][1]

                if dl.nodes['type_map'][r_global][0] != h_type or dl.nodes['type_map'][c_global][0] != t_type:
                    continue

                neighbor_string = f"{self.node_type2name[t_type]}{t_id_local}"
                if h_id_local < len(neigh_list[h_type]):
                     neigh_list[h_type][h_id_local].append(neighbor_string)

        self.edge_list = {} # We don't need the intermediate edge_list anymore
        self.neigh_list = neigh_list
        # --- END: THE SINGLE MOST IMPORTANT FIX ---
        
        # Initialize neigh_list_train to prevent AttributeError
        self.neigh_list_train = {nt: [[] for _ in range(self.node_n[nt])] for nt in self.node_n}

        print("--- Finished data_generator.input_data.__init__ ---")

    def load_het_neigh_train(self, file_path):
        """
        Loads the training neighbors (from random walks) from a file.
        This populates self.neigh_list_train.
        """
        print(f"Info: Loading pre-generated training neighbors from {file_path}")
        self.neigh_list_train = {}
        node_n = self.dl.nodes['count']
        
        # Initialize empty lists for all node types
        for node_type in self.node_type2name.keys():
            if node_type in node_n:
                self.neigh_list_train[node_type] = [[] for _ in range(node_n[node_type])]

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) != 2:
                        continue
                    center_node_str, neigh_list_str = parts
                    
                    # Parse center node string (e.g., "cell0", "drug5")
                    center_type = -1
                    center_local_id = -1
                    for type_id, type_name in self.node_type2name.items():
                        if center_node_str.startswith(type_name):
                            id_part = center_node_str[len(type_name):]
                            if id_part.isdigit():
                                center_type = type_id
                                center_local_id = int(id_part)
                                break
                    
                    if center_type != -1 and center_local_id != -1:
                        if center_type in self.neigh_list_train and center_local_id < len(self.neigh_list_train[center_type]):
                            self.neigh_list_train[center_type][center_local_id] = neigh_list_str.split(',')
            print("Info: Finished loading training neighbors.")
        except FileNotFoundError:
            print(f"ERROR: Could not find neighbor file at {file_path}. Please generate it first.")
            # Proceeding with empty neigh_list_train, which might cause downstream errors
            pass

    def gen_het_w_walk_restart(self, file_):
        print(f'Info: generate {file_} start.')
        node_n = self.dl.nodes['count']
        neigh_list_train = dict()
        active_node_types = list(self.node_type2name.keys())

        for node_type in active_node_types:
            if node_type not in node_n:
                print(f"  Warning: Node type {node_type} from mapping not in dl.nodes['count']. Skipping for walk_restart init.")
                continue
            neigh_list_train[node_type] = [[] for _ in range(node_n[node_type])]

        # Sort node type names by length (desc) to avoid prefix ambiguity (e.g., 'drug' vs 'drug_target')
        sorted_node_types_by_name_len = sorted(self.node_type2name.items(), key=lambda item: len(item[1]), reverse=True)

        print(f"  Starting random walks with restart for {len(active_node_types)} node types...")
        for node_type in active_node_types:
            if node_type not in node_n: continue
            print(f"    Walking for node type: {node_type} ({self.node_type2name.get(node_type, 'Unknown')}), count: {node_n[node_type]}")
            for n_id_local in range(node_n[node_type]):
                if n_id_local >= len(self.neigh_list.get(node_type, [])):
                    continue

                neigh_temp = self.neigh_list[node_type][n_id_local]
                neigh_train = neigh_list_train[node_type][n_id_local]
                
                if node_type not in self.node_type2name:
                    print(f"      Error: Starting node_type {node_type} not in self.node_type2name. Skipping node {n_id_local}.")
                    continue
                curNode = self.node_type2name[node_type] + str(n_id_local)
                curNodeType = node_type

                if len(neigh_temp):
                    neigh_L_count = 0
                    node_L_counts = np.zeros(len(self.dl.nodes['count']), dtype=int)
                    while_iter_count = 0
                    
                    while neigh_L_count < self.neigh_L and while_iter_count < self.while_max_count:
                        while_iter_count += 1
                        rand_p = random.random()
                        
                        if rand_p > 0.5:
                            # Robustly extract local ID from curNode
                            curNode_parsed = False
                            for mapped_type_id, mapped_type_name in sorted_node_types_by_name_len:
                                if curNode.startswith(mapped_type_name):
                                    # Defined with an underscore
                                    local_id_str_from_curNode = curNode[len(mapped_type_name):]
                                    # USE the same name (with an underscore)
                                    if local_id_str_from_curNode.isdigit():
                                        # And USE it again here
                                        curNode_local_id = int(local_id_str_from_curNode)
                                        curNodeType = mapped_type_id
                                        curNode_parsed = True
                                        break
                            if not curNode_parsed:
                                print(f"        ERROR: Cannot parse local ID from '{curNode}'. Breaking walk.")
                                break

                            if curNode_local_id >= len(self.neigh_list[curNodeType]):
                                print(f"        ERROR: curNode_local_id {curNode_local_id} is out of range for self.neigh_list[{curNodeType}] (len {len(self.neigh_list[curNodeType])}). curNode was '{curNode}'. Breaking walk.")
                                break
                            
                            if not self.neigh_list[curNodeType][curNode_local_id]:
                                curNode = self.node_type2name[node_type] + str(n_id_local)
                                curNodeType = node_type
                                continue
                                
                            curNode = random.choice(self.neigh_list[curNodeType][curNode_local_id])
                            
                            # Parse new curNode's type and local ID
                            new_curNode_type_parsed = False
                            for mapped_type_id, mapped_type_name in sorted_node_types_by_name_len:
                                if curNode.startswith(mapped_type_name):
                                    curNodeType = mapped_type_id
                                    new_curNode_type_parsed = True
                                    break
                            if not new_curNode_type_parsed:
                                print(f"        ERROR: Could not parse type from new curNode '{curNode}'. Breaking walk.")
                                break
                            
                            if curNodeType >= len(self.standand_node_L) or curNodeType >= len(node_L_counts):
                                print(f"        ERROR: new curNodeType {curNodeType} ('{self.node_type2name.get(curNodeType)}') is out of bounds for standand_node_L/node_L_counts. Breaking walk.")
                                break

                            if node_L_counts[curNodeType] < self.standand_node_L[curNodeType]:
                                neigh_train.append(curNode)
                                neigh_L_count += 1
                                node_L_counts[curNodeType] += 1
                        else:
                            curNode = self.node_type2name[node_type] + str(n_id_local)
                            curNodeType = node_type
        
        # ... (rest of the function to write to file remains the same) ...
        for node_type_outer_idx in active_node_types:
             if node_type_outer_idx not in node_n: continue
             for n_id_inner_idx in range(node_n[node_type_outer_idx]):
                 neigh_list_train[node_type_outer_idx][n_id_inner_idx] = list(set(neigh_list_train[node_type_outer_idx][n_id_inner_idx])) # Make unique

        neigh_f = open(file_, "w")
        for node_type_outer_idx in active_node_types:
            if node_type_outer_idx not in node_n: continue
            for n_id_inner_idx in range(node_n[node_type_outer_idx]):
                neigh_train_content = neigh_list_train[node_type_outer_idx][n_id_inner_idx]
                curNode_str = self.node_type2name[node_type_outer_idx] + str(n_id_inner_idx)
                if len(neigh_train_content):
                    neigh_f.write(curNode_str + ":")
                    for k_idx in range(len(neigh_train_content) - 1):
                        neigh_f.write(neigh_train_content[k_idx] + ",")
                    neigh_f.write(neigh_train_content[-1] + "\n")
        neigh_f.close()
        self.neigh_list_train = neigh_list_train
        print(f'Info: generate {file_} done.')

    def gen_het_w_walk(self, file_):
        print(f'Info: generate {file_} start, for ALL node types.')
        het_walk_f = open(file_, "w")
        
        # --- NEW: Loop over every node type to start walks ---
        for node_type in self.node_n.keys():
            print(f"  Starting walks from node type: {node_type} ({self.node_type2name[node_type]})")
            
            # --- NEW: Loop over every node of that type ---
            for n_id_local in range(self.node_n[node_type]):
                # Start a fixed number of walks from this specific node
                for i in range(self.args.walk_n):
                    
                    # Make sure the starting node has neighbors
                    if n_id_local >= len(self.neigh_list.get(node_type, [])) or not self.neigh_list[node_type][n_id_local]:
                        continue

                    curNode = self.node_type2name[node_type] + str(n_id_local)
                    curNodeType = node_type
                    
                    current_walk = [curNode]
                    for l_idx in range(self.args.walk_L - 1):
                        # Robustly extract local ID string part
                        current_node_type_prefix = self.node_type2name.get(curNodeType)
                        if not current_node_type_prefix: break
                        
                        local_id_str_from_curNode = curNode.replace(current_node_type_prefix, "", 1)
                        
                        if not local_id_str_from_curNode.isdigit(): break
                        curNode_local_id = int(local_id_str_from_curNode)

                        if curNode_local_id >= len(self.neigh_list[curNodeType]) or not self.neigh_list[curNodeType][curNode_local_id]:
                            break

                        curNode = random.choice(self.neigh_list[curNodeType][curNode_local_id])
                        current_walk.append(curNode)
                        
                        new_curNode_type_parsed = False
                        for mapped_type_id, mapped_type_name in self.node_type2name.items():
                            if curNode.startswith(mapped_type_name):
                                curNodeType = mapped_type_id
                                new_curNode_type_parsed = True
                                break
                        if not new_curNode_type_parsed: break
                    
                    het_walk_f.write(" ".join(current_walk) + "\n")

        het_walk_f.close()
        print(f'Info: generate {file_} done.')

    def compute_sample_p(self):
        print("Info: compute sampling ratio for each kind of triple start.")
        # Ensure self.temp_dir is set, e.g., from args or main script context
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
             # Fallback or error: temp_dir should be initialized based on dataset name
             # This was added to __init__: self.temp_dir = os.path.join(sys.path[0], f'{self.args.data}-temp')
             # Here sys.path[0] refers to data_generator.py's location.
             # For HetGNN LP, temp_dir is in LP/benchmark/methods/HetGNN/CellDrug-temp/
             # For HetGNN NC, temp_dir is in NC/benchmark/methods/HetGNN/code/ACM/ACM-temp/
             # Let's assume self.temp_dir was correctly set in __init__
             current_script_path = os.path.dirname(os.path.abspath(__file__))
             self.temp_dir = os.path.join(current_script_path, f'{self.args.data}-temp') # Consistent with main.py
             print(f"  Fallback temp_dir for compute_sample_p: {self.temp_dir}")


        window = self.args.window
        walk_L = self.args.walk_L
        # self.node_n is already dl.nodes['count']
        total_triple_n = np.zeros((len(self.node_n), len(self.node_n)), dtype=np.float64)
        
        walk_file_path = os.path.join(self.temp_dir, "het_random_walk.txt")
        print(f"  Reading walks from: {walk_file_path}")
        if not os.path.exists(walk_file_path):
            print(f"  ERROR: Walk file {walk_file_path} not found for compute_sample_p. Generating it first.")
            self.gen_het_w_walk(walk_file_path) # Try to generate if missing
            if not os.path.exists(walk_file_path):
                 print(f"  FATAL ERROR: Still could not generate/find {walk_file_path}")
                 # Return a uniform probability or raise error
                 return np.ones((len(self.node_n), len(self.node_n)), dtype=np.float64) * 0.1 


        het_walk_f = open(walk_file_path, "r")
        # ... (rest of compute_sample_p remains the same)
        centerNode = ''
        neighNode = ''
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            for j in range(len(path)): # Iterate over actual path length
                centerNode = path[j]
                if not centerNode or len(centerNode) < 1 : continue # Skip if empty node string

                # Robustly get centerType
                centerType = -1
                center_node_type_prefix = ""
                for type_id, type_name in self.node_type2name.items():
                    if centerNode.startswith(type_name):
                        # Ensure that the part after prefix is numeric if it's not just the prefix
                        potential_id_part = centerNode[len(type_name):]
                        if not potential_id_part or potential_id_part.isdigit():
                             centerType = type_id
                             center_node_type_prefix = type_name
                             break
                if centerType == -1:
                    # print(f"    Warning: Could not parse type for centerNode '{centerNode}' in compute_sample_p. Skipping.")
                    continue
                
                # Validate if centerNode has a valid local ID part
                center_local_id_str = centerNode.replace(center_node_type_prefix, "", 1)
                if not center_local_id_str.isdigit() and center_local_id_str != "": # Allow for just type prefix if ID is 0 and not explicit
                     # print(f"    Warning: centerNode '{centerNode}' has non-digit ID part '{center_local_id_str}'. Skipping.")
                     continue


                for k in range(max(0, j - window), min(len(path), j + window + 1)):
                    if k == j: continue
                    
                    neighNode = path[k]
                    if not neighNode or len(neighNode) < 1: continue

                    neighType = -1
                    neigh_node_type_prefix = ""
                    for type_id, type_name in self.node_type2name.items():
                        if neighNode.startswith(type_name):
                            potential_id_part = neighNode[len(type_name):]
                            if not potential_id_part or potential_id_part.isdigit():
                                neighType = type_id
                                neigh_node_type_prefix = type_name
                                break
                    if neighType == -1:
                        # print(f"    Warning: Could not parse type for neighNode '{neighNode}' in compute_sample_p. Skipping.")
                        continue
                    
                    neigh_local_id_str = neighNode.replace(neigh_node_type_prefix, "", 1)
                    if not neigh_local_id_str.isdigit() and neigh_local_id_str != "":
                         # print(f"    Warning: neighNode '{neighNode}' has non-digit ID part '{neigh_local_id_str}'. Skipping.")
                         continue

                    if 0 <= centerType < total_triple_n.shape[0] and 0 <= neighType < total_triple_n.shape[1]:
                         total_triple_n[centerType][neighType] += 1
                    # else:
                         # print(f"    Warning: centerType {centerType} or neighType {neighType} out of bounds for total_triple_n. Shapes: {total_triple_n.shape}")


        het_walk_f.close()
        
        # Avoid division by zero if some type pairs have no co-occurrences
        total_triple_n[total_triple_n == 0] = 1e-9 # Replace 0s with a small number
        
        sample_p = self.args.batch_s / (total_triple_n * window * 2) # Corrected denominator
        sample_p = np.clip(sample_p, 0.0, 1.0) # Ensure probabilities are [0,1]

        print("Info: compute sampling ratio for each kind of triple done.")
        return sample_p

    def gen_embeds_w_neigh(self):
        # This function now only needs to compute the sampling probabilities.
        # The feature list/embedding layer is now handled directly inside the model.
        self.triple_sample_p = self.compute_sample_p()
        print("Info: Triple sample probabilities computed. Feature embeddings will be handled by the model.")


    def sample_het_walk_triple(self):
        print("Info: sampling triple relations start.")
        triple_list = defaultdict(list) # Key: (centerType_int, neighType_int)
        window = self.args.window
        # walk_L = self.args.walk_L # Not directly used here, path length is from file
        
        # Ensure self.temp_dir is set
        if not hasattr(self, 'temp_dir') or not self.temp_dir:
             current_script_path = os.path.dirname(os.path.abspath(__file__))
             self.temp_dir = os.path.join(current_script_path, f'{self.args.data}-temp')
             print(f"  Fallback temp_dir for sample_het_walk_triple: {self.temp_dir}")

        walk_file_path = os.path.join(self.temp_dir, "het_random_walk.txt")
        print(f"  Reading walks from: {walk_file_path} for triple sampling.")
        if not os.path.exists(walk_file_path):
            print(f"  ERROR: Walk file {walk_file_path} not found for sample_het_walk_triple. Cannot sample.")
            return triple_list # Return empty

        het_walk_f = open(walk_file_path, "r")
        for line in het_walk_f:
            line = line.strip()
            path = re.split(' ', line)
            if len(path) < 2: continue # Need at least two nodes for a pair

            for j_center_idx in range(len(path)):
                centerNode_str = path[j_center_idx]
                if not centerNode_str or len(centerNode_str) < 1: continue

                # Robustly parse centerNode
                centerType = -1
                center_node_type_prefix = ""
                for type_id, type_name in self.node_type2name.items():
                    if centerNode_str.startswith(type_name):
                        potential_id_part = centerNode_str[len(type_name):]
                        if not potential_id_part or potential_id_part.isdigit():
                            centerType = type_id
                            center_node_type_prefix = type_name
                            break
                if centerType == -1: continue
                
                center_local_id_str = centerNode_str.replace(center_node_type_prefix, "", 1)
                if not center_local_id_str.isdigit(): continue
                center_local_id = int(center_local_id_str)
                if centerType not in self.node_n or center_local_id >= self.node_n[centerType]: continue # Invalid center node

                for k_context_idx in range(max(0, j_center_idx - window), min(len(path), j_center_idx + window + 1)):
                    if k_context_idx == j_center_idx: continue
                    
                    neighNode_str = path[k_context_idx]
                    if not neighNode_str or len(neighNode_str) < 1: continue

                    # Robustly parse neighNode
                    neighType = -1
                    neigh_node_type_prefix = ""
                    for type_id, type_name in self.node_type2name.items():
                        if neighNode_str.startswith(type_name):
                            potential_id_part = neighNode_str[len(type_name):]
                            if not potential_id_part or potential_id_part.isdigit():
                                neighType = type_id
                                neigh_node_type_prefix = type_name
                                break
                    if neighType == -1: continue

                    neigh_local_id_str = neighNode_str.replace(neigh_node_type_prefix, "", 1)
                    if not neigh_local_id_str.isdigit(): continue
                    neigh_local_id = int(neigh_local_id_str)
                    if neighType not in self.node_n or neigh_local_id >= self.node_n[neighType]: continue # Invalid neighbor node

                    # Check sampling probability
                    # self.triple_sample_p is indexed by integers
                    if not (0 <= centerType < self.triple_sample_p.shape[0] and \
                            0 <= neighType < self.triple_sample_p.shape[1]):
                        # print(f"    Warning: centerType {centerType} or neighType {neighType} out of bounds for triple_sample_p. Using default 0.1")
                        current_sample_p = 0.1 
                    else:
                        current_sample_p = self.triple_sample_p[centerType][neighType]

                    if random.random() < current_sample_p:
                        # Sample negative node of the same type as neighNode
                        if self.node_n[neighType] == 0: continue # Cannot sample negative if no nodes of this type

                        negNode_local_id = random.randint(0, self.node_n[neighType] - 1)
                        # Ensure negative node has some connectivity (original logic)
                        # However, self.neigh_list is indexed by type, then local_id, then it's a list of neighbor strings
                        # The condition len(self.neigh_list[neighType][negNode_local_id]) == 0 makes sense
                        # Need to ensure neighType and negNode_local_id are valid for self.neigh_list
                        if neighType not in self.neigh_list or \
                           negNode_local_id >= len(self.neigh_list[neighType]):
                            # This situation should be rare if node_n is consistent
                            # Default to a random node again or skip
                            negNode_local_id = random.randint(0, self.node_n[neighType] - 1)
                            # If still problematic, this type might have no entries in self.neigh_list, so skip
                            if neighType not in self.neigh_list or negNode_local_id >= len(self.neigh_list[neighType]):
                                continue


                        # Max attempts to find a connected negative node
                        max_neg_sample_attempts = 10 
                        current_attempts = 0
                        while len(self.neigh_list[neighType][negNode_local_id]) == 0 and current_attempts < max_neg_sample_attempts:
                            negNode_local_id = random.randint(0, self.node_n[neighType] - 1)
                            current_attempts += 1
                        
                        # If still couldn't find a connected one, might just use the last random one or skip
                        if len(self.neigh_list[neighType][negNode_local_id]) == 0 and current_attempts == max_neg_sample_attempts:
                            # print(f"    Warning: Could not find a connected negative sample for type {neighType} after {max_neg_sample_attempts} attempts. Using last random or skipping.")
                            # Optionally skip if no connected negative node is found
                            continue 
                            
                        triple = [center_local_id, neigh_local_id, negNode_local_id]
                        triple_list[(centerType, neighType)].append(triple)
        het_walk_f.close()
        print("Info: sampling triple relations done.")
        return triple_list