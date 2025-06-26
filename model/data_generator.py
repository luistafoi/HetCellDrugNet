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
        # Assuming main.py defines temp_dir based on its location and args.data
        # For consistency, let's use the path construction from main.py if needed
        # current_script_dir_main = os.path.dirname(os.path.abspath(sys.argv[0])) # Path of main.py
        # self.temp_dir = os.path.join(current_script_dir_main, f'{self.args.data}-temp')
        # However, HetGNN often creates temp relative to its own script if sys.path[0] is used.
        # For now, let's assume temp_dir is correctly handled for file writing/reading later
        # We will use self.dl.path for reading original data files.

        print("--- Inside data_generator.input_data.__init__ ---")
        print(f"  args.data: {args.data}")
        print(f"  dl.path (HGB dataset path): {self.dl.path}")
        
        node_n, node_shift = self.dl.nodes['count'], self.dl.nodes['shift']
        self.node_n, self.node_shift = node_n, node_shift
        print(f"  Node counts per type (self.node_n): {self.node_n}")
        print(f"  Node ID shifts per type (self.node_shift): {self.node_shift}")

        # Load info.dat to get node type name mapping
        # Path to info.dat should be self.dl.path + 'info.dat'
        info_dat_abs_path = os.path.join(self.dl.path, 'info.dat')
        print(f"  Attempting to load info.dat from: {info_dat_abs_path}")
        try:
            with open(info_dat_abs_path, 'r', encoding='utf-8') as f:
                data_info_content = json.load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load or parse {info_dat_abs_path}: {e}")
            sys.exit(1)

        self.node_type2name, self.node_name2type = dict(), dict()
        
        # Adapt to potential HGB info.dat structure vs simpler structure
        if "node.dat" in data_info_content and "node type" in data_info_content["node.dat"]: # Strict HGB format
            node_type_mapping_from_info = data_info_content["node.dat"]["node type"]
            for type_id_str, type_info_list in node_type_mapping_from_info.items():
                type_id_int = int(type_id_str)
                type_name = type_info_list[0]
                self.node_type2name[type_id_int] = type_name
                self.node_name2type[type_name] = type_id_int
        elif "node.dat" in data_info_content and isinstance(data_info_content["node.dat"], dict): 
            node_type_mapping_from_info = data_info_content["node.dat"]
            for type_id_str, type_info_list_value in node_type_mapping_from_info.items(): # Renamed for clarity
                type_id_int = int(type_id_str)
                
                # Extract the actual string name from the list
                actual_type_name_string = type_info_list_value[0] 
                
                self.node_type2name[type_id_int] = actual_type_name_string # Store the string name
                self.node_name2type[actual_type_name_string] = type_id_int # Use the string name as the key
        else:
            print(f"FATAL ERROR: info.dat at {info_dat_abs_path} does not have the expected 'node.dat' structure.")
            sys.exit(1)
            
        print(f"  self.node_type2name map: {self.node_type2name}")
        print(f"  self.node_name2type map: {self.node_name2type}")

        node_types = list(self.node_type2name.keys()) # These are now integer type IDs

        self.while_max_count = 1e5
        if args.data=='amazon':
            self.standand_node_L = [100]
            self.top_k = {0:10} # Assuming type 0 for Amazon
            self.neigh_L = 100
        elif args.data=='LastFM' or args.data=='LastFM_magnn':
            self.standand_node_L = [20,70,10] # For 3 types
            self.top_k = {0:10,1:10,2:5}
            self.neigh_L = 80
        elif args.data=='PubMed':
            self.standand_node_L = [20,30,40,10] # For 4 types
            self.top_k = {0:10,1:10,2:10,3:5}
            self.neigh_L = 70
            self.while_max_count = 1e3
        elif args.data=='CellDrug':
            # Assuming 3 node types: 0:gene, 1:cell, 2:drug
            # Ensure these type IDs match what's in your info.dat and node_type2name map
            if not all(k in self.node_type2name for k in [0,1,2]):
                 print(f"FATAL: 'CellDrug' config expects node types 0, 1, 2, but self.node_type2name is {self.node_type2name}")
                 sys.exit(1)
            if len(self.node_type2name) != 3:
                 print(f"FATAL: 'CellDrug' config expects 3 node types, but found {len(self.node_type2name)} from info.dat")
                 sys.exit(1)
            self.standand_node_L = [30, 40, 30]
            self.top_k = {0:10, 1:10, 2:10}
            self.neigh_L = 100 
            self.while_max_count = 1e5
            print(f"  Applied CellDrug specific HetGNN params: neigh_L={self.neigh_L}, standand_node_L={self.standand_node_L}, top_k={self.top_k}")
        else: 
            print(f"Warning: Using generic default HetGNN parameters for dataset: {args.data}")
            num_node_types_actual = len(self.dl.nodes['count'])
            self.standand_node_L = [20] * num_node_types_actual
            self.top_k = {i: 5 for i in range(num_node_types_actual)}
            self.neigh_L = sum(self.standand_node_L) if self.standand_node_L else 50
            print(f"  Fallback HetGNN params: neigh_L={self.neigh_L}, standand_node_L={self.standand_node_L}, top_k={self.top_k}")


        edge_list = dict()
        for edge_type in sorted(dl.links['meta'].keys()): # edge_type is integer relation ID
            h_type, t_type = dl.links['meta'][edge_type] # h_type, t_type are integer node type IDs
            h_node_count_for_type = dl.nodes['count'][h_type]
            edge_list[edge_type] = [[] for _ in range(h_node_count_for_type)]
        
        neigh_list = dict()
        for node_type_key in dl.nodes['count'].keys(): # node_type_key is integer node type ID
            count_for_this_type = dl.nodes['count'][node_type_key]
            neigh_list[node_type_key] = [[] for _ in range(count_for_this_type)]

        print("\n  Populating edge_list (neighbors for each edge type)...")
        for edge_type in sorted(dl.links['meta'].keys()):
            h_type, t_type = dl.links['meta'][edge_type]
            row, col = self.dl.links['data'][edge_type].nonzero() # global IDs
            
            for r_global, c_global in zip(row, col):
                h_id_local = r_global - node_shift[h_type]
                t_id_local = c_global - node_shift[t_type]

                # --- Extensive Debugging for edge_list population ---
                # print(f"\n    DEBUG POPULATING EDGE_LIST:")
                # print(f"      Edge Type: {edge_type}, HGB Relation Meta: {self.dl.links['meta'][edge_type]} (h_type={h_type}, t_type={t_type})")
                # print(f"      Global Link: {r_global} -> {c_global}")
                # print(f"      h_type {h_type}: Shift={node_shift[h_type]}, Count={node_n[h_type]}")
                # print(f"      t_type {t_type}: Shift={node_shift[t_type]}, Count={node_n[t_type]}")
                # print(f"      Calculated h_id_local: {h_id_local}")
                # print(f"      Calculated t_id_local: {t_id_local}")

                if not (0 <= h_id_local < node_n[h_type]):
                    print(f"      CRITICAL ERROR (h_id_local): Local head ID {h_id_local} for global {r_global} is out of bounds for type {h_type} (max local: {node_n[h_type]-1}). Skipping this edge.")
                    continue 
                if not (0 <= t_id_local < node_n[t_type]):
                    print(f"      CRITICAL ERROR (t_id_local): Local tail ID {t_id_local} for global {c_global} is out of bounds for type {t_type} (max local: {node_n[t_type]-1}). Skipping this edge.")
                    continue
                if t_type not in self.node_type2name:
                    print(f"      CRITICAL ERROR: Target node type ID {t_type} not in self.node_type2name map: {self.node_type2name.keys()}. Skipping this edge.")
                    continue
                
                node_name_prefix = self.node_type2name[t_type]
                neighbor_string_to_append = f"{node_name_prefix}{t_id_local}"
                # print(f"      Appending to edge_list[{edge_type}][{h_id_local}]: '{neighbor_string_to_append}'")
                # --- End Debugging for edge_list population ---
                
                edge_list[edge_type][h_id_local].append(neighbor_string_to_append)
        
        print("\n  Populating neigh_list (aggregated neighbors for each node)...")
        for node_type_key in dl.nodes['count'].keys(): # This is an integer node type
            for edge_type_key in edge_list.keys(): # This is an integer link type
                h_type, _ = dl.links['meta'][edge_type_key] # h_type is an integer node type
                if node_type_key == h_type:
                    # node_n[node_type_key] is count for this node_type_key
                    for n_id_local in range(node_n[node_type_key]): # n_id_local is local id for node_type_key
                        # edge_list[edge_type_key] is list for source nodes of h_type. Its length is node_n[h_type]
                        # So, n_id_local (which is for node_type_key == h_type) should be a valid index.
                        if n_id_local < len(edge_list[edge_type_key]): # Safety check
                             neigh_list[node_type_key][n_id_local] += edge_list[edge_type_key][n_id_local]
                        # else:
                        #    print(f"    Warning: n_id_local {n_id_local} out of bounds for edge_list[{edge_type_key}] (len {len(edge_list[edge_type_key])}) when populating neigh_list for node_type {node_type_key}")
                            
        self.edge_list = edge_list
        self.neigh_list = neigh_list
        
        # Initialize neigh_list_train to prevent AttributeError if it's not generated/loaded.
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