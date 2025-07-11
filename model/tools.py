# Final, "Best On Paper" version of model/tools.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from args import read_args

args = read_args()

class HetAgg(nn.Module):
    def __init__(self, args, dl, input_data, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.embed_d = args.embed_d
        self.n_layers = args.n_layers
        self.device = device
        self.dl = dl
        self.input_data = input_data

        self.node_types = sorted(self.dl.nodes['count'].keys())
        self.node_min_size = self.input_data.standand_node_L

        self.feat_proj = nn.ModuleDict()
        for n_type in self.node_types:
            num_nodes_of_type = self.dl.nodes['count'][n_type]
            self.feat_proj[str(n_type)] = nn.Embedding(num_nodes_of_type, self.embed_d).to(device)

        # The RNN aggregator is part of the intended architecture
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer_modules = nn.ModuleDict()
            for n_type in self.node_types:
                layer_modules[f'rnn_{n_type}'] = nn.RNN(self.embed_d, self.embed_d).to(device)
                layer_modules[f'sem_att_{n_type}'] = nn.Linear(self.embed_d * 2, 1, bias=False).to(device)
            self.gnn_layers.append(layer_modules)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.lp_bilinear = None
        self.drug_type_name = None
        self.cell_type_name = None

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, (nn.RNN, nn.Embedding, nn.Bilinear)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.1)

    def setup_link_prediction(self, drug_type_name: str, cell_type_name: str):
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        # self.lp_bilinear = nn.Bilinear(self.embed_d, self.embed_d, 1).to(self.device) (this is the original line)
        # The new dimension is doubled to account for the concatenation
        self.lp_bilinear = nn.Bilinear(self.embed_d * 2, self.embed_d * 2, 1).to(self.device) #for skip conncection
        print(f"INFO: Setup link prediction head for drug ('{drug_type_name}') and cell ('{cell_type_name}').")

    #this function is used to get both the initial and final embeddings for skip connection
    def get_combined_embedding(self, id_batch_local, node_type):
            """
            Gets both the initial and final (post-GNN) embeddings and concatenates them.
            """
            # 1. Get initial embeddings (before message passing)
            initial_embeds = self.conteng_agg(id_batch_local, node_type)

            # 2. Get final embeddings (after message passing)
            final_embeds = self.node_het_agg(id_batch_local, node_type)

            # 3. Concatenate them side-by-side
            return torch.cat([initial_embeds, final_embeds], dim=1)

    # def link_prediction_loss(self, drug_indices_global, cell_indices_global, labels, isolation_ratio=0.0):
    #     """
    #     Calculates link prediction loss. A fraction of cells can be "isolated"
    #     to have their embeddings generated without message passing.
    #     """
    #     drug_type_id = self.input_data.node_name2type[self.drug_type_name]
    #     cell_type_id = self.input_data.node_name2type[self.cell_type_name]

    #     # --- Get drug embeddings using full GNN message passing ---
    #     drug_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in drug_indices_global.tolist()]
    #     drug_embeds = self.node_het_agg(drug_indices_local, drug_type_id)

    #     # --- Get cell embeddings with the new isolation logic ---
    #     cell_indices_local = torch.tensor(
    #         [self.dl.nodes['type_map'][g_idx][1] for g_idx in cell_indices_global.tolist()],
    #         device=self.device
    #     )
        
    #     batch_size = len(cell_indices_local)
    #     final_cell_embeds = torch.zeros(batch_size, self.embed_d, device=self.device)

    #     # Create a random mask to decide which cells to isolate
    #     should_isolate = torch.rand(batch_size, device=self.device) < isolation_ratio
        
    #     # --- NEW: Print statement to verify the isolation ---
    #     num_isolated = should_isolate.sum().item()
    #     if num_isolated > 0:
    #         print(f"    > Isolating {num_isolated} / {batch_size} cells for this batch ({num_isolated/batch_size*100:.1f}%)")
        
    #     # --- A. For cells that ARE connected to the graph (most of them) ---
    #     graph_connected_mask = ~should_isolate
    #     if graph_connected_mask.any():
    #         graph_cell_indices = cell_indices_local[graph_connected_mask].tolist()
    #         graph_cell_embeds = self.node_het_agg(graph_cell_indices, cell_type_id)
    #         final_cell_embeds[graph_connected_mask] = graph_cell_embeds

    #     # --- B. For cells that are "isolated" (a small fraction) ---
    #     if should_isolate.any():
    #         isolated_cell_indices = cell_indices_local[should_isolate].tolist()
    #         isolated_cell_embeds = self.conteng_agg(isolated_cell_indices, cell_type_id)
    #         final_cell_embeds[should_isolate] = isolated_cell_embeds
            
    #     # --- Calculate loss as before, using the combined embeddings ---
    #     scores = self.lp_bilinear(drug_embeds, final_cell_embeds).squeeze(-1)
    #     return F.binary_cross_entropy_with_logits(scores, labels.float())

    # def link_prediction_loss(self, drug_indices_global, cell_indices_global, labels): #This is the original function no isolation no nothing
    #     """
    #     Calculates link prediction loss using the standard GNN embeddings.
    #     """
    #     drug_type_id = self.input_data.node_name2type[self.drug_type_name]
    #     cell_type_id = self.input_data.node_name2type[self.cell_type_name]

    #     # Get local indices for both drugs and cells
    #     drug_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in drug_indices_global.tolist()]
    #     cell_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in cell_indices_global.tolist()]

    #     # Get final embeddings for both using the full GNN message passing
    #     drug_embeds = self.node_het_agg(drug_indices_local, drug_type_id)
    #     cell_embeds = self.node_het_agg(cell_indices_local, cell_type_id)

    #     # Calculate loss using the bilinear layer
    #     scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
    #     return F.binary_cross_entropy_with_logits(scores, labels.float())

    def link_prediction_loss(self, drug_indices_global, cell_indices_global, labels):
            drug_type_id = self.input_data.node_name2type[self.drug_type_name]
            cell_type_id = self.input_data.node_name2type[self.cell_type_name]

            drug_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in drug_indices_global.tolist()]
            cell_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in cell_indices_global.tolist()]

            # Use the new helper to get combined embeddings
            drug_embeds_combined = self.get_combined_embedding(drug_indices_local, drug_type_id)
            cell_embeds_combined = self.get_combined_embedding(cell_indices_local, cell_type_id)
            
            scores = self.lp_bilinear(drug_embeds_combined, cell_embeds_combined).squeeze(-1)
            return F.binary_cross_entropy_with_logits(scores, labels.float())

    def conteng_agg(self, local_id_batch, node_type):
        if not local_id_batch:
            return torch.empty(0, self.embed_d).to(self.device)
        local_indices_tensor = torch.LongTensor(local_id_batch).to(self.device)
        return self.feat_proj[str(node_type)](local_indices_tensor)


# In model/tools.py, replace the entire old `node_het_agg` function with this one.
    def node_het_agg(self, id_batch_local, node_type):
        """
        Aggregates features for a batch of nodes of a specific type.
        This version is robust to unseen nodes and uses correct ID mapping.
        """
        # Get the initial embeddings for the nodes in the current batch.
        current_embeds = self.conteng_agg(id_batch_local, node_type)
        current_batch_size = len(id_batch_local)

        # Process through each GNN layer
        for l in range(self.n_layers):
            # This will hold the aggregated embeddings from each neighbor type
            agg_batch = {}

            # For each neighbor type, gather and process the neighbors
            for nt_id in self.node_types:
                
                # 1. For each node in our batch, find its neighbors of type `nt_id`.
                #    This list will hold the padded/sampled neighbor lists for the entire batch.
                processed_neighbor_lists = []
                for node_local_id in id_batch_local:
                    
                    # This is the critical safety check for test/validation nodes.
                    # If a node's ID is outside the range of the pre-computed neighbor list,
                    # treat it as having no neighbors.
                    neighbors_str = []
                    if node_local_id < len(self.input_data.neigh_list_train[node_type]):
                        neighbors_str = self.input_data.neigh_list_train[node_type][node_local_id]

                    # From the string neighbors, parse the local IDs of the correct type
                    neighbors_local_ids = []
                    neighbor_type_name_to_match = self.input_data.node_type2name[nt_id]
                    for neigh_str in neighbors_str:
                        if neigh_str.startswith(neighbor_type_name_to_match):
                            try:
                                # This correctly parses the local ID from strings like "gene123"
                                neighbors_local_ids.append(int(neigh_str[len(neighbor_type_name_to_match):]))
                            except (ValueError, IndexError):
                                continue
                    
                    # 2. Pad or sample the neighbor list to a fixed size.
                    min_size = self.node_min_size[nt_id]
                    if len(neighbors_local_ids) < min_size:
                        num_nodes_of_neighbor_type = self.dl.nodes['count'][nt_id]
                        if num_nodes_of_neighbor_type > 0:
                            num_to_pad = min_size - len(neighbors_local_ids)
                            padding = [random.randint(0, num_nodes_of_neighbor_type - 1) for _ in range(num_to_pad)]
                            neighbors_local_ids.extend(padding)
                    elif len(neighbors_local_ids) > min_size:
                        neighbors_local_ids = random.sample(neighbors_local_ids, min_size)
                    
                    processed_neighbor_lists.append(neighbors_local_ids)

                # 3. Aggregate the features of all neighbors in the batch.
                flat_neighs_local = [item for sublist in processed_neighbor_lists for item in sublist]

                if not flat_neighs_local:
                    agg_batch[nt_id] = torch.zeros(current_batch_size, self.embed_d, device=self.device)
                else:
                    neigh_embeds = self.conteng_agg(flat_neighs_local, nt_id)
                    
                    # Reshape for RNN processing
                    neigh_embeds = neigh_embeds.view(current_batch_size, self.node_min_size[nt_id], self.embed_d)
                    neigh_embeds = neigh_embeds.permute(1, 0, 2)
                    
                    rnn = self.gnn_layers[l][f'rnn_{nt_id}']
                    all_states, _ = rnn(neigh_embeds)
                    
                    # Average over the neighbors to get a single vector per node in the batch
                    agg_batch[nt_id] = torch.mean(all_states, 0)

            # 4. Use semantic attention to combine aggregated neighbor embeddings.
            c_agg_batch_prev_layer = current_embeds
            concat_embeds_list = [torch.cat((c_agg_batch_prev_layer, c_agg_batch_prev_layer), 1)]
            for nt_id in self.node_types:
                concat_embeds_list.append(torch.cat((c_agg_batch_prev_layer, agg_batch[nt_id]), 1))
            
            concat_embeds = torch.stack(concat_embeds_list, dim=1)
            sem_att = self.gnn_layers[l][f'sem_att_{node_type}']
            atten_w = sem_att(concat_embeds).squeeze(-1)
            atten_w = self.softmax(atten_w).unsqueeze(-1)
            
            embeds_to_combine = [c_agg_batch_prev_layer] + [agg_batch[nt_id] for nt_id in self.node_types]
            embeds_to_combine = torch.stack(embeds_to_combine, dim=1)
            
            # Update the embeddings for the next layer
            current_embeds = torch.sum(atten_w * embeds_to_combine, dim=1)
            current_embeds = self.act(current_embeds)
            
        return current_embeds
    
    def het_agg(self, triple_pair, c_id_batch, pos_id_batch, neg_id_batch):
        c_agg = self.node_het_agg(c_id_batch, triple_pair[0])
        p_agg = self.node_het_agg(pos_id_batch, triple_pair[1])
        n_agg = self.node_het_agg(neg_id_batch, triple_pair[1])
        return c_agg, p_agg, n_agg

    def aggregate_all(self, triple_list_batch, triple_pair):
        c_id_batch = [x[0] for x in triple_list_batch]
        pos_id_batch = [x[1] for x in triple_list_batch]
        neg_id_batch = [x[2] for x in triple_list_batch]
        return self.het_agg(triple_pair, c_id_batch, pos_id_batch, neg_id_batch)

    def forward(self, triple_list_batch, triple_pair):
        return self.aggregate_all(triple_list_batch, triple_pair)

    # def link_prediction_forward(self, drug_indices_global, cell_indices_global): #this is the original function
    #     if self.lp_bilinear is None:
    #         raise RuntimeError("Link prediction layers not initialized.")
    #     drug_type_id = self.input_data.node_name2type[self.drug_type_name]
    #     cell_type_id = self.input_data.node_name2type[self.cell_type_name]
    #     drug_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in drug_indices_global.tolist()]
    #     cell_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in cell_indices_global.tolist()]
    #     drug_embeds = self.node_het_agg(drug_indices_local, drug_type_id)
    #     cell_embeds = self.node_het_agg(cell_indices_local, cell_type_id)
    #     scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
    #     return torch.sigmoid(scores)

    def link_prediction_forward(self, drug_indices_global, cell_indices_global):
            if self.lp_bilinear is None:
                raise RuntimeError("Link prediction layers not initialized.")
                
            drug_type_id = self.input_data.node_name2type[self.drug_type_name]
            cell_type_id = self.input_data.node_name2type[self.cell_type_name]
            
            drug_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in drug_indices_global.tolist()]
            cell_indices_local = [self.dl.nodes['type_map'][g_idx][1] for g_idx in cell_indices_global.tolist()]

            # Use the new helper to get combined embeddings
            drug_embeds_combined = self.get_combined_embedding(drug_indices_local, drug_type_id)
            cell_embeds_combined = self.get_combined_embedding(cell_indices_local, cell_type_id)
            
            scores = self.lp_bilinear(drug_embeds_combined, cell_embeds_combined).squeeze(-1)
            return torch.sigmoid(scores)

    def get_final_embeddings(self):
        final_embeds = {}
        all_nodes = {nt: list(range(count)) for nt, count in self.dl.nodes['count'].items()}
        for node_type in self.node_types:
            embeds = []
            node_ids_local = all_nodes[node_type]
            batch_number = math.ceil(len(node_ids_local) / self.args.mini_batch_s)
            for j in range(batch_number):
                id_batch_local = node_ids_local[j * self.args.mini_batch_s: (j + 1) * self.args.mini_batch_s]
                if not id_batch_local:
                    continue
                out_temp = self.node_het_agg(id_batch_local, node_type=node_type)
                embeds.append(out_temp.cpu().detach())
            if embeds:
                final_embeds[self.input_data.node_type2name[node_type]] = torch.cat(embeds, dim=0)
        return final_embeds

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, n_tasks):
        super(MultiTaskLossWrapper, self).__init__()
        self.n_tasks = n_tasks
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

def cross_entropy_loss(c_embed, p_embed, n_embed):
    p_score = torch.sum(torch.mul(c_embed, p_embed), dim=1)
    n_score = torch.sum(torch.mul(c_embed, n_embed), dim=1)
    return -torch.mean(F.logsigmoid(p_score) + F.logsigmoid(-n_score))