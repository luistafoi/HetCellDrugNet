import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from args import read_args

args = read_args()


class HetAgg(nn.Module):
    def __init__(self, args, feature_list, neigh_list_train, dl, input_data, device):
        super(HetAgg, self).__init__()
        self.args = args
        self.embed_d = args.embed_d
        self.n_layers = args.n_layers
        self.device = device
        self.dl = dl
        self.input_data = input_data
        self.feature_list = {int(k): v.to(device) for k, v in feature_list.items()}
        self.neigh_list_train = neigh_list_train

        self.node_types = sorted(self.dl.nodes['count'].keys())
        self.node_min_size = self.input_data.standand_node_L

        # Feature projection layer (used before the first GNN layer)
        self.feat_proj = nn.ModuleDict()
        for n_type in self.node_types:
            feat_dim = self.feature_list[n_type].shape[1]
            self.feat_proj[str(n_type)] = nn.Linear(feat_dim, self.embed_d).to(device)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer_modules = nn.ModuleDict()
            for n_type in self.node_types:
                # Node-level aggregation (RNN for neighbors of a specific type)
                layer_modules[f'rnn_{n_type}'] = nn.RNN(self.embed_d, self.embed_d).to(device)
                # Semantic-level aggregation (attention over different neighbor types)
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
            elif isinstance(m, nn.RNN):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.1)

    def setup_link_prediction(self, drug_type_name: str, cell_type_name: str):
        self.drug_type_name = drug_type_name
        self.cell_type_name = cell_type_name
        self.lp_bilinear = nn.Bilinear(self.embed_d, self.embed_d, 1).to(self.device)
        print(f"INFO: Setup link prediction head for drug ('{drug_type_name}') and cell ('{cell_type_name}').")

    def link_prediction_loss(self, drug_indices_global, cell_indices_global, labels):
        if self.lp_bilinear is None:
            raise RuntimeError("Link prediction layers are not initialized. Call model.setup_link_prediction() first.")
        
        drug_type_id = self.input_data.node_name2type[self.drug_type_name]
        cell_type_id = self.input_data.node_name2type[self.cell_type_name]

        drug_embeds = self.node_het_agg(drug_indices_global, drug_type_id)
        cell_embeds = self.node_het_agg(cell_indices_global, cell_type_id)

        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return F.binary_cross_entropy_with_logits(scores, labels.float())

    def conteng_agg(self, id_batch, node_type):
        if not id_batch:
            return torch.empty(0, self.embed_d).to(self.device)
        local_indices = [self.dl.nodes['type_map'][g_idx][1] for g_idx in id_batch]
        local_indices = torch.LongTensor(local_indices).to(self.device)
        initial_features = self.feature_list[node_type][local_indices]
        return self.feat_proj[str(node_type)](initial_features)

    def node_het_agg(self, id_batch, node_type):
        # Get initial embeddings from features
        current_embeds = self.conteng_agg(id_batch, node_type)

        for l in range(self.n_layers):
            # Gather neighbors for the current batch
            neigh_by_type = {nt: [] for nt in self.node_types}
            for g_idx in id_batch:
                local_id = self.dl.nodes['type_map'][g_idx][1]
                neighbors_str = self.neigh_list_train[node_type][local_id]
                
                node_neigh_by_type = {nt: [] for nt in self.node_types}
                for neigh_str in neighbors_str:
                    for nt_id, nt_name in self.input_data.node_type2name.items():
                        if neigh_str.startswith(nt_name):
                            n_local_id = int(neigh_str[len(nt_name):])
                            n_global_id = n_local_id + self.dl.nodes['shift'][nt_id]
                            node_neigh_by_type[nt_id].append(n_global_id)
                            break
                
                for nt_id in self.node_types:
                    neighs = node_neigh_by_type[nt_id]
                    min_size = self.node_min_size[nt_id]
                    if len(neighs) > min_size:
                        neighs = random.sample(neighs, min_size)
                    else:
                        neighs.extend([g_idx] * (min_size - len(neighs))) # Pad with self
                    neigh_by_type[nt_id].append(neighs)

            # Aggregate neighbor embeddings (based on their initial features)
            agg_batch = {}
            for nt_id in self.node_types:
                flat_neighs = [item for sublist in neigh_by_type[nt_id] for item in sublist]
                if not flat_neighs:
                    agg_batch[nt_id] = torch.zeros(len(id_batch), self.embed_d).to(self.device)
                    continue
                
                # Always use initial features for neighbors
                neigh_embeds = self.conteng_agg(flat_neighs, nt_id)
                neigh_embeds = neigh_embeds.view(len(id_batch), self.node_min_size[nt_id], self.embed_d)
                neigh_embeds = neigh_embeds.permute(1, 0, 2)
                
                # Use the RNN from the current GNN layer
                rnn = self.gnn_layers[l][f'rnn_{nt_id}']
                all_states, _ = rnn(neigh_embeds)
                agg_batch[nt_id] = torch.mean(all_states, 0)

            # Semantic aggregation
            c_agg_batch_prev_layer = current_embeds
            
            concat_embeds_list = [torch.cat((c_agg_batch_prev_layer, c_agg_batch_prev_layer), 1)]
            for nt_id in self.node_types:
                concat_embeds_list.append(torch.cat((c_agg_batch_prev_layer, agg_batch[nt_id]), 1))
            
            concat_embeds = torch.stack(concat_embeds_list, dim=1)
            
            # Use the attention from the current GNN layer
            sem_att = self.gnn_layers[l][f'sem_att_{node_type}']
            atten_w = sem_att(concat_embeds).squeeze(-1)
            atten_w = self.softmax(atten_w).unsqueeze(-1)
            
            embeds_to_combine = [c_agg_batch_prev_layer] + [agg_batch[nt_id] for nt_id in self.node_types]
            embeds_to_combine = torch.stack(embeds_to_combine, dim=1)
            
            # Update the embeddings for the next layer
            current_embeds = torch.sum(atten_w * embeds_to_combine, dim=1)
            # Apply activation function
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

    def link_prediction_forward(self, drug_indices_global, cell_indices_global):
        """
        Scores a batch of drug-cell links for evaluation. Returns sigmoid probabilities.
        """
        if self.lp_bilinear is None:
            raise RuntimeError("Link prediction layers are not initialized. Call model.setup_link_prediction() first.")
        
        drug_type_id = self.input_data.node_name2type[self.drug_type_name]
        cell_type_id = self.input_data.node_name2type[self.cell_type_name]

        drug_embeds = self.node_het_agg(drug_indices_global, drug_type_id)
        cell_embeds = self.node_het_agg(cell_indices_global, cell_type_id)

        scores = self.lp_bilinear(drug_embeds, cell_embeds).squeeze(-1)
        return torch.sigmoid(scores)

    def get_final_embeddings(self):
        final_embeds = {}
        all_nodes = {nt: list(range(count)) for nt, count in self.dl.nodes['count'].items()}

        for node_type in self.node_types:
            embeds = []
            node_ids_local = all_nodes[node_type]
            node_ids_global = [self.dl.nodes['shift'][node_type] + i for i in node_ids_local]
            
            batch_number = math.ceil(len(node_ids_global) / self.args.mini_batch_s)
            for j in range(batch_number):
                id_batch = node_ids_global[j * self.args.mini_batch_s: (j + 1) * self.args.mini_batch_s]
                if not id_batch:
                    continue
                out_temp = self.node_het_agg(id_batch=id_batch, node_type=node_type)
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

def cross_entropy_loss(c_embed, p_embed, n_embed, embed_d):
    p_score = torch.sum(torch.mul(c_embed, p_embed), dim=1)
    n_score = torch.sum(torch.mul(c_embed, n_embed), dim=1)
    return -torch.mean(F.logsigmoid(p_score) + F.logsigmoid(-n_score))
