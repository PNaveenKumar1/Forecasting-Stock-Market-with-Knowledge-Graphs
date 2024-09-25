import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F

from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge, HypergraphConv, RGATConv, HEATConv
from torch_geometric.nn import global_mean_pool

from torchkge.models.translation import TorusEModel
from torchkge.models.bilinear import ComplExModel, HolEModel
from .tkge_models import *
from torch_geometric.nn import global_mean_pool
import torch_geometric.nn as gnn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # TODO: Check div term, correct or wrong
        div_term = torch.exp(torch.arange(0, 2*d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)[:, 0::2]
        #print(pe.shape, position.shape, div_term.shape)
        pe[:, 1::2] = torch.cos(position * div_term)[:, 1::2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
        # return x

# ----------- Model -----------
class Transformer_Ranking(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True, USE_KG = True, NUM_NODES = 87, config=None, ENCODER_LAYER = 'lstm', USE_RELATION_KG = False):
        super().__init__()

        SEC_EMB, n = 25, 0 # 1 For LSTM Embedding
        if USE_GRAPH:
            if HYPER_GRAPH:
                n += 0
            n += 2
        if USE_KG:
            n += 2
        if USE_RELATION_KG:
            n += 1
        

        config['embedding_size'] = SEC_EMB*2

        self.embeddings = nn.Embedding(105, 10)

        #self.pos_enc_x = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=W)
        #self.pos_enc_y = PositionalEncoding(d_model=D_MODEL, dropout=DROPOUT, max_len=T)

        self.encoder_architecture = ENCODER_LAYER
        if self.encoder_architecture == 'transf':
            encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_FF, batch_first=True )
            self.transformer_encoder_first = nn.TransformerEncoder(encoder_layer, num_layers=ENC_LAYERS)
        elif self.encoder_architecture == 'lstm':
            self.lstm_encoder = nn.LSTM(input_size = 5, hidden_size = D_MODEL, num_layers = ENC_LAYERS, batch_first = True, bidirectional = False)
        elif self.encoder_architecture == 'gru':
            self.gru_encoder = nn.GRU(input_size = 5, hidden_size = D_MODEL, num_layers = ENC_LAYERS, batch_first = True, bidirectional = False)
        else:
            raise NotImplementedError("Encoder Architecture not implemented. Choose between [transf, lstm, gru]")
        
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.pred = nn.Linear((D_MODEL+(SEC_EMB*n))*NUM_NODES, NUM_NODES)
        self.pred2 = nn.Linear(NUM_NODES*10, NUM_NODES)

        self.hold_pred = nn.Linear(D_MODEL+(SEC_EMB*n), 1)

        self.is_pos = USE_POS_ENCODING
        self.time_steps = T

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:# == True:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(32, SEC_EMB*2),
                    ])
            elif self.use_graph == 'gcn':
                self.graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(16, 64), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(64, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, SEC_EMB*2),
                        ])
        self.use_relation_graph = USE_RELATION_KG
        if self.use_relation_graph == 'gcn':
            self.rel_node_emb = nn.Embedding(5000, 8)
            self.rel_graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(8, 32), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(32, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            nn.Linear(64, SEC_EMB),
                        ])
        elif self.use_relation_graph == 'hypergraph' or self.use_relation_graph == 'with_sector':
            self.rel_node_emb = nn.Embedding(5000, 8)
            self.rel_graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(32, SEC_EMB),
                    ])

        if self.use_graph == 'rgat':
            self.conv1 = RGATConv(20, 32, config['relation_total'], edge_dim=16, heads=1, dim=1)#, mod="additive")
            #self.global_pool = aggr.MaxAggregation()
            self.conv2 = RGATConv(32, 64, config['relation_total'], edge_dim=16, heads=1, dim=1)#, mod="additive")
            #self.global_pool2 = aggr.MaxAggregation()
            #self.conv3 = RGATConv(64, 64, config['relation_total'], edge_dim=16, heads=1, dim=1)
            self.lin = nn.Linear(64, SEC_EMB*2)
            self.lin2 = nn.Linear(SEC_EMB*2, 16)

            self.rel_type_embeddings = nn.Embedding(config['relation_total']+2, 16)
            #self.year_embeddings    = nn.Embedding(24, 16, padding_idx=0)
            self.month_embeddings   = nn.Embedding(13, 16, padding_idx=0)
            self.day_embeddings     = nn.Embedding(32, 16, padding_idx=0)
            self.hour_embeddings    = nn.Embedding(25, 16, padding_idx=0)
            self.minutes_embeddings = nn.Embedding(61, 16, padding_idx=0)
            self.sec_embeddings     = nn.Embedding(61, 16, padding_idx=0)
        if self.use_graph == 'hgat':
            self.node_type = config['node_type']
            self.conv1 = HEATConv(20, 8, config['num_node_type'], config['relation_total'], 16, 16, SEC_EMB*2, 4, dropout=0.1)
            #self.global_pool = aggr.MaxAggregation()
            #self.bn1 = gnn.norm.BatchNorm(32)
            self.conv2 = HEATConv(8*4, 16, config['num_node_type'], config['relation_total'], 16, 16, SEC_EMB*2, 4, dropout=0.1)
            #self.global_pool2 = aggr.MaxAggregation()
            #self.bn2 = gnn.norm.BatchNorm(64)
            self.lin = nn.Linear(64, SEC_EMB*2)
            self.lin2 = nn.Linear(SEC_EMB*2, 16)
            self.num_rel = config['relation_total']
            self.rel_type_embeddings = nn.Embedding(config['relation_total']+2, 16)
            #self.year_embeddings    = nn.Embedding(24, 16, padding_idx=0)
            self.month_embeddings   = nn.Embedding(13, 16, padding_idx=0)
            self.day_embeddings     = nn.Embedding(32, 16, padding_idx=0)
            self.hour_embeddings    = nn.Embedding(25, 16, padding_idx=0)
            self.minutes_embeddings = nn.Embedding(61, 16, padding_idx=0)
            self.sec_embeddings     = nn.Embedding(61, 16, padding_idx=0)
                
        self.use_kg = USE_KG
        self.config = config

        if self.use_kg:
            #self.relation_kge = TorusEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.relation_kge = HolEModel(n_entities= config['entity_total'], n_relations = config['relation_total'], emb_dim = config['embedding_size'] )
             
            self.kge = TTransEModel(config) #TADistmultModel(config) #TTransEModel(config)
        if self.use_kg:
            pass
            #self.temporal_kge = TorusEModel(n_entities= 5500, n_relations = 50, emb_dim = SEC_EMB, dissimilarity_type='torus_L2')
            #self.temporal_kge = HolEModel(n_entities= 5500, n_relations = 40, emb_dim = SEC_EMB) 
        self.num_nodes = NUM_NODES

        self.ent_embeddings = nn.Embedding(config['entity_total']+2, 20)
        self.num_nodes = config['entity_total']+2

  
    def forward(self, xb, yb=None, graph=None, kg=None, tkg=None, relation_graph = None):
        kg_loss = 0
        #if self.is_pos:
        #    xb = self.pos_enc_x(xb)
        #    yb = self.pos_enc_y(yb)
        #yb = torch.cat((yb, emb2), dim=2)

        if self.encoder_architecture == 'transf':
            x = self.transformer_encoder_first(xb.transpose(1,2)).mean(dim=1)
            x = x.unsqueeze(dim=0)
        elif self.encoder_architecture == 'lstm':
            x, y = self.lstm_encoder(xb)
            x = y[0][-1, :, :].unsqueeze(dim=0)          # x: [B, C, W*F
        elif self.encoder_architecture == 'gru':
            x, y = self.gru_encoder(xb)
            x = y[-1, :, :].unsqueeze(dim=0)

        if self.use_graph and self.is_hyper_graph:
            print("we are at 1")
            g_emb = self.graph_model(graph['x'], graph['hyperedge_index']).unsqueeze(dim=0)
            #g_emb = g_emb.repeat(1, self.time_steps, 1)
            x = torch.cat((x, g_emb), dim=2)
        elif self.use_graph == 'gcn' and not self.is_hyper_graph:
            print("we are at 2")
            edge = torch.cat((tkg[0].unsqueeze(0), tkg[2].unsqueeze(0)), dim=0)
            batch = torch.ones(edge.shape[1]).long().to(tkg[0].device)
            g_emb = self.graph_model(self.ent_embeddings.weight, edge, batch)[tkg[4].long()]
            g_emb = g_emb.unsqueeze(dim=0)            
            
            #g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])    
            x = torch.cat((x, g_emb), dim=2)

        if self.use_relation_graph:
            if self.use_relation_graph == 'gcn':
                print("we are at 3")
                nodes = relation_graph[1].max() +1
                node_list = torch.arange(nodes).to(relation_graph.device)
                node_emb = self.rel_node_emb(node_list)
                batch = torch.ones(relation_graph[1].shape).to(relation_graph.device)
                rel_emb = self.rel_graph_model(node_emb, relation_graph, batch)[:x.shape[1]].unsqueeze(dim=0)
            elif self.use_relation_graph == 'hypergraph' or self.use_relation_graph == 'with_sector':
                print("we are at 4")
                nodes = relation_graph[0].max() +1
                node_list = torch.arange(nodes).to(relation_graph.device)
                node_emb = self.rel_node_emb(node_list)
                rel_emb = self.rel_graph_model(node_emb, relation_graph)[:x.shape[1]].unsqueeze(dim=0)
            x = torch.cat((x, rel_emb), dim=2)


        elif self.use_graph == 'rgat':
            print("we are at 5")
            edge = torch.cat((tkg[0].unsqueeze(0), tkg[2].unsqueeze(0)), dim=0)
            batch = torch.ones(edge.shape[1]).long().to(tkg[0].device)

            edge_attr = self.month_embeddings(tkg[3][:, 1]) + \
                    self.day_embeddings(tkg[3][:, 2]) + self.hour_embeddings(tkg[3][:, 3]) + \
                    self.rel_type_embeddings(tkg[1])
            
            gx = self.conv1(self.ent_embeddings.weight, edge, tkg[1], edge_attr).relu()
            #gx = self.global_pool(gx, batch)
            gx = self.conv2(gx, edge, tkg[1], edge_attr).relu()
            #gx = self.global_pool2(gx, batch)
            #gx = self.conv3(gx, edge, tkg[1], edge_attr).relu()
            node_emb = self.lin(gx)
            #g_emb = self.graph_model(self.ent_embeddings.weight, edge, batch)[tkg[4].long()]
            g_emb = node_emb[tkg[4].long()]   
            self.tsne_emb = self.lin2(g_emb)    
            self.tsne_emb_all = self.lin2(node_emb)   
            g_emb = g_emb.unsqueeze(dim=0)
            
            x.fill_(0)
            #g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])   
            print(x)
            x = torch.cat((x, g_emb), dim=2)

            pos = torch.sum(self.lin2(node_emb[tkg[0].long()]) + edge_attr - self.lin2(node_emb[tkg[2].long()]), dim=1)
            kg_loss = torch.mean(pos)

            self.ent_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.rel_type_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.month_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.day_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.hour_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.minutes_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
            self.sec_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1)
        if self.use_graph == 'hgat':
            print("we are at 6")
            edge = torch.cat((tkg[0].unsqueeze(0), tkg[2].unsqueeze(0)), dim=0) 
            batch = torch.ones(self.num_nodes).long().to(tkg[0].device)
            node_type = self.node_type.to(tkg[0].device)
            
            temp_emb = self.month_embeddings(tkg[3][:, 1]) + \
                    self.day_embeddings(tkg[3][:, 2]) + self.hour_embeddings(tkg[3][:, 3]) 
            rel_emb = self.rel_type_embeddings(tkg[1])
            edge_attr = temp_emb + rel_emb

            weight = self.ent_embeddings.weight.clone()
            weight[tkg[4].long()] = self.ent_embeddings.weight[tkg[4].long()] + x[0]
            
            gx = self.conv1(weight, edge, node_type, tkg[1], edge_attr).relu()
            #gx = self.bn1(gx)
            gx = self.conv2(gx, edge, node_type, tkg[1], edge_attr).relu()
            #gx = self.bn2(gx)
            #gx = self.conv3(gx, edge, tkg[1], edge_attr).relu()
            node_emb = self.lin(gx)
            #g_emb = self.graph_model(self.ent_embeddings.weight, edge, batch)[tkg[4].long()]
            g_emb = node_emb[tkg[4].long()]   
            self.tsne_emb = self.lin2(g_emb)    
            self.tsne_emb_all = self.lin2(node_emb)   
            g_emb = g_emb.unsqueeze(dim=0)
            
            #g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])    
            x = torch.cat((x, g_emb), dim=2)

            pos = torch.sum(self.lin2(node_emb[tkg[0].long()]) + rel_emb + temp_emb - self.lin2(node_emb[tkg[2].long()]), dim=1)
            kg_loss = torch.mean(pos)
            
            self.ent_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.rel_type_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.month_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.day_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.hour_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.minutes_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            self.sec_embeddings.weight.data.renorm_(p=2, dim=1, maxnorm=1).detach()
            
        #
        if self.use_kg:
            #kg_loss = self.relation_kge.scoring_function(tkg[0], tkg[2], tkg[1]).mean()
            #kg_emb, rel_emb = self.relation_kge.get_embeddings()
            #kg_emb = kg_emb[tkg[4].long()]
            #self.relation_kge.normalize_parameters()
            print("we are at 7")
            kg_loss = self.kge(tkg[0], tkg[2], tkg[1], tkg[3])
            kg_emb = self.kge.ent_embeddings.weight[tkg[4].long()]
            self.kge.regularize_embeddings()
            
            kg_emb = kg_emb.unsqueeze(dim=0)
            x = torch.cat((x, kg_emb), dim=2)

        #if self.use_kg:
        #    #head, relation, tail = torch.cat((kg[0], tkg[0])), torch.cat((kg[1], tkg[1])), torch.cat((kg[2], tkg[2]))
        #    kg_loss += self.temporal_kge.scoring_function(tkg[0], tkg[2], tkg[1]).mean()
        #    kg_loss /= 2
        #    kg_emb, rel_emb = self.temporal_kge.get_embeddings()
        #    kg_emb = kg_emb[kg[3].long()]
        #    kg_emb = kg_emb.unsqueeze(dim=0)
        #    x = torch.cat((x, kg_emb), dim=2)

        #    self.temporal_kge.normalize_parameters()

        x = x.view(-1)
        #print(x.shape)
        price_pred = self.pred(x)
        #price_pred = F.relu(price_pred)
        #price_pred = self.pred2(price_pred)

        return price_pred, kg_loss


# ----------- Model -----------
class Saturation(nn.Module):
    
    def __init__(self, W, T, D_MODEL, N_HEAD, ENC_LAYERS, DEC_LAYERS, D_FF, DROPOUT, USE_POS_ENCODING = False, USE_GRAPH = False, HYPER_GRAPH = True, USE_KG = True, NUM_NODES = 87):
        super().__init__()

        SEC_EMB, n = 5, 0 # 1 For LSTM Embedding
        if USE_GRAPH:
            n += 1
        if USE_KG:
            n += 1

        self.lstm_encoder = nn.Linear(D_MODEL, 1)
        self.transformer_encoder = nn.Linear(W, D_MODEL)

        self.use_graph = USE_GRAPH
        self.is_hyper_graph = HYPER_GRAPH
        if self.use_graph:
            if self.is_hyper_graph:
                self.graph_model = Sequential('x, hyperedge_index', [
                        #(Dropout(p=0.5), 'x -> x'),
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                        #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                        #(global_mean_pool, 'x, batch -> x'),
                        nn.Linear(32, SEC_EMB),
                    ])
            else:
                self.graph_model = Sequential('x, edge_index, batch', [
                            #(Dropout(p=0.5), 'x -> x'),
                            (GCNConv(8, 32), 'x, edge_index -> x1'),
                            nn.ReLU(inplace=True),
                            (GCNConv(32, 64), 'x1, edge_index -> x2'),
                            nn.ReLU(inplace=True),
                            #(lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
                            #(JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
                            #(global_mean_pool, 'x, batch -> x'),
                            nn.Linear(64, SEC_EMB),
                        ])

        self.pred = nn.Linear(D_MODEL, 1)

    def forward(self, xb, yb=None, graph=None, kg=None):
        x = self.lstm_encoder(xb).squeeze()
        x = self.transformer_encoder(x)               # x: [B, C, W*F]

        if self.use_graph and self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['hyperedge_index'])
            #g_emb = g_emb.repeat(1, self.time_steps, 1)
            #x = torch.cat((x, g_emb), dim=1)
        elif self.use_graph and not self.is_hyper_graph:
            g_emb = self.graph_model(graph['x'], graph['edge_list'], graph['batch'])
            #g_emb = g_emb.repeat(1, self.time_steps, 1)        
            #x = torch.cat((x, g_emb), dim=1)

        price_pred = self.pred(x)
        return price_pred, 0, 0


