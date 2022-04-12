import numpy as np
import torch
import pickle

import torch.nn as nn 


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py




class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, dataset, args):
        super(SASRec, self).__init__()

        self.concat_layer = nn.Linear(2*args.hidden_units, args.hidden_units)
        self.LeakyReLU = nn.LeakyReLU(0.1)

        [user_train, user_valid, user_test, usernum, itemnum, bucket_idx2entity] = dataset

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device 
        self.args = args

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        if args.classKG  == 'false':
            self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
            self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units)
        elif args.classKG  == 'true':
            pt_item_emb = LoadPretrainEmbedding(bucket_idx2entity, args.hidden_units, self.item_num)
            weight = torch.FloatTensor(pt_item_emb)
            self.item_emb = nn.Embedding.from_pretrained(weight, padding_idx=0, freeze=False)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        
        # attention container
        self.attention_score_container = list()



    def log2feats(self, log_seqs, show_attention=False):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        attention_score_unit = []

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attention_score = self.attention_layers[i](Q, seqs, seqs, 
                                                                    attn_mask=attention_mask)
                                                                       # key_padding_mask=timeline_mask
                                                                       # need_weights=False) this arg do not work? 
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)
            if show_attention is True:
                attention_score_unit.append(list(attention_score.tolist())[0])
        # collect attention_score
        self.attention_score_container.append(attention_score_unit)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)\

        return log_feats



    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet
        user_ids_long = np.concatenate([user_ids.reshape(-1, 1) for _ in range(self.args.maxlen)], 1)
        user_embedding_long = self.user_emb(torch.LongTensor(user_ids_long).to(self.dev))

        log_feats = torch.cat([log_feats, user_embedding_long], -1)
        #log_feats = self.LeakyReLU(self.concat_layer(log_feats))
        log_feats = self.concat_layer(log_feats)


        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))


        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        
        return pos_logits, neg_logits # pos_pred, neg_pred



    def predict(self, user_ids, log_seqs, item_indices, show_attention=False): # for inference
        log_feats = self.log2feats(log_seqs, show_attention) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
        #print('final_feat : ', final_feat.shape)
        user_embedding = self.user_emb(torch.tensor(user_ids).to(self.dev))
        final_feat = torch.cat([final_feat, user_embedding], -1)
        #final_feat = self.LeakyReLU(self.concat_layer(final_feat))
        final_feat = self.concat_layer(final_feat)
        #print('final_feat : ', final_feat.shape)

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)


        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users
        if show_attention is False:
            return logits # preds # (U, I)
        else:
            return logits, self.attention_score_container







def LoadPretrainEmbedding(bucket_idx2entity, hidden_units, item_num):
    '''
    input : org_idx2emb_pt
    output : pretrain_emb (padding + idx)
    => idx -> org_idx -> emb_pt
    '''
    entity_embedding = np.load('data/'+'entity_embeddingH.npy')
    path = 'KG_class_data_padding.pkl'
    with open('data/'+path, "rb") as f:
        data = pickle.load(f)
    entity_name2idx = data['entity_name2idx']
    # build org_idx2emb_pt
    org_idx2emb_pt = dict()
    for entity_name in list(entity_name2idx.keys()):
        if entity_name != 'padding' and entity_name.split('@')[1] == 'MaterialID':
            MaterialID = int(entity_name.split('@')[0])
            kg_idx = entity_name2idx[entity_name]
            emb_pt = list(entity_embedding[kg_idx])
            org_idx2emb_pt[MaterialID] = emb_pt
    # idx -> org_idx -> emb_pt
    organic_idx2item = bucket_idx2entity['organic_idx2item']
    cold_item = 0
    idx2item_emb_pt = dict()
    for org_idx in list(organic_idx2item.keys()):
        idx = organic_idx2item[org_idx]
        if org_idx in org_idx2emb_pt:
            idx2item_emb_pt[idx] = org_idx2emb_pt[org_idx]
        else:
            emb_pt_random = org_idx2emb_pt[random.sample(list(org_idx2emb_pt.keys()),1)[0]]
            idx2item_emb_pt[idx] = emb_pt_random
            cold_item +=1
    print('cold_item : ', cold_item)
    print('warm_item : ', len(list(organic_idx2item.keys())) - cold_item)

    # padding
    padding_emb = [0 for _ in range(hidden_units)]

    # pretrain_embedding
    pt_item_emb = []
    pt_item_emb.append(padding_emb)
    for i in range(item_num):
        idx = i+1
        pt_item_emb.append(idx2item_emb_pt[idx])
    return pt_item_emb
