'''
Descripttion: 
version: 
Author: wyh
Date: 2022-03-24 15:09:16
LastEditors: wyh
LastEditTime: 2022-03-30 14:07:15
'''
import sys
import copy
import torch.nn as nn
from torch.nn import CrossEntropyLoss
sys.path.extend(['./', '../', '../../'])
from src.cross_attention import *
from src.config import get_argparse
from transformers import RobertaModel, AlbertModel
args = get_argparse().parse_args()

class Cross_Model(nn.Module):
    "Fusioncoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, dropout_prob=0.1, d_model=768, d_ff=1248, h=8, dropout=0.1):
        super(Cross_Model, self).__init__()
        c = copy.deepcopy
        # 定义encoder
        self.n_class = args.n_class
        self.encoder = AlbertModel.from_pretrained(args.crossmodel)
        # 定义cross encoder所需要的代码
        self.attn = MultiHeadedAttention(h, d_model)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 建立两个单独的cross-encoder
        self.Fusion1 = FusionCoder(d_model, c(self.attn), c(self.ff), dropout)
        # self.Fusion2 = FusionCoder(d_model, c(self.attn), c(self.ff), dropout)
        # 建立一个线性层
        self.linear = nn.Linear(d_model*2, args.output_dim)
        # 池化层
        self.pool = nn.AdaptiveMaxPool2d((1, args.hidden_size))
        
        # 分类网络
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(args.hidden_size, self.n_class)
 
    def forward(self, query_input, event_hidden, label):
        '''query_input为tokenizer后转化为dataset再转化为dataloader的结果；
        
            event_input为交互文本得到的最后一层hidden_state
        '''
        # 直接返回cls
        # return self.encoder(input_ids = query_input['input_ids'],
        #                       attention_mask=query_input['attention_mask'],
        #                       token_type_ids=query_input['token_type_ids']).last_hidden_state[:, 0]
        # 返回cross-attention的结果
        # query_hidden, event_hideen = self.encoder_input(query_input), self.encoder_input(event_input)
        input_ids, attention_mask, token_type_ids = query_input["input_ids"].squeeze(), query_input["attention_mask"].squeeze(), query_input["token_type_ids"].squeeze()
        query_input = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}
        query_hidden = self.encoder_input(query_input)
        query = self.Fusion1(query_hidden, event_hidden)
        # event = self.Fusion2(query_hidden, event_hideen)

        # query = torch.cat((query, event), -1)
        # query = self.linear(query)
        query = self.pool(query).squeeze()
        query = query.squeeze()
        logits = self.classifier(query)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.n_class), label.view(-1))
        return logits, loss
    
    def encoder_input(self, query):
        '''
        通过Albert_tiny获取语义块的embedding
        '''
        output = self.encoder(**query)  
        query_cls = output[0]
        query_hidden = output.last_hidden_state
        return query_hidden



if __name__ == '__main__':
    '''
    query函数使用实例
    传入query，event的embedding，返回经过cross-attention之后embedding
    '''
    query_embedding = np.ones(shape = (8, 512, 512)).astype(np.float32)
    event_embedding =  np.ones(shape = (8, 512, 512)).astype(np.float32)
    query_embedding = torch.from_numpy(np.resize(query_embedding, (8, 512, 512)))
    event_embedding = torch.from_numpy(np.resize(event_embedding, (8, 512, 512)))
    N=6; d_model=512; d_ff=2048; h=8; dropout=0.1
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 建立两个单独的cross-encoder
    Fusion1 = FusionCoder(d_model, c(attn), c(ff), dropout)
    Fusion2 = FusionCoder(d_model, c(attn), c(ff), dropout)

    query = Fusion1(event_embedding, query_embedding)
    event = Fusion2(query_embedding, event_embedding)
    # query = torch.cat((query, event), -1)

    # print(query(query_embedding, event_embedding))
    # print(query_embedding.shape)
    print(query.shape, event.shape)