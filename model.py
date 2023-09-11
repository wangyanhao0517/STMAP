import os
import random
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from math import sqrt
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import torch.nn.functional as F


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu1, logvar1, mu2, logvar2):
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp()))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=1)
        return kl.mean(dim=0)


class NewBert(nn.Module):
    def __init__(self, args):
        super(NewBert, self).__init__()
        self.model_name = args["model"]
        self.bert_model = BertForSequenceClassification.from_pretrained(
            args["model"], num_labels=args["n_class"])
        self.noise_net = nn.Sequential(nn.Linear(args["hidden_size"],
                                                 args["hidden_size"]),
                                       nn.ReLU(),
                                       nn.Linear(args["hidden_size"],
                                                 args["hidden_size"] * 2))
        config = self.bert_model.config
        self.config = config
        self.dropout = config.hidden_dropout_prob  # 0.1
        self.args = args
        # 定义Gate 网络; 假设输入是2 X 768架构
        # self.pool = nn.AdaptiveAvgPool2d((1, 768))
        if self.args["gate"]:
            self.Gate = nn.Sequential(nn.Linear( 2 * args["hidden_size"],
                                                    args["hidden_size"]),
                                        nn.ReLU(),
                                        nn.Linear(args["hidden_size"],
                                                    3))
        # y1= F.softmax(x, dim = 0) #对每一列进行softmax
        
        
    def forward(self, input_ids,
                attention_mask,
                token_type_ids, input_chunk, labels):
        '''
                input_chunk 用不到，是为了baseline模型的效果测试
        '''
        input_ids, attention_mask, token_type_ids = input_ids.squeeze(), attention_mask.squeeze(), token_type_ids.squeeze()
        if self.args["aug"]:
            embeddings = self.bert_model.get_input_embeddings()
            encoder = self.bert_model.bert
            with torch.no_grad():
                encoder_inputs = {"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "token_type_ids": token_type_ids}

                outputs = encoder(**encoder_inputs)
                hiddens = outputs[0]
            inputs_embeds = embeddings(input_ids)
            if self.args['uniform']:
                # low is 0.95, high is 1.05 Try to produce softer noise as much as possible，to avoid semantic space collapse
                uniform_noise = torch.empty(inputs_embeds.shape).uniform_(0.9995, 1.0005).to(self.args['device'])
                noise = uniform_noise
            else:
                mask = attention_mask.view(-1)
                indices = (mask == 1)
                mu_logvar = self.noise_net(hiddens)
                mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
                zs = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                noise = zs

                prior_mu = torch.ones_like(mu)
                # If p < 0.5, sqrt makes variance the larger
                prior_var = torch.ones_like(mu) * sqrt(self.dropout / (1-self.dropout))
                prior_logvar = torch.log(prior_var)

                kl_criterion = GaussianKLLoss()
                h = hiddens.size(-1)
                _mu = mu.view(-1, h)[indices]
                _log_var = log_var.view(-1, h)[indices]
                _prior_mu = prior_mu.view(-1, h)[indices]
                _prior_logvar = prior_logvar.view(-1, h)[indices]

                kl = kl_criterion(_mu, _log_var, _prior_mu, _prior_logvar)

            # Random discarding to aug
            rands = list(set([random.randint(1, inputs_embeds.shape[0] - 1) for i in range(self.args["zero_peturb"])]))
            for index in rands:
                embed_ = inputs_embeds[index, :, :]
                length = random.randint(1, 3)
                for iter in range(length):
                    index_ = random.randint(1, inputs_embeds.shape[1] - 1)
                    vec = torch.rand(1, inputs_embeds.shape[-1]).to(self.args["device"])
                    embed_[index_] = vec
            
            inputs = {"inputs_embeds": inputs_embeds * noise,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids,
                      "labels":labels}

            noise_outputs = self.bert_model(**inputs, output_hidden_states = True)
            noise_loss = noise_outputs[0]

            new_inputs = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "labels":labels}

            outputs = self.bert_model(**new_inputs, output_hidden_states = True)
            nll = outputs[0]
            
            if self.args["gate"]:
                # 获取CLS
                last_noise = noise_outputs.hidden_states[-1]
                last = outputs.hidden_states[-1]
                cls_noise = last_noise[:,:1,:].squeeze()
                cls = last[:,:1,:].squeeze()
                cls_total = torch.cat((cls_noise, cls), dim = 1)
                cls_total = torch.mean(cls_total, dim = 0).unsqueeze(dim = 0)
                # 将CLS通过Gate网络
                res = self.Gate(cls_total)
                Gates = F.softmax(res, dim = -1).squeeze()
                loss = noise_loss * Gates[0] + nll * Gates[1]
            else:
                loss = nll + 0.001 * noise_loss
            if self.args['uniform']:
                return (loss, 0 * loss, outputs.logits)
            else:
                return (loss, kl, outputs.logits)
        else:
            inputs = {"input_ids": input_ids,
                      "attention_mask": attention_mask,
                      "token_type_ids": token_type_ids}
            # output_hidden_states = True 输出隐含状态
            outputs = self.bert_model(**inputs, labels=labels)
            return outputs.loss, 0, outputs.logits
            