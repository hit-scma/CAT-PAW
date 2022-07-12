import transformers
from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
import torch.nn.functional as F
import torch.optim as optim
import argparse
import json


class Attn(torch.nn.Module):
    def __init__(self, embed_size, label_size):
        super(Attn, self).__init__()
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.embed_size = embed_size
        self.label_size = label_size
        self.c_key = nn.Linear(self.embed_size, self.embed_size)
        self.c_query = nn.Linear(self.embed_size, self.embed_size)
        self.c_value = nn.Linear(self.embed_size, self.embed_size)
        self.v_drop = nn.Dropout(0.5)
        self.head = nn.Sequential(
            nn.Linear(self.embed_size, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, self.label_size)
        )
    
    def forward(self, hidden, attention_mask=None):
        key = self.c_key(hidden).permute(0,2,1)
        query = self.c_query(hidden)
        value = self.v_drop(self.c_value(hidden))
        w = torch.matmul(query, key)

        nd, ns = w.size(-2), w.size(-1)

        mask = self.bias[:, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        #print(w.shape)

        batch_size = hidden.shape[0]

        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"

            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, :]
            #print(attention_mask.shape)
            attention_mask = (1.0 - attention_mask) * -10000.0
            w = w + attention_mask
            #print(w.shape)
        
        w = nn.Softmax(dim=-1)(w)
        hidden = torch.matmul(w, value)
        hidden = self.head(hidden)
        
        return hidden

class MLP(torch.nn.Module):
    def __init__(self, embed_size, label_size):
        super(MLP, self).__init__()
        self.embed_size = embed_size
        self.label_size = label_size
        self.head = nn.Sequential(
            nn.Linear(self.embed_size, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, self.label_size)
        )
    
    def forward(self, hidden, attention_mask=None):


        #if attention_mask is not None:
        #    hidden = hidden * attention_mask[:,:,None]
        #hidden = torch.mean(hidden, dim=1).squeeze()
        #print(hidden.shape)
        
        hidden = self.head(hidden)
        
        return hidden



class Annotator(torch.nn.Module):

    def __init__(
            self,
            device='cuda',
            label_size=10,
            annotator='attn'
    ):
        super(Annotator, self).__init__()
        self.encoder = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.label_size = label_size
        '''
        self.annotator = nn.Sequential(
            nn.Linear(self.embed_size, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, label_size)
        )
        '''
        if annotator == 'attn':
            self.annotator = Attn(self.embed_size, self.label_size)

        elif annotator == 'mlp':
            self.annotator = MLP(self.embed_size, self.label_size)

        self.device = device

        self.encoder.to(device)
        self.annotator.to(device)

        self.freeze_encoder()

    def get_annotator(self):
        return self.annotator

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def train_custom(self):
        self.annotator.train()
    
    def eval_custom(self):
        self.annotator.eval()


    def forward(self, tokens, attention_mask=None, labels=None, temperature=1):
        hidden, _ = self.encoder.transformer(tokens, attention_mask=attention_mask)
        hidden = self.annotator(hidden, attention_mask=attention_mask)
        #probs = F.softmax(self.annotator(hidden))
        #probs = torch.sigmoid(self.annotator(hidden))

        loss = None
        if labels is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            shift_probs = hidden[..., :-1, :].contiguous() #batch_size,  seq_length
            shift_labels = labels[..., 1:].contiguous()
            shift_probs = shift_probs.permute(2,0,1)[:, shift_attention_mask.bool()].permute(1,0)
            #shift_probs = torch.masked_select(shift_probs, shift_attention_mask.bool())
            shift_labels = torch.masked_select(shift_labels, shift_attention_mask.bool())

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_probs.view(-1, self.label_size), shift_labels.view(-1))


            return loss
        return F.softmax(hidden/temperature, dim=-1)
