from logging import raiseExceptions
import os
import random
import json
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline, set_seed, GPT2Tokenizer
from modeling_gpt2 import GPT2LMHeadModel

import sys
sys.path.append("..") 
from classifier.annotator import Attn, MLP


class ClassificationHead(torch.nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, class_size=5, embed_size=2048):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = (torch.nn.Linear(embed_size, class_size))

    def forward(self, hidden_state):
        lm_logits = self.mlp(hidden_state)
        return lm_logits





def main(args):
    if args.neg:
        label = 3
    else:
        label = 2
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token

    if args.type == 'dis':
        annotator = Attn(1024, 10)
        annotator.load_state_dict(torch.load("../classifier/Yelp_model_epoch15.pt"))
        annotator.to(args.device)
        annotator.eval()
    else:
        annotator = None

    if args.type == 'bow':
        word_index = []
        with open('../wordlists/sentiment_pos.txt', 'r') as f:
            for line in f.readlines():
                word_index.append(tokenizer(line.strip().replace('[SPC]', ' ')).input_ids)
        with open('../wordlists/sentiment_neg.txt', 'r') as f:
            for line in f.readlines():
                word_index.append(tokenizer(line.strip().replace('[SPC]', ' ')).input_ids)
        #print(word_index)
            


    def exam_BOW_distribution(good_index, probs):
        #good_index = [[input_id1], [inputid2], ...]
        #print(probs)
        sum = 0
        for indices in good_index:

            for ids in indices:
                
                sum += probs[ids].item()

        return sum
    
    def exam_Disc_distribution(true_hidden, annotator, temperature=0.5):
        probs = F.softmax(annotator(true_hidden)/temperature, dim=-1)[:,-1,:].to('cpu')
        size = probs.shape[-1]
        dist = torch.tensor(range(size)) * (1/size) + (0.5/size)
        
        res = torch.abs(torch.sum(probs * dist, dim=-1).squeeze() - 0.5).item()

        #print(res)
        #raise Exception

        return res


    model = args.model
    #model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(args.device)
    #model.eval()
    #for param in model.parameters():
    #    param.requires_grad = False

    conditioning_model = args.conditioning_model

    with open('pos_' + args.type + '.json', 'a') as f:
        res_list = []
        for prefix in args.input_text.split(';'):
            for i in tqdm(range(args.samples)):
                with torch.no_grad():
                    batch_size = 1
                    #temperature = 1.0
                    #if args.condition_lambda > 2 * temperature:
                    #    temperature = args.condition_lambda / 2
                    input_text = torch.tensor([tokenizer(tokenizer.eos_token + prefix).input_ids]*batch_size).long().to(args.device)
                    cur_len = input_text.shape[1]
                    max_length = args.length
                    past_key_values = None
                    accumulated_hidden_states = 0
                    origin_hidden_states = None
                    prev = None
                    result = input_text
                    while cur_len < max_length:
                        if past_key_values is None:
                            dic = model(input_text, return_dict=True, use_cache=True)
                            logits, past_key_values, hidden_states = dic.logits, dic.past_key_values, dic.hidden_states
                            accumulated_hidden_states = torch.sum(hidden_states.detach(), dim=1).squeeze()
                            if args.type == 'dis':
                                origin_hidden_states = hidden_states.detach()#[batch_size=1, seq_length, hidden_size=1024]
                        else:
                            dic = model(prev, past_key_values=past_key_values, return_dict=True, use_cache=True)
                            logits, past_key_values, hidden_states = dic.logits, dic.past_key_values, dic.hidden_states
                            accumulated_hidden_states += hidden_states.detach().squeeze()
                            if args.type == 'dis':
                                origin_hidden_states = torch.cat((origin_hidden_states, hidden_states), dim=1)
                        #print(accumulated_hidden_states.shape)
                        probs = torch.softmax(logits[:,-1,:].squeeze(), dim=-1)#[tokens=50257]

                        if args.type == 'raw':
                            alter_scale = 1.0
                        elif args.type == 'bow':
                            alter_scale = exam_BOW_distribution(word_index, probs) / args.activesize
                            #print(alter_scale)
                        elif args.type == 'dis':
                            alter_scale = exam_Disc_distribution(origin_hidden_states, annotator) / args.activesize
                            
                        top_probs, top_indices = torch.topk(probs, args.precondition_topk, dim=-1)
                        next_hidden_states = model(top_indices.unsqueeze(1), past_key_values=past_key_values, return_dict=True, use_cache=False).hidden_states.detach().squeeze()
                        cond_probs = torch.softmax(conditioning_model((next_hidden_states + accumulated_hidden_states.unsqueeze(0).expand(args.precondition_topk, 1024)) / (cur_len + 1)), dim=-1)[:,label]

                        #cond_probs.shape: [200]
                        #top_probs.shape: [200]

                        tot_scale = alter_scale * args.condition_lambda
                        #print(alter_scale)
                        if tot_scale > 10.0:
                            tot_scale = 10.0
                        full_probs = torch.exp((torch.log(top_probs) + tot_scale * torch.log(cond_probs)))
                        
                        #print(full_probs)
                        if args.topk >= args.precondition_topk:
                            tmp_prev = torch.multinomial(full_probs, num_samples=1)
                            cur_len += 1
                            prev = torch.index_select(top_indices, -1, tmp_prev).view(batch_size, 1)
                        else:
                            sorted, indices = torch.sort(full_probs, descending=True)
                            tmp_prev = torch.multinomial(sorted[:args.topk], num_samples=1)
                            tmp_prev = indices[tmp_prev]
                            cur_len += 1
                            prev = torch.index_select(top_indices, -1, tmp_prev).view(batch_size, 1)
                            #full_probs = torch.softmax(full_probs, dim=-1)
                            #sub_top_probs, sub_top_indices = torch.topk(full_probs, args.topk, dim=-1)
                            #tmp_prev = torch.multinomial(sub_top_probs, num_samples=1)
                            #cur_len += 1
                            #prev = torch.index_select(sub_top_indices, -1, tmp_prev)
                            #prev = torch.index_select(top_indices, -1, prev).view(batch_size, 1)
                        #print(sub_top_probs)
                        #print(sub_top_indices)
                        #
                        #print(test_tmp_prev)
                        
                        #print(tmp_prev)
                        
                        #print(prev)
                        #prev = torch.index_select(top_indices, -1, tmp_prev).view(batch_size, 1)
                        #print(prev)
                        
                        result = torch.cat((result, prev), dim=-1)
                        #print(tokenizer.decode(result[0]))
                    res_list.append(tokenizer.decode(result[0]))
        f.write(json.dumps({'pos'+str(args.condition_lambda):res_list}))
        f.write('\n')






if __name__=='__main__':
    parser = ArgumentParser()

    # DATA

    parser.add_argument('--input_text', type=str, default="My dog died", help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--condition_lambda', type=float, default=5.5, help='lambda weight on conditioning model')
    parser.add_argument('--length', type=int, default=50, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--neg', action='store_true', default=False)
    parser.add_argument('--type', type=str, default='raw', choices=('raw', 'bow', 'dis'))
    parser.add_argument('--activesize', type=float, default=0.2)
    parser.add_argument('--topk', type=int, default=200)

    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    args.model = model

    conditioning_model = ClassificationHead(class_size=5, embed_size=1024).to(args.device)
    conditioning_model.load_state_dict(torch.load("../PPLM/discrim_models/sentiment_classifierhead.pt"))
    for param in conditioning_model.parameters():
        param.requires_grad = False
    conditioning_model.eval()

    args.conditioning_model = conditioning_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ###testing
    
    main(args)
    ###
    #main(args)