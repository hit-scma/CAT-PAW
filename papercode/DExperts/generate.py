
from typing import Union, List
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json
import random

import numpy as np

from argparse import ArgumentParser

import sys
sys.path.append("..") 
from classifier.annotator import Attn, MLP


def exam_BOW_distribution(good_index, probs):
    #good_index = [50257]
    #print(probs)
    #prob [bsz, vocab]
    return torch.sum(probs * good_index[None,:],dim=-1)


def exam_Disc_distribution(true_hidden, annotator, temperature=0.5):
    probs = F.softmax(annotator(true_hidden)/temperature, dim=-1)[:,-1,:].detach()
    size = probs.shape[-1]
    dist = torch.tensor(range(size)).to(probs.device) * (1/size) + (0.5/size)
    
    res = torch.abs(torch.sum(probs * dist, dim=-1).squeeze() - 0.5)

    #print(res)
    #raise Exception

    return res


def clip(args, hyperscale, threshold):
    if args.alpha > threshold:
        threshold = args.alpha
    hyperscale = torch.min(hyperscale, torch.tensor(threshold).float().to(hyperscale.device))
    return hyperscale


def main(args):
    base_model = args.base_model
    expert_model = args.expert
    antiexpert_model = args.antiexpert

    if args.task == 'sentiment':
        file_name = 'res/sentiment/pos_' + args.type + '.json'
    elif args.task == 'toxicity':
        file_name = 'res/toxicity/non_' + args.type + '.json'
    
    with open(file_name, 'a') as f:
        res_list = []
        for prefix in args.prompts.split(';'):
            for i in range(1):
                with torch.no_grad():
                    batch_size = args.samples
                    tokens = args.tokenizer([args.tokenizer.eos_token + prefix] * batch_size, return_tensors='pt')
                    input_text = tokens.input_ids.long().to(args.device)
                    attention_mask = tokens.attention_mask.long().to(args.device)
                    cur_len = input_text.shape[1]
                    max_length = args.length
                    base_past = None
                    expert_past = None
                    antiexpert_past = None
                    base_hiddens = None
                    prev = None
                    result = input_text
                    while cur_len < max_length:
                        if base_past is None:
                            base_dict = base_model(input_text, attention_mask=attention_mask, return_dict=True, use_cache=True, output_hidden_states=True)
                            base_logits, base_past, hidden_states = base_dict.logits, base_dict.past_key_values, base_dict.hidden_states[-1]
                            if args.type == 'dis':
                                base_hiddens = hidden_states.detach()

                            expert_dict = expert_model(input_text, attention_mask=attention_mask, return_dict=True, use_cache=True)
                            expert_logits, expert_past = expert_dict.logits, expert_dict.past_key_values
                            assert expert_past is not None

                            antiexpert_dict = antiexpert_model(input_text, attention_mask=attention_mask, return_dict=True, use_cache=True)
                            antiexpert_logits, antiexpert_past = antiexpert_dict.logits, antiexpert_dict.past_key_values
                            assert antiexpert_past is not None

                        else:
                            base_dict = base_model(prev, past_key_values=base_past, attention_mask=attention_mask, return_dict=True, use_cache=True, output_hidden_states=True)
                            base_logits, base_past, hidden_states = base_dict.logits, base_dict.past_key_values, base_dict.hidden_states[-1]
                            if args.type == 'dis':
                                base_hiddens = torch.cat((base_hiddens, hidden_states), dim=1)
                            
                            expert_dict = expert_model(prev, past_key_values=expert_past, attention_mask=attention_mask, return_dict=True, use_cache=True)
                            expert_logits, expert_past = expert_dict.logits, expert_dict.past_key_values
                            assert expert_past is not None

                            antiexpert_dict = antiexpert_model(prev, past_key_values=antiexpert_past, attention_mask=attention_mask, return_dict=True, use_cache=True)
                            antiexpert_logits, antiexpert_past = antiexpert_dict.logits, antiexpert_dict.past_key_values
                            assert antiexpert_dict is not None

                        
                        
                        
                        
                        if args.type == 'raw':
                            hyperscale = args.alpha
                        elif args.type == 'bow':
                            base_probs = torch.softmax(base_logits[:,-1,:].squeeze(), dim=-1)
                            alter_scale = exam_BOW_distribution(args.word_index, base_probs) 
                            hyperscale = alter_scale * (args.alpha/args.activesize)
                            hyperscale = clip(args, hyperscale, 1.3)[:,None,None]
                        elif args.type == 'dis':
                            alter_scale = exam_Disc_distribution(base_hiddens.detach(), args.annotator)
                            hyperscale = alter_scale * (args.alpha/args.activesize)
                            hyperscale = clip(args, hyperscale.view(batch_size, 1), 1.3).squeeze()[:, None, None]
                            #print(hyperscale.shape)

                        


                        ensemble_logits = base_logits + hyperscale * (expert_logits - antiexpert_logits)
                        
                        ensemble_probs = torch.softmax(ensemble_logits[:,-1,:].squeeze(), dim=-1)
                        top_probs, top_indices = torch.topk(ensemble_probs, args.topk, dim=-1)

                        ind = torch.multinomial(top_probs, num_samples=1)

                        prev = top_indices.gather(-1, ind)
                        #ensemble_probs = torch.softmax(top_k_filtering(ensemble_logits[:,-1,:]), dim=-1)

                        #prev = torch.multinomial(ensemble_probs, num_samples=1)
                        

                        result = torch.cat([result, prev], dim=-1)

                        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
                        cur_len += 1
                        #prev = torch.index_select(top_indices, -1, tmp_prev[0]).view(batch_size, 1)
                        #result = torch.cat((result, prev), dim=-1)
                    res_list.extend([args.tokenizer.decode(r) for r in result])
        f.write(json.dumps({'pos'+str(args.alpha):res_list}))
        f.write('\n')



if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument('--prompts', type=str, default="My dog died;The food is awful")
    parser.add_argument('--alpha', type=float, default=0.68)
    parser.add_argument('--length', type=int, default=50, help='max length')
    #parser.add_argument('--step', type=float, default=0)

    parser.add_argument('--task', type=str, default='sentiment', choices=('sentiment', 'toxicity'))

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--type', type=str, default='raw', choices=('raw', 'bow', 'dis'))
    parser.add_argument('--activesize', type=float, default=0.2)
    parser.add_argument('--topk', type=int, default=100)

    args = parser.parse_args()
    print('Loading...')
    base_model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(args.device)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    args.base_model = base_model
    print('Language Model Loaded')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.task == 'sentiment':
        expert = GPT2LMHeadModel.from_pretrained('experts/sentiment/large/finetuned_gpt2_positive').to(args.device)
    elif args.task == 'toxicity':
        expert = GPT2LMHeadModel.from_pretrained('experts/toxicity/large/finetuned_gpt2_nontoxic').to(args.device)
    for param in expert.parameters():
        param.requires_grad = False
    expert.eval()
    args.expert = expert

    print('Expert Loaded')

    if args.task == 'sentiment':
        antiexpert = GPT2LMHeadModel.from_pretrained('experts/sentiment/large/finetuned_gpt2_negative').to(args.device)
    elif args.task == 'toxicity':
        antiexpert = GPT2LMHeadModel.from_pretrained('experts/toxicity/large/finetuned_gpt2_toxic').to(args.device)
    for param in antiexpert.parameters():
        param.requires_grad = False
    antiexpert.eval()
    args.antiexpert = antiexpert

    print('Antiexpert Loaded')
    args.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    args.tokenizer.pad_token = args.tokenizer.eos_token

    if args.task == 'sentiment':
        if args.type == 'bow':
            args.word_index = [0] * 50257
            with open('../wordlists/sentiment_pos.txt', 'r') as f:
                for line in f.readlines():
                    args.word_index[args.tokenizer(line.strip().replace('[SPC]', ' ')).input_ids[0]]=1
            with open('../wordlists/sentiment_neg.txt', 'r') as f:
                for line in f.readlines():
                    args.word_index[args.tokenizer(line.strip().replace('[SPC]', ' ')).input_ids[0]]=1
            args.word_index = torch.tensor(args.word_index).long().to(args.device)
        else:
            args.word_index = None
        
        if args.type == 'dis':
            args.annotator = Attn(1024, 10)
            args.annotator.load_state_dict(torch.load("../classifier/Yelp_model_epoch15.pt"))
            args.annotator.to(args.device)
            args.annotator.eval()
        else:
            args.annotator = None
    
    elif args.task == 'toxicity':
        if args.type not in ['raw', 'bow']:
            args.type = 'raw'
        args.word_index = [0] * 50257
        with open('../wordlists/toxicity.txt', 'r') as f:
            for line in f.readlines():
                args.word_index[args.tokenizer(line.strip().replace('[SPC]', ' ')).input_ids[0]]=1

        args.word_index = torch.tensor(args.word_index).long().to(args.device)

        args.annotator = None


    main(args)

