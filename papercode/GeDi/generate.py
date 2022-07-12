import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
from tqdm import tqdm
import json
import random
import numpy as np
import torch.nn.functional as F

import fcntl
import time

from modeling_gpt2 import GPT2LMHeadModel as GeDiModel
import sys

sys.path.append("..") 
from classifier.annotator import Attn, MLP


parser = argparse.ArgumentParser()

parser.add_argument(
    "--gedi_model_name_or_path",
    default="./pretrained_models/gedi_sentiment",
    type=str,
        help="Path to GeDi model. Assumes path from --mode if set to None",
)
parser.add_argument("--length", type=int, default=50, help= "generation length")
#parser.add_argument("--disc_weight", type=float, default=30.0,
#                    help="weight for GeDi discriminator",
#)
parser.add_argument("--samples", type=int, default=10)
parser.add_argument("--do_sample", action="store_true",
                    help="If set to False greedy decoding is used. Otherwise sampling is used. Defaults to True.")
parser.add_argument("--mode", type=str, default="sentiment", help="topic, sentiment, detoxify")
parser.add_argument("--prompt", type=str, default="My dog died",  help= "prompt for generation" )
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--type', type=str, default='raw', choices=('raw', 'bow', 'dis'))
parser.add_argument('--activesize', type=float, default=0.2)
parser.add_argument('--topk', type=int, default=100)

parser.add_argument('--disc_weight', type=float, default=40.0)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.device = device

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
for param in model.parameters():
    param.requires_grad = False
model.to(device)
model.eval()

gedi_model = GeDiModel.from_pretrained(args.gedi_model_name_or_path)
for param in gedi_model.parameters():
    param.requires_grad = False
gedi_model.to(device)
gedi_model.eval()

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

if args.mode == "sentiment":
    args.code_desired = "positive"
    args.code_undesired = "negative"


result_list = []

for prefix in args.prompt.split(';'):
    for i in tqdm(range(args.samples)):
        with torch.no_grad():
            batch_size = 1
            input_text = torch.tensor([tokenizer(tokenizer.eos_token + prefix).input_ids]*batch_size).long().to(args.device)
            cur_len = input_text.shape[1]
            max_length = args.length
            past_key_values = None
            result = input_text
            prev = None
            origin_hidden_states = None
            
            
            ##GeDi
            pt_id = tokenizer.encode(args.code_desired)[0]
            nt_id = tokenizer.encode(args.code_undesired)[0]
            seq_a = (torch.ones(input_text.shape[0])*pt_id).type_as(input_text).view(-1,1)
            seq_b = (torch.ones(input_text.shape[0])*nt_id).type_as(input_text).view(-1,1)
            seq_a = torch.cat((seq_a, input_text), dim=1)[:,:]
            seq_b = torch.cat((seq_b, input_text), dim=1)[:,:]
            seq_batched = torch.cat((seq_a,seq_b),dim=0)
            gedi_past = None
            ##
            while cur_len < max_length:
                if past_key_values is None:
                    dic = model(input_text, return_dict=True, use_cache=True, output_hidden_states=True)
                    origin_logits, past_key_values, hidden_states = dic.logits, dic.past_key_values, dic.hidden_states[-1]
                    assert hidden_states is not None
                    if args.type == 'dis':
                        origin_hidden_states = hidden_states.detach()
                    
                    #print(len(origin_hidden_states))
                    #print(origin_hidden_states[-1].shape)

                    ##GeDi
                    gedi_outputs = gedi_model(seq_batched)
                    shift_logits = gedi_outputs[0][..., :-1, :].contiguous()
                    shift_labels = seq_batched[..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    logits_r  = -1*loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    logits_r = logits_r.view(seq_batched.shape[0], -1)
                    #seq_len = logits_r.shape[1]
                    logits_r = torch.sum(logits_r,1)
                    #logits_pos,logits_neg = torch.split(logits_r/seq_len,input_text.shape[0])
                    #logits0 = torch.stack((logits_pos,logits_neg),1)
                    #if "logit_scale" in dir(gedi_model):
                    #    logits0 = gedi_model.logit_scale*logits0
                    #if "bias" in dir(gedi_model):
                    #    logits0 = logits0 + gedi_model.bias
                    #logp_desired = torch.log_softmax(logits0,-1)[:,0]
                    #logp_undesired = torch.log_softmax(logits0,-1)[:,1]
                    ##

                else:
                    dic = model(prev, past_key_values=past_key_values, return_dict=True, use_cache=True, output_hidden_states=True)
                    origin_logits, past_key_values, hidden_states = dic.logits, dic.past_key_values, dic.hidden_states[-1]
                    assert hidden_states is not None
                    if args.type == 'dis':
                        origin_hidden_states = torch.cat((origin_hidden_states, hidden_states), dim=1)
                    
                    #print(len(origin_hidden_states))
                    #print(origin_hidden_states[-1].shape)

                    ##GeDi
                    gedi_outputs = gedi_model(torch.cat((prev,prev), dim=0), past=gedi_past)
                    ##
                
                if args.type == 'raw':
                    alter_scale = 1.0
                elif args.type == 'bow':
                    #print(origin_logits.shape)
                    alter_scale = exam_BOW_distribution(word_index, torch.softmax(origin_logits, dim=-1)[:,-1,:].squeeze()) / args.activesize
                    #print(alter_scale)
                    #print(alter_scale)
                elif args.type == 'dis':
                    alter_scale = exam_Disc_distribution(origin_hidden_states, annotator) / args.activesize
                    #print(alter_scale)

                tot_scale = alter_scale * args.disc_weight
                if args.disc_weight > 110:
                    if tot_scale > args.disc_weight:
                        tot_scale = args.disc_weight
                else:
                    if tot_scale > 110:
                        tot_scale = 110.0

                cur_len = cur_len + 1
                gedi_logits = (torch.log_softmax(gedi_outputs[0][:, -1, :],-1)+logits_r.unsqueeze(1))
                logits_pos,logits_neg = torch.split(gedi_logits/cur_len,input_text.shape[0])
                logits = torch.stack((logits_pos,logits_neg),2)
                if "logit_scale" in dir(gedi_model):
                    logits = gedi_model.logit_scale*logits

                if "bias" in dir(gedi_model):
                    logits = logits + gedi_model.bias

                logp_desired_t = torch.log_softmax(logits,-1)[:,:,0]
                logp_undesired_t = torch.log_softmax(logits,-1)[:,:,1]
                
            
                log_origin_logits = torch.log_softmax(origin_logits[:, -1, :].view(batch_size, origin_logits.shape[-1]),-1)
                next_token_logits = log_origin_logits + tot_scale*(logp_desired_t)
                gedi_past = gedi_outputs[1]

                #print(next_token_logits.shape)
                #if args.do_sample:
                top_probs, top_indices = torch.topk(torch.exp(next_token_logits), args.topk, dim=-1)
                #print(top_probs)
                #print(top_indices)
                tmp_prev = torch.multinomial(top_probs, num_samples=1)
                #print(tmp_prev)
                prev = torch.index_select(top_indices, -1, tmp_prev[0]).view(batch_size, 1)
                result = torch.cat((result, prev), dim=-1)
                #print(tokenizer.decode(result[0]))
            result_list.append(tokenizer.decode(result[0]))
    
with open('pos_' + args.type + '.txt', 'a') as f:
    f.write(json.dumps(result_list))





