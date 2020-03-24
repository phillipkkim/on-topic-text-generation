'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import numpy as np
from model_learn_embed import (GPT2LMHeadModel)
from utils import load_weight
from config import GPT2Config
from sample import sample_sequence
from encoder import get_encoder
from csv_dataset import CSVDataset, NewsDataset
import requests
import csv
from tqdm import tqdm
from bert_encoder import bert_forward
from transformers import BertModel, BertTokenizer
import time

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("USING CUDA")
else:
    device = torch.device('cpu')

def setup(n_enc_layer = 1):
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    codec = get_encoder()
    config = GPT2Config(n_enc_layer = n_enc_layer)
    model = GPT2LMHeadModel(config)
    if not os.path.exists('../gpt2-pytorch_model.bin'):
        print("Downloading GPT-2 checkpoint...")
        url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
        r = requests.get(url, allow_redirects=True)
        open('../gpt2-pytorch_model.bin', 'wb').write(r.content)

    model = load_weight(model, torch.load('../gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    return codec, model, config

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--current_ctx_length", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--n_enc_layer", type=int, default=1)
    parser.add_argument("--test_path", type=str, default="../../data/test.csv")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--wte_lr", type=bool, default=False)
    parser.add_argument("--best", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="../../outputs/bergpt_init_embed")
    parser.add_argument("--model_path", type=str, default="../../saved_models/bergpt_init_embed")
    # parser.add_argument("--trial", type=int, default=500)
    args = parser.parse_args()
    if args.wte_lr:
        args.output_path = args.output_path + "_lr"
        args.model_path = args.model_path + "_lr"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "dev"))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.quiet is False:
        print(args)

    if args.prompt:
        filename = "bergpt_test_prompt.csv"
        args.batch_size = 1
    else:
        filename = "bergpt_test_noprompt.csv"

    codec, model, config = setup(n_enc_layer = args.n_enc_layer)

    # print(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth"))

    # if os.path.exists(os.path.join(args.model_path, "bergpt_best.pth")):
    #     print("================LOADING FROM BEST SAVED MODEL", "bergpt_best.pth")
    #     model.load_state_dict(torch.load(os.path.join(args.model_path, "bergpt_best.pth")))
    if args.best:
        if os.path.exists(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth")):
            print("================LOADING FROM BEST SAVED MODEL", str(args.n_enc_layer) + "_bergpt_best.pth")
            model.load_state_dict(torch.load(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth")))
            filename = "best_" + filename
    else:
        if os.path.exists(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")):
            print("================LOADING FROM BEST SAVED MODEL", str(args.n_enc_layer) + "_bergpt_newest.pth")
            model.load_state_dict(torch.load(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")))
            filename = "newest_" + filename

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    test_dataset = NewsDataset(path = args.test_path, ctx_length = args.current_ctx_length, codec = codec, start_from_zero = True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False
    )

    start_time = time.time()

    model.eval()

    with torch.no_grad():
        with open(os.path.join(args.output_path, filename), "w") as f:
            fieldnames = ["highlight", "reference", "generated"]
            dev_writer = csv.DictWriter(f, fieldnames = fieldnames)
            dev_writer.writeheader()
            for dev_i, dev_data in enumerate(test_loader):
                if dev_i % 10 == 0:
                    print(dev_i, len(test_loader), time.time()-start_time)
                highlights = bert_forward(bert_model, bert_tokenizer, dev_data["highlight"], device).to(device)
                highlights = highlights.unsqueeze(1)
                preclude_length = 0
                if args.prompt:
                    context = codec.encode(dev_data["highlight"][0][:1024])
                    highlights = highlights.repeat(1, len(context), 1)
                    out = sample_sequence(
                        model=model, length=args.gen_length,
                        context = context,
                        highlights = highlights,
                        temperature=args.temperature, top_k=args.top_k, device=device
                    )
                    preclude_length = highlights.shape[0]
                else:
                    out = sample_sequence(
                        model=model, length=args.gen_length,
                        start_token = 50256,
                        highlights = highlights,
                        temperature=args.temperature, top_k=args.top_k, device=device
                    )
                    preclude_length = 1
                out = out.tolist()
                for j in range(len(out)):
                    text = codec.decode(out[j][preclude_length:])
                    stories = codec.decode(dev_data["story"][j].tolist())
                    dev_writer.writerow({"highlight": dev_data["highlight"][j], "generated" : text, "reference": stories})
                # break

    print("Finished Generating")

def ppl():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--current_ctx_length", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--n_enc_layer", type=int, default=1)
    parser.add_argument("--test_path", type=str, default="../../data/test.csv")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--prompt", type=bool, default=False)
    parser.add_argument("--wte_lr", type=bool, default=False)
    parser.add_argument("--best", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="../../outputs/bergpt_init_embed")
    parser.add_argument("--model_path", type=str, default="../../saved_models/bergpt_init_embed")
    # parser.add_argument("--trial", type=int, default=500)
    args = parser.parse_args()
    if args.wte_lr:
        args.output_path = args.output_path + "_lr"
        args.model_path = args.model_path + "_lr"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "dev"))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.quiet is False:
        print(args)

    if args.prompt:
        args.batch_size = 1

    codec, model, config = setup(n_enc_layer = args.n_enc_layer)

    # print(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth"))

    # if os.path.exists(os.path.join(args.model_path, "bergpt_best.pth")):
    #     print("================LOADING FROM BEST SAVED MODEL", "bergpt_best.pth")
    #     model.load_state_dict(torch.load(os.path.join(args.model_path, "bergpt_best.pth")))
    if args.best:
        if os.path.exists(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth")):
            print("================LOADING FROM BEST SAVED MODEL", str(args.n_enc_layer) + "_bergpt_best.pth")
            model.load_state_dict(torch.load(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth")))
    else:
        if os.path.exists(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")):
            print("================LOADING FROM BEST SAVED MODEL", str(args.n_enc_layer) + "_bergpt_newest.pth")
            model.load_state_dict(torch.load(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    test_dataset = NewsDataset(path = args.test_path, ctx_length = args.current_ctx_length, codec = codec, start_from_zero = True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False
    )

    start_time = time.time()

    model.eval()

    criterion = nn.CrossEntropyLoss(reduction = "sum")
    with torch.no_grad():
        devloss_total = 0
        for dev_i, dev_data in enumerate(test_loader):
            if dev_i % 10 == 0:
                print(dev_i, len(test_loader), time.time()-start_time)
            if args.prompt:
                highlight_input = np.array(codec.encode(dev_data["highlight"][0]))
                targets = torch.tensor(np.concatenate((highlight_input,dev_data["story"][0]))).unsqueeze(0).to(device)
            else:
                targets = torch.cat([torch.tensor(s).unsqueeze(0) for s in dev_data["story"]], dim = 0).to(device)
            highlights = bert_forward(bert_model, bert_tokenizer, dev_data["highlight"], device).to(device)
            highlights = highlights.unsqueeze(1).repeat(1, args.current_ctx_length + len(highlight_input), 1)
            # position_ids = torch.LongTensor(dev_data["pos_ids"]).to(device)
            lm_logits, presents = model(targets, highlights)
            # lm_logits, presents = model(targets, highlights, position_ids = position_ids)
            lm_logits = lm_logits[:,len(highlight_input):-1,:].reshape(-1, 50257)
            targets = targets[:, len(highlight_input)+1:].reshape(-1)
            devloss_total += criterion(lm_logits, targets).item()/ highlights.shape[0] / args.current_ctx_length
            # break
        print("=========TEST=========TESTLoss: {}, TESTPPL: {}=========".format(devloss_total/len(test_loader), np.exp(devloss_total/len(test_loader))))
        if args.best:
            with open(args.output_path +  "/" + str(args.prompt) + "_test_best_log.txt", "a") as logfile:
                logfile.write("{},{}\n".format(devloss_total/len(test_loader), np.exp(devloss_total/len(test_loader))))
        else:
            with open(args.output_path +  "/" + str(args.prompt) + "_test_newest_log.txt", "a") as logfile:
                logfile.write("{},{}\n".format(devloss_total/len(test_loader), np.exp(devloss_total/len(test_loader))))

    print("Finished Testing PPL")

if __name__ == '__main__':
    # test()
    ppl()
