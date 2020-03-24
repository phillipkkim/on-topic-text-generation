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
from model import (GPT2LMHeadModel)
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


def text_generator(state_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    print(args.text)
    context_tokens = enc.encode(args.text)

    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = enc.decode(out[i])
            if args.quiet is False:
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument('--unconditional', type=bool, default=False, help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    codec, model, config = setup()
    dataset = NewsDataset(path=data_folder)

    if args.batch_size == -1:
        args.batch_size = 1

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,  # keep this fixed at 1, and use chunksize in dataset instead
        num_workers=8,  # try to match num cpu
        shuffle=False
    )

    for i,data in enumerate(loader):#TODO
        ## Use paper abstracts as the starting text
        # if i % 100 == 0:
        print("Generating", i, "/", len(loader))
            # print(data[0][2][0])
        start_text = data["highlight"]
        # start_text = [start_text[j] + data["prompt"][j] for j in range(args.batch_size)]
        # print(start_text)
        # start_text = [codec.encode(start_text[j]) for j in range(args.batch_size)]
        start_text = codec.encode(start_text[0])
        # print(start_text)
        out = sample_sequence(
            model=model, length=args.length,
            context=start_text  if not  args.unconditional else None,
            start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
            gen_batch_size=args.gen_batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(start_text):].tolist()

        for j in range(args.batch_size):
            text = codec.decode(out[j])
            with open("../../output/original_gpt2/" + str(i+j) + ".txt", "w") as f:
                # f.write(data["prompt"][j] + " " + text)
                f.write(text)
        # if i % 100 == 0:
        #     print(text)
        # break

    print("# Samples written to original-gpt2")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--current_ctx_length", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--n_enc_layer", type=int, default=1)
    parser.add_argument("--train_path", type=str, default="../../data/train.csv")
    parser.add_argument("--dev_path", type=str, default="../../data/dev.csv")
    parser.add_argument("--test_path", type=str, default="../../data/test.csv")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--gen_freq", type=int, default=1)
    parser.add_argument("--ppl_freq", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=250)
    parser.add_argument("--wte_grad", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="../../outputs/bergpt_wte_grad/")
    parser.add_argument("--model_path", type=str, default="../../saved_models/bergpt_wte_grad/")
    # parser.add_argument("--trial", type=int, default=500)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(os.path.join(args.output_path, "dev"))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if args.quiet is False:
        print(args)

    codec, model, config = setup(n_enc_layer = args.n_enc_layer)

    # print(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_best.pth"))

    # if os.path.exists(os.path.join(args.model_path, "bergpt_best.pth")):
    #     print("================LOADING FROM BEST SAVED MODEL", "bergpt_best.pth")
    #     model.load_state_dict(torch.load(os.path.join(args.model_path, "bergpt_best.pth")))
    if os.path.exists(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")):
        print("================LOADING FROM MOST CURRENT SAVED MODEL", str(args.n_enc_layer) + "_bergpt_newest.pth")
        model.load_state_dict(torch.load(os.path.join(args.model_path, str(args.n_enc_layer) + "_bergpt_newest.pth")))

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "enc_h" in name:
            # print(name)
            param.requires_grad = True
        elif "wte.weight" in name and args.wte_grad:
            print(name)
            param.requires_grad = True
        else:
            for p in range(config.n_layer - config.n_enc_layer + 1, config.n_layer):
                if "h."+str(p) in name:
                    # print(name)
                    param.requires_grad = True


    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained('bert-base-cased')

    train_dataset = NewsDataset(path = args.train_path, ctx_length = args.current_ctx_length, codec = codec, start_from_zero = False)
    dev_dataset = NewsDataset(path = args.dev_path, ctx_length = args.current_ctx_length, codec = codec, start_from_zero = True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True
    )

    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True
    )
    criterion = nn.CrossEntropyLoss(reduction = "sum")
    optimizer = optim.AdamW(model.parameters(), weight_decay=0.005)

    best_loss = 1000000000
    start_time = time.time()

    for epoch in range(args.start_epoch, args.start_epoch + args.max_epoch):
        for i,data in enumerate(train_loader):
            # highlights = data["highlight"]
            # stories = data["story"]
            # start_text = [codec.encode(start_text[j]) for j in range(args.batch_size)]
            # for s in data["story"]:
            #     print(len(s))
            targets = torch.cat([torch.tensor(s).unsqueeze(0) for s in data["story"]], dim = 0).to(device)
            highlights = bert_forward(bert_model, bert_tokenizer, data["highlight"], device).to(device)
            highlights = highlights.unsqueeze(1).repeat(1, args.current_ctx_length, 1)
            position_ids = torch.LongTensor(data["pos_ids"]).to(device)
            optimizer.zero_grad()
            # lm_logits, presents = model(targets, highlights)
            lm_logits, presents = model(targets, highlights, position_ids = position_ids)
            lm_logits = lm_logits[:,:-1,:].reshape(-1, 50257)
            targets = targets[:, 1:].reshape(-1)
            loss = criterion(lm_logits, targets)
            loss.backward()
            optimizer.step()

            if i % args.print_freq == 0:
                avgloss = loss.item() / highlights.shape[0] / (args.current_ctx_length-1)
                print("=========Epoch: {}, Batch: {}, AVGLoss: {}, PPL: {}, Time: {}=========".format(\
                    epoch, i, avgloss, np.exp(avgloss), time.time()-start_time))
                with open(args.output_path + str(config.n_enc_layer) + "_train_log.txt", "a") as logfile:
                    logfile.write("{},{},{},{}\n".format(epoch, i, avgloss, np.exp(avgloss)))
            if i % args.save_freq == 0:
                torch.save(model.state_dict(), args.model_path + str(args.n_enc_layer) + "_bergpt_newest.pth")
                avgloss = loss.item() / highlights.shape[0] / (args.current_ctx_length-1)
                if best_loss > avgloss:
                    best_loss = avgloss
                    torch.save(model.state_dict(), args.model_path + str(args.n_enc_layer) + "_bergpt_best.pth")
            # break

        torch.save(model.state_dict(), args.model_path + str(args.n_enc_layer) + "_bergpt_newest.pth")
        avgloss = loss.item() / highlights.shape[0] / (args.current_ctx_length-1)
        if best_loss > avgloss:
            best_loss = avgloss
            torch.save(model.state_dict(), args.model_path + str(args.n_enc_layer) + "_bergpt_best.pth")


        if epoch % args.gen_freq == 0:
            with torch.no_grad():
                model.eval()
                with open(args.output_path + "dev/" + str(epoch) + ".csv", "w") as f:
                    fieldnames = ["highlights", "generated story"]
                    dev_writer = csv.DictWriter(f, fieldnames = fieldnames)
                    dev_writer.writeheader()
                    for dev_i, dev_data in enumerate(dev_loader):
                        highlights = bert_forward(bert_model, bert_tokenizer, dev_data["highlight"], device).to(device)
                        highlights = highlights.unsqueeze(1)
                        out = sample_sequence(
                            model=model, length=args.gen_length,
                            start_token = 50256,
                            highlights = highlights,
                            temperature=args.temperature, top_k=args.top_k, device=device
                        )
                        out = out.tolist()
                        for j in range(len(out)):
                            text = codec.decode(out[j][1:])
                            dev_writer.writerow({"highlights": dev_data["highlight"][j], "generated story" : text})
                        # break
        if epoch % args.ppl_freq == 0:
            with torch.no_grad():
                model.eval()
                devloss_total = 0
                for dev_i, dev_data in enumerate(dev_loader):
                    targets = torch.cat([torch.tensor(s).unsqueeze(0) for s in dev_data["story"]], dim = 0).to(device)
                    highlights = bert_forward(bert_model, bert_tokenizer, dev_data["highlight"], device).to(device)
                    highlights = highlights.unsqueeze(1).repeat(1, args.current_ctx_length, 1)
                    position_ids = torch.LongTensor(dev_data["pos_ids"]).to(device)
                    # lm_logits, presents = model(targets, highlights)
                    lm_logits, presents = model(targets, highlights, position_ids = position_ids)
                    lm_logits = lm_logits[:,:-1,:].reshape(-1, 50257)
                    targets = targets[:, 1:].reshape(-1)
                    devloss_total += criterion(lm_logits, targets).item()/ highlights.shape[0] / args.current_ctx_length
                    # break
                print("=========DEV=========Epoch: {}, DEVLoss: {}, DEVPPL: {}=========".format(epoch, devloss_total/len(dev_loader), np.exp(devloss_total/len(dev_loader))))
                with open(args.output_path + "grad_" + str(config.n_enc_layer) + "_dev_log.txt", "a") as logfile:
                    logfile.write("{},{},{}\n".format(epoch, devloss_total/len(dev_loader), np.exp(devloss_total/len(dev_loader))))

        model.train()
        # break

    print("Finished Training")

if __name__ == '__main__':
    train()
