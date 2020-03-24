'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
sys.path.append("..")
import torch
import random
import argparse
import numpy as np
from model import (GPT2LMHeadModel)
from utils import load_weight
from config import GPT2Config
from sample import sample_sequence
from encoder import get_encoder
from csv_dataset import NewsDataset
# from news_dataset import NewsDataset
import requests
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def setup(data_folder):
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    codec = get_encoder()

    dataset = NewsDataset(path=data_folder, ctx_length = 128, codec = codec, start_from_zero = True)

    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    if not os.path.exists('gpt2-pytorch_model.bin'):
        print("Downloading GPT-2 checkpoint...")
        url = 'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin'
        r = requests.get(url, allow_redirects=True)
        open('gpt2-pytorch_model.bin', 'wb').write(r.content)

    model = load_weight(model, torch.load('gpt2-pytorch_model.bin', map_location=device))
    model = model.to(device)
    model.eval()
    return codec, model, dataset, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", type=bool, default=False)
    parser.add_argument('--unconditional', type=bool, default=False, help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gen_batch_size", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if args.quiet is False:
        print(args)

    codec, model, dataset, config = setup('../../data/test.csv')

    if args.batch_size == -1:
        args.batch_size = 1

    if args.length == -1:
        # args.length = config.n_ctx // 2
        args.length = 128
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,  # keep this fixed at 1, and use chunksize in dataset instead
        num_workers=8,  # try to match num cpu
        shuffle=False
    )

    with open('../../outputs/gpt2_generated.csv', mode='w') as csv_file:
        fieldnames = ['generated', 'reference', 'highlight']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i,data in enumerate(loader):#TODO
            ## Use paper as the starting text
            # if i % 100 == 0:
            print("Generating", i, "/", len(loader))
                # print(data[0][2][0])
            # start_text = data["highlight"]
            # start_text = [start_text[j] + data["prompt"][j] for j in range(args.batch_size)]
            # print(start_text)
            # start_text = [codec.encode(start_text[j]) for j in range(args.batch_size)]
            # start_text = codec.encode(start_text[0])
            # print(start_text)
            out = sample_sequence(
                model=model, length=args.length,
                context= codec.encoder['<|endoftext|>'],
                start_token=enc.encoder['<|endoftext|>'] if args.unconditional else None,
                gen_batch_size=args.gen_batch_size,
                temperature=args.temperature, top_k=args.top_k, device=device
            )
            out = out[:, 1:].tolist()

            for j in range(args.batch_size):
                text = codec.decode(out[j])
                print(len(text.split()))
                writer.writerow({'generated': text, 'reference': codec.decode(dataset[i]['story']), 'highlight': data['highlight'][0]})

                # with open("../../output/" + str(i+j) + ".txt", "w") as f:
                #     # f.write(data["prompt"][j] + " " + text)
                #     f.write(text)
            # if i % 100 == 0:
            #     print(text)
            # break

    print("# Samples written to original-gpt2")

'''
Ignore this function, we are not using it, but you can use it for understanding the functionality of GPT sampling
'''
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

if __name__ == '__main__':
    # if os.path.exists('gpt2-pytorch_model.bin'):
    #     state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
    #     text_generator(state_dict)
    # else:
    #     print('Please download gpt2-pytorch_model.bin')
    #     sys.exit()
    main()
