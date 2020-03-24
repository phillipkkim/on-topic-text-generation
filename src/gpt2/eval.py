import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from main import setup
from rouge import Rouge
from collections import Counter
from tqdm import tqdm
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def nll_calc(model, codec, highlight, text):
    """
    Compute the negative log-likelihoods for a string `text`
    :param model: The GPT-2 model
    :param texts: A tensor of shape (1, T), where T is the length of the text
    :return: The log-likelihood. It should be a Python scalar.
        NOTE: for simplicity, you can ignore the likelihood of the first token in `text`.
    """


    past = None
    loss = nn.CrossEntropyLoss(reduction = "sum")
    with torch.no_grad():
        ##  1) Compute the logits from `model`;
        ##  2) Return the log-likelihood of the `text` string. It should be a Python scalar.
        ##      NOTE: for simplicity, you can ignore the likelihood of the first token in `text`
        text = codec.encode(text)[:512]
        length = len(text) - 1
        highlight = codec.encode(highlight)

        ## with prompt
        # text = highlight + text
        # text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        # logits, past = model(text, past=past)
        # logits = logits[:,len(highlight):-1,:].view(-1, 50257)
        # text = text[:,len(highlight)+1:].view(-1,)

        ## without prompt
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        logits, past = model(text, past=past)
        logits = logits[:,:-1,:].view(-1, 50257)
        text = text[:,1:].view(-1,)

        nll =  float(loss(logits, text))
        ppl = np.exp(nll / length)
    return ppl

def main():
    codec, model, _, config = setup('../../data/test.csv')

    path = os.path.join("../../outputs", "gpt2_generated.csv")
    # path = os.path.join("../../outputs", "gpt2_with_prompt.csv")

    rouge = Rouge()
    nlls = 0.0
    ppls = 0.0
    rouges = {}
    length = 0

    with open(path, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            row = dict(row)
            text = row['generated']
            highlight_text = row['highlight']
            target_text = row['reference']
            ppl = nll_calc(model, codec, highlight_text, target_text)
            ppls += ppl
            rouge_score = rouge.get_scores(text, target_text, avg=True)
            if not rouges:
                for cat in rouge_score.keys():
                    rouges[cat] = Counter(rouge_score[cat])
            else:
                for cat in rouge_score.keys():
                    rouges[cat] += Counter(rouge_score[cat])
            length += 1
            print(length, '/', 4595)
    # print(np.exp(nlls / (511*length)))
    print("perplexity =", ppls/length)
    for cat in rouges.keys():
        for key in rouges[cat].keys():
            rouges[cat][key] /= length
    print("rouges =", rouges)

if __name__ == "__main__":
    main()
