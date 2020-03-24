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
        text = highlight + text
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        logits, past = model(text, past=past)
        logits = logits[:,len(highlight):-1,:].view(-1, 50257)
        text = text[:,len(highlight)+1:].view(-1,)
        nll =  float(loss(logits, text))
        # ppl = np.exp(nll / length)
    return nll

def main():
    codec, model, _, config = setup('../../data/test')
    path = os.path.join("../../output", "original_gpt2")
    target_path = os.path.join("../../data", "test/stories")
    highlight_path = os.path.join("../../data", "test/highlights")
    rouge = Rouge()
    nlls = 0.0
    rouges = {}
    length = len(os.listdir(path))
    for i, file in enumerate(tqdm(os.listdir(path))):
    # for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            text = f.read()
            # ppl = ppl_calc(model, codec, text)
            # print(ppl)
            # ppls.append(ppl)
        with open(os.path.join(target_path, file), "r") as t, open(os.path.join(highlight_path, file), "r") as h:
            target_text = t.read()
            highlight_text = h.read()
            nlls += nll_calc(model, codec, highlight_text, target_text)
            # print(nlls)
            # print(ppl)
            # nlls.append(nll)
            rouge_score = rouge.get_scores(text, target_text, avg=True)
            if not rouges:
                for cat in rouge_score.keys():
                    rouges[cat] = Counter(rouge_score[cat])
            else:
                for cat in rouge_score.keys():
                    rouges[cat] += Counter(rouge_score[cat])
    print(np.exp(nlls / (511*length)))
    for cat in rouges.keys():
        for key in rouges[cat].keys():
            rouges[cat][key] /= length
    print(rouges)

if __name__ == "__main__":
    main()
