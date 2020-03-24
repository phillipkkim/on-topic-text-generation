import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
from scipy.spatial import distance
from main import setup
from tqdm import tqdm
import csv


def main():
    # path = os.path.join("../../outputs", "gpt2_generated.csv")
    path = os.path.join("../../outputs", "gpt2_with_prompt.csv")
    length = 0

    # dct = Dictionary(common_texts)

    # model_path = os.path.join(os.getcwd(), "..", "lda_model", "model")
    # lda = models.ldamodel.LdaModel.load(model_path)

    lda = models.ldamodel.LdaModel.load(os.path.join("../lda_model", "model"))
    dct = Dictionary.load_from_text(os.path.join("../lda_model", "dictionary"))

    jsd_sum = 0.0

    with open(path, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            row = dict(row)
            text = row['generated']
            target_text = row['reference']
            text_vector = np.zeros(50)
            # text_tokenized = text.split()[:512]
            text_tokenized = text.split()[:80]
            text_processed = dct.doc2bow(text_tokenized)
            for elem in lda[text_processed]:
                text_vector[elem[0]] += elem[1]

            target_vector = np.zeros(50)
            # target_tokenized = target_text.split()[:512]
            target_tokenized = target_text.split()[:80]
            target_processed = dct.doc2bow(target_tokenized)
            for elem in lda[target_processed]:
                target_vector[elem[0]] += elem[1]
            length += 1

            jsd_sum += distance.jensenshannon(text_vector, target_vector)
    print(jsd_sum / length)



if __name__ == "__main__":
    main()
