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

def main():
    path = os.path.join("../../output", "original_gpt2")
    target_path = os.path.join("../../data", "test/stories")
    length = len(os.listdir(path))

    common_dictionary = Dictionary(common_texts)

    model_path = os.path.join(os.getcwd(), "..", "lda_model", "model")
    lda = models.ldamodel.LdaModel.load(model_path)

    jsd_sum = 0.0

    for i, file in enumerate(tqdm(os.listdir(path))):
    # for file in os.listdir(path):
        with open(os.path.join(path, file), "r") as f:
            text = f.read()
            text_vector = np.zeros(10)
            text_tokenized = text.split()[:512]
            text_processed = common_dictionary.doc2bow(text_tokenized)
            for elem in lda[text_processed]:
                text_vector[elem[0]] += elem[1]
            # text_vector = [elem[1] for elem in lda[text_processed]]
            # print(text)
            # print(text_vector)

        with open(os.path.join(target_path, file), "r") as t:
            target_text = t.read()
            target_vector = np.zeros(10)
            target_tokenized = target_text.split()[:512]
            target_processed = common_dictionary.doc2bow(target_tokenized)
            for elem in lda[target_processed]:
                target_vector[elem[0]] += elem[1]
            # target_vector = [elem[1] for elem in lda[target_processed]]
            # print(len(target_vector))

        jsd_sum += distance.jensenshannon(text_vector, target_vector)
        # print(jsd_sum)
        # break

    print(jsd_sum / length)



if __name__ == "__main__":
    main()
