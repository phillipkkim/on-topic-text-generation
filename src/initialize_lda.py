from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import models
import os
import csv

def initialize_lda():
    path = os.path.join("../data", "train.csv")
    dct = Dictionary(common_texts)
    corpus = [dct.doc2bow(text) for text in common_texts]

    with open(path, 'r') as file:
        csv_file = csv.DictReader(file)
        for row in csv_file:
            row = dict(row)
            new_texts = [row['story'].split()]
            dct.add_documents(new_texts)
            corpus += [dct.doc2bow(text) for text in new_texts]
    lda = models.ldamodel.LdaModel(corpus, num_topics=50)
    lda.save(os.path.join("lda_model", "model"))
    dct.save_as_text(os.path.join("lda_model", "dictionary"))

if __name__ == '__main__':
    initialize_lda()