import json
import math
import torch
import numpy as np

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from config import word2vec_weights
from timer_utils import timer

class EpochSaver(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        model.save(".\\checkpoints\\OpTC\\w2v_optc.model")
        self.epoch += 1

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

class PositionalEncoder:
    def __init__(self, d_model, max_len=100000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe = torch.zeros(max_len, d_model)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def embed(self, x):
        return x + self.pe[:x.size(0)]

encoder = PositionalEncoder(20)


w2vmodel = Word2Vec.load(word2vec_weights)

def infer(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]

    if not word_embeddings:
        return np.zeros(20)

    output_embedding = torch.tensor(word_embeddings, dtype=torch.float)
    if len(document) < 100000:
        output_embedding = encoder.embed(output_embedding)

    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)

def prepare_sentences(df):
    nodes = {}
    for index, row in df.iterrows():
        for key in ["actorID", "objectID"]:
            node_id = row[key]
            nodes.setdefault(node_id, []).extend(row["phrase"])
    return list(nodes.values())

def train_word2vec_model(train_file_path):
    from event_processing import transform

    with open(train_file_path, "r") as file:
        content = [json.loads(line) for line in file]

    events = transform(content)
    phrases = prepare_sentences(events)

    logger = EpochLogger()
    saver = EpochSaver()
    word2vec = Word2Vec(
        sentences=phrases,
        vector_size=20,
        window=5,
        min_count=1,
        workers=8,
        epochs=300,
        callbacks=[saver, logger]
    )
    return word2vec