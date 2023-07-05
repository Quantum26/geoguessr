import os
from gensim.models import Word2Vec
from data_processing import generate_line_list

def train(model_folder = "models"):
    model = Word2Vec(sentences=generate_line_list(load=True))
    model.save(os.path.join(model_folder, "word2vec.model"))

def load_word2vec(model_folder = "models"):
    model = Word2Vec.load(os.path.join(model_folder, "word2vec.model"))
    return model

if __name__ == "__main__":
    train()