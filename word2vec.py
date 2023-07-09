import os, argparse
from gensim.models import Word2Vec
from data_processing import line_list_generator

def train(model_folder = "models", num_files=50):
    data_gen = line_list_generator(load=True, max_files=num_files)
    print("Started Training")
    print("Training on file 1", end="\r")
    model = Word2Vec(sentences=next(data_gen))
    for n, data in enumerate(data_gen):
        print("Training on file " + str(n+2) + ".", end='\r')
        model.train(data, total_examples=len(data), epochs=1)

    model.save(os.path.join(model_folder, "word2vec.model"))

def load_word2vec(model_folder = "models"):
    model = Word2Vec.load(os.path.join(model_folder, "word2vec.model"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_files", dest="num_files", default=50, help="Number of files to train on. Each file contains 100,000 entries.")
    args = parser.parse_args()
    train(num_files=int(args.num_files))