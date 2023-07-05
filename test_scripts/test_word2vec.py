from gensim.models import Word2Vec
import gensim.downloader
import os
import argparse

def main(models, model_to_load, model_folder):
    if models:
        print('\n'.join(list(gensim.downloader.info()['models'].keys())))
        return

    print("loading Word2Vec model....")
    if model_to_load is not None:
        model = gensim.downloader.load(model_to_load)
    else:
        model = Word2Vec.load(os.path.join(model_folder, "word2vec.model")).wv
    print("model loaded!")

    while True:
        print("Enter a word to get similar words for. (Hit enter without typing a word to stop.)")
        word = input().split(' ')[0].lower()
        if len(word)>0:
            print(model.most_similar(word)) # print(model[word])
        else:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action='store_true', help="Use this flag to print the models that can be downloaded from gensim.")
    parser.add_argument("--model_name", dest = "model_to_load", default=None, help="Name of model to download.")
    parser.add_argument("--model_folder", dest = "model_folder", default = "models", help="Folder for custom Word2Vec model.")
    parser.set_defaults(models=False)
    args = parser.parse_args()
    main(args.models, args.model_to_load, args.model_folder)