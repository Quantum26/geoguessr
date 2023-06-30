from gensim.models import Word2Vec
import gensim.downloader
from model.model_cnn import SentimentModel
from data_funcs import data_generator

print("loading Word2Vec....")
model = gensim.downloader.load('glove-wiki-gigaword-50')
print("Word2Vec loaded!")

