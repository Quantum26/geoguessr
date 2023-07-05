from gensim.models import Word2Vec
import gensim.downloader
from models.model_cnn import SentimentModel
from data_processing import data_generator

print("loading Word2Vec....")
model = gensim.downloader.load('glove-wiki-gigaword-50')
print("Word2Vec loaded!")

