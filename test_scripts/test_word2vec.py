from gensim.models import Word2Vec
import gensim.downloader

print("loading Word2Vec model....")
model = gensim.downloader.load('glove-twitter-50')
print("model loaded!")

while True:
    print("Enter a word to get similar words for. (Hit enter without typing a word to stop.)")
    word = input().split(' ')[0].lower()
    if len(word)>0:
        print(model.most_similar(word))
    else:
        break