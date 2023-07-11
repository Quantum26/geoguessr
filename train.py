from gensim.models import Word2Vec
from word2vec import load_word2vec
from models.model_cnn import SentimentModel
import torch.nn as nn
from data_processing import minibatch_generator

import torch.optim as optim

print("loading Word2Vec....")
w2vmodel = load_word2vec()
print("Word2Vec loaded!")

def embedding_layer(x):
    return w2vmodel.wv[x]

model = SentimentModel(embedding_layer)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

running_loss = 0.0
for i, data in enumerate(minibatch_generator(minibatch=5)):
    # get the inputs; data is a dictionary of lists {"labels":[], "data":[]}
    labels = data["labels"]
    inputs = data["data"]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')
