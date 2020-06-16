import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from lxml import etree
import os
import pymorphy2
import re

def get_key(word_id):
    for key,val in word_to_ix.items():
        if(val == word_id):
            print(key)

def cluster_embeddings(filename,nclusters):
    X = np.load(filename)
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(X)
    center = kmeans.cluster_centers_
    distances = euclidean_distances(X,center)

    for i in np.arange(0,distances.shape[1]):
        word_id = np.argmin(distances[:,i])
        print(word_id)
        get_key(word_id)

def read_data(files):
    data = ''
    for file in files:
        with open(file[0]) as fobj:
            xml = fobj.read()
            descs = extractDesc(xml)
            claims = extractClaims(xml)
            data = data + ' ' + ' '.join(list(filter(None, descs))) 
    tokenizer = RegexpTokenizer(r'[а-я]+\-*[а-я]+')
    tokenized_data = tokenizer.tokenize(data.lower())
    stop_words = set(stopwords.words('russian'))
    stop_words.update(['.',',',':',';','(',')','#','--','...','"'])
    morph = pymorphy2.MorphAnalyzer()
    cleaned_words = [ morph.parse(i)[0].normal_form for i in tokenized_data if i not in stop_words ]
    return(cleaned_words)

def getListOfPatentsXml(pathOfPatents):
    """
    Поиск хмл файлов
    Аргументы:
    pathOfPatents - путь директории содержащей хмл файлы
    Возвращаемое значение:
    listOfPatentsXml - список всех путей к файлам хмл
    """
    listOfPatentsXml = []
    for file in os.listdir(pathOfPatents):
        path = os.path.join(pathOfPatents, file)
        if (os.path.isdir(path) == False):
            if (path.find(".xml") > 0):
                listOfPatentsXml.append(path)
        else:
            listOfPatentsXml += getListOfPatentsXml(path)
    return listOfPatentsXml
    
def extractDesc(data):
    """
    Поиск в тексте информации о патентной формуле изобретения
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    descs - список описаний изобретения
    """ 
    presence_dgh = ["для ", "при "]
    descs = []
    txt = ""
    sent = ""
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//description") is not None):
        for desc in root.find(".//description"):
            ind = 0
            txt = str(desc.text).lower()
            if txt.find("изобретение") != -1 and txt.find("относится") != -1:
                for el in presence_dgh:
                    ind = txt.find(el)
                    if ind != -1:
                        descs.append(txt)
                if txt.find("Результат") != -1:
                    ind = txt.find("в ")
                    if ind != -1:
                        descs.append(txt)
    return descs

def extractClaims(data):
    """
    Поиск в тексте информации о патентной формуле изобретения
    Аргументы:
    data - текст патента
    Возвращаемое значение:
    claims - список патентных формул изобретения
    """ 
    claims = []
    root = etree.fromstring(data.encode('utf-8'))
    if (root.find(".//claims") is not None):
        for claim in root.find(".//claims"):
            for el in claim:
                claims.append(el.text)
    # print("Формулы")
    # print(claims)
    return claims

class CBOWModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOWModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out1 = F.relu(self.linear1(embeds))
        out2 = self.linear2(out1)
        log_probs = F.log_softmax(out2, dim=1)
        return log_probs

    def predict(self,input):
        context_idxs = torch.tensor([word_to_ix[w] for w in input], dtype=torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        res_val = res_val[0][:3]
        res_ind = res_ind[0][:3]
        for arg in zip(res_val,res_ind):
            print([(key,val,arg[0]) for key,val in word_to_ix.items() if val == arg[1]])

    def freeze_layer(self,layer):
        for name,child in model.named_children():
            print(name,child)
            if(name == layer):
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())
                    params.requires_grad= False

    def print_layer_parameters(self):
        for name,child in model.named_children():
                print(name,child)
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())

    def write_embedding_to_file(self,filename):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename,weights)

def train_cbow(files):
    test_sentence = read_data(files)
    print(test_sentence)
    from gensim.models import Word2Vec
    model_w2v = Word2Vec([test_sentence], min_count = 1)
    model_w2v.save("word2vec.model")

    ngrams = []
    for i in range(len(test_sentence) - CONTEXT_SIZE):
        tup = [test_sentence[j] for j in np.arange(i , i + CONTEXT_SIZE) ]
        ngrams.append((tup,test_sentence[i + CONTEXT_SIZE]))

    vocab = set(test_sentence)
    print("Length of vocabulary",len(vocab))
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    losses = []
    loss_function = nn.NLLLoss()
    model = CBOWModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(400):
        total_loss = 0

        for context, target in ngrams:

            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            model.zero_grad()

            log_probs = model(context_idxs)

            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(total_loss)
        losses.append(total_loss)

    torch.save(model.state_dict(), 'model1.pt')
    with open('config1.txt', 'w') as file:
        line = str(EMBEDDING_DIM) + " " + str(CONTEXT_SIZE) + "\n"
        file.write(line)
        for word in vocab:
            file.write('%s ' % word)
    return model, word_to_ix

def load_cbow():
    lines = []
    with open('config1.txt', 'r') as file:
        for line in file:
            lines.append(line)
    if len(lines) == 0:
        return None, []
    EMBEDDING_DIM, CONTEXT_SIZE = lines[0].split(' ')
    EMBEDDING_DIM = int(EMBEDDING_DIM)
    CONTEXT_SIZE = int(CONTEXT_SIZE)
    print(str(EMBEDDING_DIM), str(CONTEXT_SIZE))
    vocab = lines[1].split(' ')
    del vocab[-1]
    print("Length of vocabulary",len(vocab))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    model = CBOWModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    model.load_state_dict(torch.load('model1.pt'))
    model.eval()
    return model, word_to_ix

def get_cbow_cosine(model, word_to_ix, word1, word2):
    lookup_tensor1 = torch.tensor([word_to_ix[word1]], dtype=torch.long)
    x = model.embeddings(lookup_tensor1)
    lookup_tensor2 = torch.tensor([word_to_ix[word2]], dtype=torch.long)
    y = model.embeddings(lookup_tensor2)
    arr = torch.cosine_similarity(x, y).tolist()
    return arr[0]

vword = []
with open('333333.txt', 'r') as file:
    for line in file:
        vword.append(line[:-1])

torch.manual_seed(1)
CONTEXT_SIZE = 3
EMBEDDING_DIM = 10
morph = pymorphy2.MorphAnalyzer()
if __name__ == "__main__":
    train_cbow('/home/anatoly/Загрузки/Выборки данных/DATA (С15)')
