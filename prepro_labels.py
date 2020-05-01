import unicodedata
import re
import json
import torch
import torch.nn as nn
from opts import opts
opt = opts
# SOS_token = 0
# EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS


    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readCaps(input_file):
    print("Reading annotations...")
    data = json.load(open(input_file,'r'))
    annotations = data['annotations']

    pairs = [[normalizeString(s).strip('. ') for s in ann['split']]
             +[normalizeString(ann['caption']).strip('. ')] + [ann['image_id_true']] \
             for ann in annotations]
    
    image_ids = [ann['image_id_true'] for ann in annotations]
    lang = Lang('lang')
    
    return lang, pairs, image_ids


def filterPair(p):
    return len(p[0].split(' ')) < opt.MAX_LENGTH/2 and len(p[1].split(' ')) < opt.MAX_LENGTH/2


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(input_file, MAX_LENGTH):
    lang, pairs, image_ids = readCaps(input_file)
    print("Read %s captions" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[2])
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, pairs, image_ids


def add_to_lang(input_file, lang):
    _, pairs, image_ids = readCaps(input_file)
    print("Read %s captions" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        lang.addSentence(pair[2])
    print("Counted words:")
    print(lang.name, lang.n_words)
    return lang, image_ids


def embed(lang, word_vector):
    emb_dim = 300
    emb_size = lang.n_words
    print('number of words: %d'% lang.n_words)
    w2v = torch.zeros([emb_size, emb_dim], dtype=torch.float32)
    nn.init.kaiming_normal_(w2v)
    for i in range(lang.n_words):
        word = lang.index2word[i]
        if word in word_vector.vocab:
            w2v[i,:] = torch.tensor(word_vector[word])
            
    return w2v