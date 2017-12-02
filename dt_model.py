import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import random
from torch.nn.init import *
import numpy as np
import crf_param

def Parameter(shape=None, init=xavier_uniform):
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))

def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))

class RNNState():
    def __init__(self, cell, hidden=None):
        self.cell = cell
        self.hidden = hidden
        if not hidden:
            self.hidden = Variable(torch.zeros(1, self.cell.hidden_size)), \
                          Variable(torch.zeros(1, self.cell.hidden_size))

    def next(self, input):
        return RNNState(self.cell, self.cell(input, self.hidden))

    def __call__(self):
        return self.hidden[0]

class dependency_tagging_model(nn.Model):
    def _init_(self,vocab,pos,w2i,tag_map,options):
        super(dependency_tagging_model, self).__init__()
        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu,}
        self.activation = self.activations[options.activation]
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.tdims = options.tembedding_dims
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 3 for ind, word in enumerate(pos)}
        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                       external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                np_emb[i] = self.external_embedding[word]
            self.elookup = nn.Embedding(*np_emb.shape)
            self.elookup.weight = Parameter(init=np_emb)
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim
        self.builders = [
            nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
            nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]
        for i, b in enumerate(self.builders):
            self.add_module('builder%i' % i, b)

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab) + 3, self.wdims)
        self.plookup = nn.Embedding(len(pos) + 3, self.pdims)

        tag_pointer = 0
        tag_num = 0
        self.tag_table = {}
        for t in tag_map.keys():
            tag_size = tag_map[t]
            tag_num += tag_size
            tag_list = []
            for s in range(tag_size):
                tag_pointer += 1
                tag_list.append(tag_pointer)
            self.tag_table[t] = tag_list


        self.tlookup = nn.Embedding(tag_num,self.tdims)



        self.feature_param = crf_param.init_feat_param()

        def forward(self,sentence):
            for entry in sentence:
                c = float(self.wordsCount.get(entry.norm, 0))
                dropFlag = (random.random() < (c / (0.25 + c)))
                wordvec = self.wlookup(scalar(
                    int(self.vocab.get(entry.norm, 0)) if dropFlag else 0)) if self.wdims > 0 else None
                posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
                evec = None
                if self.external_embedding is not None:
                    evec = self.elookup(scalar(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (
                        dropFlag or (random.random() < 0.5)) else 0))
                entry.vec = torch.cat([wordvec, posvec, evec])
                entry.lstms = [entry.vec, entry.vec]
                entry.unary_potential = None

                lstm_forward = RNNState(self.builders[0])
                lstm_backward = RNNState(self.builders[1])

                for entry, rentry in zip(sentence, reversed(sentence)):
                    lstm_forward = lstm_forward.next(entry.vec)
                    lstm_backward = lstm_backward.next(rentry.vec)

                    entry.lstms[1] = lstm_forward()
                    rentry.lstms[0] = lstm_backward()

                entry.feats = crf_param.get_feats(entry.pos,self.tag_table)











