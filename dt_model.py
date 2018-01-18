import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import random
from torch.nn.init import *
from torch import optim
import numpy as np
import param
import eisner_parser
import utils
import time


def Parameter(shape=None, init=xavier_uniform):
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)


def cat(l, dimension=-1):
    valid_l = filter(lambda x: x, l)
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


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


class dependency_tagging_model(nn.Module):
    def __init__(self, vocab, pos, w2i, tag_map, feats, sentences,options):
        super(dependency_tagging_model, self).__init__()
        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu, }
        self.activation = self.activations[options.activation]
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.tdims = options.tembedding_dims
        self.fdims = options.feat_param_dims
        self.ldims = options.lstm_dims
        self.sentences = sentences
        self.wordsCount = vocab
        self.vocab = {word: ind for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.external_embedding, self.edim = None, 0
        self.distdim = options.dist_dim
        self.batch = options.batchsize
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


            print 'Load external embedding. Vector dimensions', self.edim
        self.builders = [
            nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims),
            nn.LSTMCell(self.wdims + self.pdims + self.edim, self.ldims)]
        for i, b in enumerate(self.builders):
            self.add_module('builder%i' % i, b)

        # self.vocab['*PAD*'] = 1
        # self.pos['*PAD*'] = 1
        #
        # self.vocab['*INITIAL*'] = 2
        # self.pos['*INITIAL*'] = 2

        self.wlookup = nn.Embedding(len(vocab), self.wdims)
        self.plookup = nn.Embedding(len(pos), self.pdims)

        tag_pointer = 0
        tag_num = 0
        self.tag_table = {}
        self.max_tag_num = 0
        for t in tag_map.keys():
            tag_size = tag_map[t]
            tag_num += tag_size
            tag_list = []
            for s in range(tag_size):
                tag_list.append(tag_pointer)
                tag_pointer += 1
            self.tag_table[t] = tag_list
            if len(tag_list) > self.max_tag_num:
                self.max_tag_num = len(tag_list)
        self.tag_num = tag_num
        self.hidden_units = options.hidden_units
        self.tlookup = nn.Embedding(tag_num, self.hidden_units)
        self.flookup = feats

        self.feature_param,self.feat_embedding = param.init_feat_param(self.flookup)

        self.recons_param = np.zeros((self.tag_num,self.tag_num,self.distdim,2))
        self.lex_param = np.zeros((self.tag_num,len(self.vocab.keys())))

        self.hidLayer = Parameter((2 * self.ldims, self.hidden_units))
        self.hidBias = Parameter(self.hidden_units)
        self.subtag_use = options.subtag_use
        self.trainer = get_optim(options.optim, self.parameters())

    def evaluate(self, sentence):
        scores = np.zeros((sentence.size, sentence.size, self.max_tag_num, self.max_tag_num))
        scores.fill(-10000)
        crf_scores = np.zeros((sentence.size, sentence.size, self.max_tag_num, self.max_tag_num))
        crf_scores.fill(-10000)
        crf_scores = Variable(torch.FloatTensor(crf_scores))
        param.fire_feats(sentence, scores, crf_scores, self.feat_embedding,self.tag_table, self.flookup,
                                         self.recons_param, self.lex_param,self.vocab)

        return scores, crf_scores

    def get_loss(self,best_parse, partition_score,crf_scores):
        tree_loss = Variable(torch.FloatTensor([0]))

        for i, h in enumerate(best_parse[0]):
            if h == -1:
                continue
            h = int(h)
            m_tag_id = int(best_parse[1][i])
            h_tag_id = int(best_parse[1][h])
            tree_loss = tree_loss + crf_scores[h,i,h_tag_id,m_tag_id]
        #print 'tree_score',param.get_scalar(tree_loss, 0)
        tree_loss = -(tree_loss - partition_score)

        #print 'partition',param.get_scalar(partition_score, 0)
        return tree_loss

    def forward(self, sentence, best_trees):
        for entry in sentence.entries:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = (random.random() < (c / (0.25 + c)))
            wordvec = self.wlookup(scalar(
                int(self.vocab.get(entry.norm, 0)) if dropFlag else 0)) if self.wdims > 0 else None
            posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            evec = None
            if self.external_embedding is not None:
                evec = self.elookup(scalar(self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (
                    dropFlag or (random.random() < 0.5)) else 0))
            entry.vec = cat([wordvec, posvec, evec])
            entry.lstms = [entry.vec, entry.vec]
            entry.unary_potential = None

            lstm_forward = RNNState(self.builders[0])
            lstm_backward = RNNState(self.builders[1])

        for entry, rentry in zip(sentence.entries, reversed(sentence.entries)):
            lstm_forward = lstm_forward.next(entry.vec)
            lstm_backward = lstm_backward.next(rentry.vec)

            entry.lstms[1] = lstm_forward()
            rentry.lstms[0] = lstm_backward()

        for entry in sentence.entries:
            entry.hidden = torch.mm(cat([entry.lstms[0], entry.lstms[1]]), self.hidLayer) + self.hidBias
            tag_list = Variable(torch.LongTensor(self.tag_table.get(entry.pos)))
            tag_emb = torch.index_select(self.tlookup.weight, 0, tag_list)
            entry.unary_potential = self.activation(torch.mm(entry.hidden, torch.t(tag_emb)))
        scores, crf_scores = self.evaluate(sentence)
        if len(best_trees.keys()) == 0:
            best_parse = eisner_parser.parse_proj(scores)
        else:
            best_parse = best_trees[sentence]
        partition_score,inside_scores = eisner_parser.partition_inside(crf_scores)
        tree_loss = self.get_loss(best_parse, partition_score,crf_scores)

        return best_parse, tree_loss

    def train(self, sentences,best_trees):
        batch_loss = 0.0
        section_loss = 0.0
        etotal = 0
        epoch_loss = 0.0
        self.recons_counter = np.zeros((self.tag_num, self.tag_num,self.distdim,2))
        self.lex_counter = np.zeros((self.tag_num, len(self.vocab.keys())))
        start = time.time()
        random.shuffle(sentences)
        param.init_param(sentences,self.vocab,self.tag_table,self.recons_param,self.lex_param)
        for iSentence, sentence in enumerate(sentences):
             #print 'sentence',iSentence
            if iSentence%10==0 and iSentence!=0:
                print 'Loss',param.get_scalar(section_loss/etotal,0),'Time',time.time()-start
                start = time.time()
                section_loss = 0.0
                etotal = 0
            best_parse, tree_loss = self.forward(sentence, best_trees)
            param.counter_update(best_parse, self.recons_counter, self.lex_counter, sentence, self.tag_table,
                                     self.vocab)
            #print tree_loss
            #print best_parse

            batch_loss += tree_loss
            section_loss += tree_loss
            epoch_loss += tree_loss
            etotal +=sentence.size
            if iSentence % self.batch == 0:
                batch_loss.backward()
                self.trainer.step()
                batch_loss = 0.0
            self.trainer.zero_grad()
        if batch_loss > 0:
            batch_loss.backward()
            self.trainer.step()
        self.trainer.zero_grad()
        print 'Epoch loss:',param.get_scalar(epoch_loss/iSentence,0)
        param.normalize(self.recons_counter, self.lex_counter, self.recons_param, self.lex_param, self.distdim)

    def test(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(utils.read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                best_parse, _ = self.forward(conll_sentence, True)
