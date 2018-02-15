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
import psutil



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
    valid_l = []
    for candidate in l:
        if candidate is not None:
            valid_l.append(candidate)
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
    def __init__(self, vocab, pos, w2i, tag_map, feats,options):
        super(dependency_tagging_model, self).__init__()
        random.seed(1)
        self.activations = {'tanh': F.tanh, 'sigmoid': F.sigmoid, 'relu': F.relu, }
        self.activation = self.activations[options.activation]
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.tdims = options.tembedding_dims
        self.fdims = options.feat_param_dims
        self.ldims = options.lstm_dims
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
            self.extrnd = {word: i for i, word in enumerate(self.external_embedding)}
            np_emb = np.zeros((len(self.external_embedding), self.edim))
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

        self.feat_embedding = param.init_feat_param(self.flookup)

        self.recons_param = np.zeros((self.tag_num, self.tag_num, self.distdim, 2))
        self.lex_param = np.zeros((self.tag_num, len(self.vocab.keys())))

        self.hidLayer = Parameter((2 * self.ldims, self.hidden_units))
        self.hidBias = Parameter(self.hidden_units)
        self.trainer = get_optim(options.optim, self.parameters())

    def evaluate(self, sentence):
        scores = np.zeros((sentence.size, sentence.size, self.max_tag_num, self.max_tag_num))
        scores.fill(-10000)
        crf_scores = np.zeros((sentence.size, sentence.size, self.max_tag_num, self.max_tag_num))
        crf_scores.fill(-10000)
        crf_scores = Variable(torch.FloatTensor(crf_scores))
        param.fire_feats(sentence, scores, crf_scores, self.feat_embedding, self.tag_table, self.flookup,
                         self.recons_param, self.lex_param, self.vocab,self.distdim)

        return scores, crf_scores

    def get_loss(self, best_parse, partition_score, crf_scores, scores):
        encoder_loss = Variable(torch.FloatTensor([0]))
        log_likelihood = 0.0
        for i, h in enumerate(best_parse[0]):
            if h == -1:
                continue
            h = int(h)
            m_tag_id = int(best_parse[1][i])
            h_tag_id = int(best_parse[1][h])
            encoder_loss = encoder_loss + crf_scores[h, i, h_tag_id, m_tag_id]
            log_likelihood += scores[h, i, h_tag_id, m_tag_id]
        # print 'tree_score',param.get_scalar(tree_loss, 0)
        encoder_loss = -(encoder_loss - partition_score)
        #encoder_loss = partition_score
        log_likelihood -= param.get_scalar(partition_score, 0)
        # print 'partition',param.get_scalar(partition_score, 0)
        return encoder_loss, log_likelihood


    #@profile
    def forward(self, sentence, tree_params):
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
        start = time.time()
        scores, crf_scores = self.evaluate(sentence)
        #print "time cost in firing features",time.time()-start
        start = time.time()
        best_parse = eisner_parser.parse_proj(scores)
        #print "time cost in parsing", time.time() - start
        start = time.time()
        partition_score, inside_scores = eisner_parser.partition_inside(crf_scores)
        #print "time cost in computing partition", time.time() - start
        tree_params[sentence] = (scores, crf_scores.data.numpy(), param.get_scalar(partition_score, 0))
        encoder_loss, log_likelihood = self.get_loss(best_parse, partition_score, crf_scores, scores)

        return best_parse, encoder_loss, log_likelihood


    def crf_train(self, sentences, best_trees, tree_params):
        batch_loss = 0.0
        section_loss = 0.0
        etotal = 0
        epoch_loss = 0.0
        training_likelihood = 0.0

        param.init_param(sentences, self.vocab, self.tag_table, self.recons_param, self.lex_param,self.distdim)
        random.shuffle(sentences)
        start = time.time()
        for iSentence, sentence in enumerate(sentences):
            print iSentence
            if iSentence % 10 == 0 and iSentence != 0:
                print 'Loss', param.get_scalar(section_loss / etotal, 0), 'Time', time.time() - start
                start = time.time()
                section_loss = 0.0
                etotal = 0
            best_parse, encoder_loss, log_likelihood = self.forward(sentence, tree_params)
            best_trees[sentence] = best_parse
            batch_loss += encoder_loss
            section_loss += encoder_loss
            epoch_loss += encoder_loss
            etotal += sentence.size
            training_likelihood += log_likelihood
            if iSentence % self.batch == 0 and iSentence != 0:
                inner_start = time.time()
                batch_loss.backward()
                print 'time cost in one backward', time.time() - inner_start
                inner_start = time.time()
                self.trainer.step()
                nn.utils.clip_grad_norm(self.parameters(),2.0)
                batch_loss = 0.0
                #print 'time cost in one update',time.time()-inner_start
            self.trainer.zero_grad()
        if param.get_scalar(batch_loss,0) > 0:
            batch_loss.backward()
            self.trainer.step()
            nn.utils.clip_grad_norm(self.parameters(),2.0)
        self.trainer.zero_grad()
        print 'Iteration loss:', param.get_scalar(epoch_loss / iSentence, 0)
        print 'Training likelihood', training_likelihood

        #     param.normalize(self.recons_counter, self.lex_counter, self.recons_param, self.lex_param, self.distdim)

    def hard_em_train(self, sentences, tree_params):
        self.recons_counter = np.zeros((self.tag_num, self.tag_num, self.distdim, 2))
        self.lex_counter = np.zeros((self.tag_num, len(self.vocab.keys())))
        training_likelihood = 0.0
        for sentence in sentences:
            sentence_scores = tree_params[sentence]
            best_parse = eisner_parser.parse_proj(sentence_scores[0])
            s_log_likelihood = param.counter_update(best_parse, sentence_scores, self.recons_counter, self.lex_counter,
                                                    sentence, self.tag_table, self.vocab,self.distdim)
            training_likelihood += s_log_likelihood
        param.normalize(self.recons_counter, self.lex_counter, self.recons_param, self.lex_param, self.distdim)
        for sentence in sentences:
            scores = param.update_scores(sentence, tree_params, self.tag_table, self.vocab, self.recons_param,
                                         self.lex_param,self.distdim)
            score_tuple = (scores, tree_params[sentence][1], tree_params[sentence][2])
            tree_params[sentence] = score_tuple

        print 'Training likelihood', training_likelihood

    def predict(self, sentence):
        for entry in sentence.entries:
            wordvec = self.wlookup(scalar(int(self.vocab.get(entry.norm, 0)))) if self.wdims > 0 else None
            posvec = self.plookup(scalar(int(self.pos[entry.pos]))) if self.pdims > 0 else None
            evec = self.elookup(scalar(int(self.extrnd.get(entry.form, self.extrnd.get(entry.norm,
                                                                                       0))))) if self.external_embedding is not None else None
            entry.vec = cat([wordvec, posvec, evec])
            entry.lstms = [entry.vec, entry.vec]
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

        scores,_ = self.evaluate(sentence)
        best_parse = eisner_parser.parse_proj(scores)
        return best_parse

    def test(self, sentences):
        best_parses = []
        for iSentence, sentence in enumerate(sentences):
            best_parse = self.predict(sentence)
            best_parses.append(best_parse[0])
        return best_parses

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))
