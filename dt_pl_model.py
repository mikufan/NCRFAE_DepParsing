import torch
import torch.nn as nn
import numpy as np
import param
from torch.nn.init import *
from torch import optim
import eisner_layer as EL
import eisner_parser
import utils


def Parameter(shape=None, init=xavier_uniform):
    shape = (shape, 1) if type(shape) == int else shape
    return nn.Parameter(init(torch.Tensor(*shape)))


def get_optim(opt, parameters):
    if opt == 'sgd':
        return optim.SGD(parameters, lr=opt.lr)
    elif opt == 'adam':
        return optim.Adam(parameters)


def scalar(f):
    if type(f) == int:
        return Variable(torch.LongTensor([f]))
    if type(f) == float:
        return Variable(torch.FloatTensor([f]))


def cat(l, dimension=-1):
    valid_l = []
    for candidate in l:
        if candidate is not None:
            valid_l.append(candidate)
    if dimension < 0:
        dimension += len(valid_l[0].size())
    return torch.cat(valid_l, dimension)


class dt_paralell_model(nn.Module):
    def __init__(self, vocab, w2i, pos, data, options):
        super(dt_paralell_model, self).__init__()
        self.tag_num = options.tag_num
        self.embedding_dim = options.wembedding_dim
        self.pdim = options.pembedding_dim
        self.hidden_dim = options.hidden_dim
        self.n_layer = options.n_layer
        self.extr_dim = 0
        self.dist_dim = options.dist_dim
        self.vocab = {word: ind for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        # self.feature_map = feature_map
        self.lstm = nn.LSTM(self.embedding_dim + self.pdim, self.hidden_dim, self.n_layer, bidirectional=True)

        self.hidden2tags = nn.Linear(2 * self.hidden_dim, self.tag_num)
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
        self.wlookup = nn.Embedding(len(vocab), self.embedding_dim)
        self.plookup = nn.Embedding(len(pos), self.pdim)

        self.transitions = nn.Embedding(self.tag_num * self.tag_num, 1)
        self.transition_map = self.set_transitions()

        # self.tlookup = nn.Embedding(self.tag_num, self.hidden_units)
        # self.flookup = flookup
        # feat_embedding = nn.Embedding(flookup.feat_num, 1)

        # self.feat_embedding = param.init_feat_param(self.flookup)
        # self.hidLayer = Parameter((2 * self.ldims, self.hidden_units))
        # self.hidBias = Parameter(self.hidden_units)
        self.recons_param, self.lex_param = param.init_decoder_param(vocab, data, self.dist_dim, self.tag_num)
        self.trainer = get_optim(options.optim, self.parameters())

    def set_tags(self, tag_map):
        tag_pointer = 0
        tag_num = 0
        tag_table = {}
        max_tag_num = 0
        for t in tag_map.keys():
            tag_size = tag_map[t]
            tag_num += tag_size
            tag_list = []
            for s in range(tag_size):
                tag_list.append(tag_pointer)
                tag_pointer += 1
            tag_table[t] = tag_list
            if len(tag_list) > self.max_tag_num:
                max_tag_num = len(tag_list)
        return tag_num, tag_table, max_tag_num

    def pre_process(self, sentence):
        embeds = list()
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
            embeds.append(entry.vec)
        return embeds

    def evaluate(self, batch_words, batch_pos, recons_param, lex_param, vocab, crf_scores):
        batch_size, sentence_length = batch_words.data.shape
        scores = crf_scores.data.numpy()
        for sentence_id in range(batch_size):
            for i in range(sentence_length):
                for j in range(sentence_length):
                    word_id = param.get_scalar(batch_words[sentence_id][j], 0)
                    if j == 0:
                        continue
                    if i == j:
                        continue
                    if abs(i - j) > self.dist_dim - 1:
                        dist = self.dist_dim - 1
                    else:
                        dist = abs(i - j)
                    if i > j:
                        dir = 0
                    else:
                        dir = 1
                    scores[sentence_id, i, j, :, :] += np.log(recons_param[:, :, dist, dir])
                    scores[sentence_id, i, j, :, :] += np.log(lex_param[:, word_id])

        return scores

    def construct_mask(self, batch_size, sentence_length):
        masks = np.zeros((batch_size, sentence_length, sentence_length, self.tag_num, self.tag_num))
        masks[:, :, 0, :, :] = 1
        masks[:, 0, :, 1:, :] = 1
        masks[:, :, :, :, 0] = 1
        for i in range(sentence_length):
            masks[:, i, i, :, :] = 1
        masks = masks.astype(int)
        mask_var = Variable(torch.ByteTensor(masks))
        return mask_var

    def set_transitions(self):
        transition_map = []
        for l_id in range(self.tag_num):
            for r_id in range(self.tag_num):
                id = l_id * self.tag_num + r_id
                transition_map.append(id)
        transition_map = np.array(transition_map)
        transition_map = transition_map.reshape(self.tag_num, self.tag_num)
        return transition_map

    def computing_crf_scores(self, lstm_feats, batch_size, sentence_length):
        crf_scores = []
        unary_potentials = lstm_feats.contiguous().view(batch_size, sentence_length * self.tag_num)
        for sentence_id in range(batch_size):
            single_unary_potentials = unary_potentials[sentence_id]
            single_crf_scores = torch.log(
                torch.ger(torch.exp(single_unary_potentials), torch.exp(single_unary_potentials)))
            single_crf_scores = single_crf_scores.contiguous().view(sentence_length, sentence_length, self.tag_num,
                                                                    self.tag_num)
            single_crf_scores.unsqueeze(0)
            crf_scores.append(single_crf_scores)
        crf_scores = torch.cat(crf_scores, 0)
        transition_map_var = utils.list2Variable(self.transition_map)
        transition_var = self.transitions(transition_map_var)
        transition_var = transition_var.view(self.tag_num, self.tag_num)
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.repeat(batch_size, sentence_length, sentence_length, 1, 1)
        crf_scores = crf_scores + transition_var
        return crf_scores

    def get_tree_score(self, crf_score, best_parse):
        best_parse = np.array(list(best_parse), dtype=int)
        _, batch_size, sentence_length = best_parse.shape
        tree_score = []
        for sentence_id in range(batch_size):
            sentence_tree_score = []
            for i in range(1, sentence_length):
                head_id = best_parse[0, sentence_id, i]
                sentence_tree_score.append(crf_score[sentence_id, best_parse[0, sentence_id, i], i, best_parse[
                    1, sentence_id, head_id], best_parse[1, sentence_id, i]])
            tree_score.append(torch.sum(torch.cat(sentence_tree_score)))
        tree_score = torch.cat(tree_score)
        return tree_score

    def forward(self, batch_words, batch_pos, extrns):
        batch_size, sentence_length = batch_words.data.size()
        w_embeds = self.wlookup(batch_words)
        p_embeds = self.plookup(batch_pos)

        batch_input = torch.cat((w_embeds, p_embeds), 2)
        hidden_out, _ = self.lstm(batch_input)
        lstm_feats = self.hidden2tags(hidden_out)

        mask = self.construct_mask(batch_size, sentence_length)
        crf_scores = self.computing_crf_scores(lstm_feats, batch_size, sentence_length)
        crf_scores = crf_scores.masked_fill(mask, -np.inf)

        scores = self.evaluate(batch_words, batch_pos, self.recons_param, self.lex_param, self.vocab,
                               crf_scores)
        best_parse = eisner_parser.batch_parse(scores)
        eisner = EL.eisner_layer(sentence_length, self.tag_num, batch_size)
        partition = eisner(crf_scores)
        best_tree_score = self.get_tree_score(crf_scores, best_parse)
        loss = partition - best_tree_score

        return loss
