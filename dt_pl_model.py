import torch
import torch.nn as nn
import numpy as np
import param
from torch.nn.init import *
from torch import optim
import eisner_layer as EL
import eisner_parser
import utils
import time
import torch.nn.functional as F
import shutil


def get_optim(opt, parameters):
    if opt.optim == 'sgd':
        return optim.SGD(parameters, lr=opt.learning_rate)
    elif opt.optim == 'adam':
        return optim.Adam(parameters)


class dt_paralell_model(nn.Module):
    def __init__(self, w2i, pos, options):
        super(dt_paralell_model, self).__init__()
        self.tag_num = options.tag_num
        self.dist_num = options.dist_num
        self.embedding_dim = options.wembedding_dim
        self.pdim = options.pembedding_dim
        self.hidden_dim = options.hidden_dim
        self.n_layer = options.n_layer
        self.external_embedding = options.external_embedding
        self.dist_dim = options.dist_dim
        self.vocab = {word: ind for word, ind in w2i.iteritems()}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.gpu = options.gpu
        self.tdim = options.tag_dim
        self.ddim = options.dist_dim
        self.trans_pos_dim = options.pembedding_dim
        self.dropout_ratio = options.dropout_ratio
        self.dir_flag = options.dir_flag

        if self.external_embedding != None:
            print 'Loading external embeddings'
            with open(self.external_embedding, 'r') as emb_file:
                extrn_dim, to_augment, = utils.use_external_embedding(emb_file, self.vocab)
            print 'External embeddings loaded'
            self.embedding_dim = extrn_dim
            self.wlookup = nn.Embedding(len(self.vocab), self.embedding_dim)
            augmented = utils.build_new_emb(self.wlookup.weight.data.numpy(), to_augment, self.vocab)
            self.wlookup.weight.data.copy_(torch.from_numpy(augmented))
        else:
            self.wlookup = nn.Embedding(len(self.vocab), self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim + self.pdim, self.hidden_dim, self.n_layer, bidirectional=True,
                            dropout=self.dropout_ratio)
        # self.hidden2tags = nn.Linear(self.hidden_dim, self.tag_num)
        self.hidden2tags = nn.Linear(2 * self.hidden_dim, self.tag_num)
        self.pre_hidden = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.trans_hidden = options.trans_hidden
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)

        self.feat2trans = nn.Linear(2 * (self.tdim + self.trans_pos_dim)+self.ddim, self.trans_hidden)
        self.trans_pos_emb = nn.Embedding(len(pos), self.pdim)
        self.tlookup = nn.Embedding(self.tag_num, self.tdim)

        self.plookup = nn.Embedding(len(pos), self.pdim)
        self.tran_pos = nn.Embedding(len(pos), self.pdim)
        self.dlookup = nn.Embedding(self.dist_num*2+1,self.ddim)

        self.transitions = nn.Embedding(self.tag_num * self.tag_num, 1)
        self.transition_map = self.set_transitions()

        self.recons_param, self.lex_param = None, None
        self.trainer = get_optim(options, self.parameters())

        self.tree_param = {}
        self.partition_table = {}
        self.encoder_score_table = {}
        self.parse_results = {}

    def evaluate(self, batch_pos, batch_words, batch_sen, crf_scores):
        batch_size, sentence_length = batch_pos.data.shape
        scores = np.copy(crf_scores.cpu().data.numpy())
        for sentence_id in range(batch_size):
            for i in range(sentence_length):
                for j in range(sentence_length):
                    word_id = param.get_scalar(batch_words[sentence_id][j].cpu(), 0)
                    h_pos_id = param.get_scalar(batch_pos[sentence_id][i].cpu(), 0)
                    m_pos_id = param.get_scalar(batch_pos[sentence_id][j].cpu(), 0)
                    if j == 0:
                        continue
                    if i == j:
                        continue
                    if abs(i - j) > self.dist_dim - 1:
                        dist = self.dist_dim - 1
                    else:
                        dist = abs(i - j) - 1
                    if self.dir_flag:
                        if i > j:
                            dir = 0
                        else:
                            dir = 1
                    else:
                        dir = 0
                    scores[sentence_id, i, j, :, :] += np.log(self.recons_param[h_pos_id, :, m_pos_id, :, dist, dir])
                    scores[sentence_id, i, j, :, :] += np.log(self.lex_param[m_pos_id, :, word_id]
                                                              .reshape(1, self.tag_num))
            self.tree_param[batch_sen[sentence_id]] = scores[sentence_id]
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
            single_crf_scores = single_crf_scores.unsqueeze(0)
            crf_scores.append(single_crf_scores)
        crf_scores = torch.cat(crf_scores, 0)
        transition_map_var = utils.list2Variable(self.transition_map, self.gpu)
        transition_var = self.transitions(transition_map_var)
        transition_var = transition_var.view(self.tag_num, self.tag_num)
        # transition_var = self.transitions
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.unsqueeze(0)
        transition_var = transition_var.repeat(batch_size, sentence_length, sentence_length, 1, 1)
        crf_scores = crf_scores + transition_var
        crf_scores = crf_scores.double()
        if torch.cuda.is_available():
            crf_scores = crf_scores.cuda()
        return crf_scores.double()

    def compute_neural_crf_scores(self, lstm_feats, trans_matrix, sentence_length):
        unary_potential = lstm_feats.unsqueeze(1)
        unary_potential = unary_potential.repeat(1, sentence_length, 1, 1)
        unary_potential = unary_potential.unsqueeze(4)
        unary_potential = unary_potential.repeat(1, 1, 1, 1, self.tag_num)
        crf_scores = unary_potential + trans_matrix
        crf_scores = crf_scores.double()
        if torch.cuda.is_available():
            crf_scores = crf_scores.cuda()
        return crf_scores

    def compute_trans(self, batch_pos):

        trans_var = []
        batch_size, sentence_length = batch_pos.data.shape

        pos_emb = self.trans_pos_emb(batch_pos)
        pos_emb_h = pos_emb.unsqueeze(2)
        pos_emb_m = pos_emb_h.permute(0, 2, 1, 3)
        pos_emb_h = pos_emb_h.repeat(1, 1, sentence_length, 1)
        pos_emb_m = pos_emb_m.repeat(1, sentence_length, 1, 1)
        pos_emb_h = pos_emb_h.unsqueeze(3)
        pos_emb_h = pos_emb_h.unsqueeze(4)
        pos_emb_m = pos_emb_m.unsqueeze(3)
        pos_emb_m = pos_emb_m.unsqueeze(4)
        pos_emb_h = pos_emb_h.repeat(1, 1, 1, self.tag_num, self.tag_num, 1)
        pos_emb_m = pos_emb_m.repeat(1, 1, 1, self.tag_num, self.tag_num, 1)

        trans_var.append(pos_emb_h)
        trans_var.append(pos_emb_m)
        tag_list = []
        for i in range(self.tag_num):
            tag_list.append(i)
        tag_var = utils.list2Variable(tag_list, self.gpu)
        tag_emb = self.tlookup(tag_var)
        tag_emb_h = tag_emb.unsqueeze(1)
        tag_emb_m = tag_emb_h.permute(1, 0, 2)
        tag_emb_h = tag_emb_h.repeat(1, self.tag_num, 1)
        tag_emb_m = tag_emb_m.repeat(self.tag_num, 1, 1)
        tag_emb_h = tag_emb_h.unsqueeze(0)
        tag_emb_h = tag_emb_h.unsqueeze(0)
        tag_emb_h = tag_emb_h.unsqueeze(0)
        tag_emb_m = tag_emb_m.unsqueeze(0)
        tag_emb_m = tag_emb_m.unsqueeze(0)
        tag_emb_m = tag_emb_m.unsqueeze(0)
        tag_emb_h = tag_emb_h.repeat(batch_size, sentence_length, sentence_length, 1, 1, 1)
        tag_emb_m = tag_emb_m.repeat(batch_size, sentence_length, sentence_length, 1, 1, 1)
        trans_var.append(tag_emb_h)
        trans_var.append(tag_emb_m)


        dist_list = list()
        for i in range(sentence_length):
            head_list = list()
            for j in range(sentence_length):
                if abs(i-j)>self.dist_num:
                    head_list.append(np.sign(i-j)*self.dist_num + self.dist_num)
                else:
                    head_list.append(i - j + self.dist_num)
            dist_list.append(head_list)
        dist_var = utils.list2Variable(dist_list,self.gpu)
        dist_emb = self.dlookup(dist_var)
        dist_emb = dist_emb.unsqueeze(2)
        dist_emb = dist_emb.unsqueeze(3)
        dist_emb = dist_emb.repeat(1,1,self.tag_num,self.tag_num,1)
        dist_emb = dist_emb.unsqueeze(0)
        dist_emb = dist_emb.repeat(batch_size,1,1,1,1,1)
        trans_var.append(dist_emb)


        trans = torch.cat(trans_var, dim=5)
        trans_hidden = self.feat2trans(trans)
        trans_vec = F.relu(trans_hidden)
        trans_matrix = torch.mean(trans_vec, dim=5)


        return trans_matrix

    def get_tree_score(self, crf_score, scores, best_parse, partition, batch_sen):
        best_parse = np.array(list(best_parse), dtype=int)
        _, batch_size, sentence_length = best_parse.shape
        tree_score = []
        likelihood = 0.0
        for sentence_id in range(batch_size):
            sentence_tree_score = []
            for i in range(1, sentence_length):
                head_id = best_parse[0, sentence_id, i]
                sentence_tree_score.append(crf_score[sentence_id, best_parse[0, sentence_id, i], i, best_parse[
                    1, sentence_id, head_id], best_parse[1, sentence_id, i]])
                likelihood += scores[sentence_id, best_parse[0, sentence_id, i], i, best_parse[
                    1, sentence_id, head_id], best_parse[1, sentence_id, i]]
            likelihood -= param.get_scalar(partition[sentence_id].cpu(), 0)
            self.partition_table[batch_sen[sentence_id]] = param.get_scalar(partition[sentence_id].cpu(), 0)
            self.encoder_score_table[batch_sen[sentence_id]] = crf_score[sentence_id].cpu().data.numpy()
            tree_score.append(torch.sum(torch.cat(sentence_tree_score)))
        tree_score = torch.cat(tree_score)
        return tree_score, likelihood

    def forward(self, batch_words, batch_pos, extrns, batch_sen):
        batch_size, sentence_length = batch_words.data.size()
        w_embeds = self.wlookup(batch_words)
        p_embeds = self.plookup(batch_pos)

        # w_embeds = self.dropout1(w_embeds)

        batch_input = torch.cat((w_embeds, p_embeds), 2)
        hidden_out, _ = self.lstm(batch_input)
        # hidden_out = self.dropout2(hidden_out)
        # pre = self.pre_hidden(hidden_out)
        # temp = F.relu(pre)
        # hidden_out_matrix = hidden_out.transpose(1,2)
        # edge_scores = []
        # for sentence_id in range(batch_size):
        #     edge_score = torch.mm(hidden_out[sentence_id],hidden_out_matrix[sentence_id])
        #     edge_score = edge_score.unsqueeze(0)
        #     edge_scores.append(edge_score)
        # edge_scores = torch.cat(edge_score)
        # edge_scores = edge_scores.view(batch_size,sentence_length,sentence_length,1,1)
        trans_matrix = self.compute_trans(batch_pos)
        lstm_feats = self.hidden2tags(hidden_out)
        # lstm_feats = self.hidden2tags(temp)
        mask = self.construct_mask(batch_size, sentence_length)
        crf_scores = self.compute_neural_crf_scores(lstm_feats, trans_matrix, sentence_length)
        # crf_scores = self.computing_crf_scores(lstm_feats, batch_size, sentence_length)
        if torch.cuda.is_available():
            mask = mask.cuda()
        crf_scores = crf_scores.masked_fill(mask, -np.inf)

        scores = self.evaluate(batch_pos, batch_words, batch_sen, crf_scores)
        best_parse = eisner_parser.batch_parse(scores)
        for i in range(batch_size):
            self.parse_results[batch_sen[i]] = (best_parse[0][i], best_parse[1][i])

        eisner = EL.eisner_layer(sentence_length, self.tag_num, batch_size)
        partition = eisner(crf_scores)
        best_tree_score, batch_likelihood = self.get_tree_score(crf_scores, scores, best_parse, partition, batch_sen)
        loss = partition - best_tree_score
        loss = loss.sum() / batch_size

        return loss, batch_likelihood

    def hard_em_e(self, batch_pos, batch_words, sen_idx, recons_counter, lex_counter):
        batch_likelihood = 0.0
        batch_size = len(batch_pos)
        batch_score = []
        for sentence_id in range(batch_size):
            sidx = sen_idx[sentence_id]
            sentence_scores = self.tree_param[sidx]
            batch_score.append(sentence_scores)
        batch_score = np.array(batch_score)
        # start = time.time()
        best_parse = eisner_parser.batch_parse(batch_score)
        # print 'Time cost in parsing this batch ', time.time() - start
        for sentence_id in range(batch_size):
            sentence_pos = batch_pos[sentence_id]
            sentence_words = batch_words[sentence_id]
            sidx = sen_idx[sentence_id]
            sentence_scores = self.tree_param[sidx]
            s_log_likelihood = param.counter_update(best_parse[0][sentence_id], best_parse[1][sentence_id],
                                                    sentence_scores, recons_counter, lex_counter, sentence_pos,
                                                    sentence_words, self.dist_dim, self.partition_table[sidx])
            batch_likelihood += s_log_likelihood
        return batch_likelihood

    def hard_em_m(self, batch_data, recons_counter, lex_counter):
        root_idx = self.pos['ROOT-POS']
        param.normalize(recons_counter, lex_counter, self.recons_param, self.lex_param, root_idx)
        batch_num = len(batch_data)
        for batch_id in range(batch_num):
            one_batch = batch_data[batch_id]
            batch_words, batch_pos, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                [s[2][0] for s in one_batch]
            batch_size = len(batch_words)
            for i in range(batch_size):
                sidx = batch_sen[i]
                scores = self.tree_param[sidx]
                crf_scores = self.encoder_score_table[sidx]
                sentence_pos = batch_pos[i]
                sentence_words = batch_words[i]
                scores = param.update_scores(sentence_pos, sentence_words, scores, crf_scores, self.recons_param,
                                             self.lex_param, self.dist_dim)
                self.tree_param[sidx] = scores

    def save(self, fn):
        tmp = fn + '.tmp'
        torch.save(self.state_dict(), tmp)
        shutil.move(tmp, fn)

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

    def init_decoder_param(self, data):
        if self.dir_flag:
            dir_dim = 2
        else:
            dir_dim = 1
        pos_num = len(self.pos.keys())
        word_num = len(self.vocab.keys())
        self.recons_param = np.zeros(
            (pos_num, self.tag_num, pos_num, self.tag_num, self.dist_dim, dir_dim), dtype=np.float64)
        self.lex_param = np.zeros((pos_num, self.tag_num, word_num), dtype=np.float64)
        max_dist = self.dist_dim - 1
        smoothing = 0.00001
        root_idx = self.pos['ROOT-POS']
        for child_idx in range(pos_num):
            if child_idx == root_idx:
                continue
            self.recons_param[root_idx, 0, child_idx, :, :, dir_dim - 1] = 1. / (pos_num * self.tag_num * self.dist_dim)

        for sentence in data:
            for i, h_entry in enumerate(sentence.entries):
                for j, m_entry in enumerate(sentence.entries):
                    if i == 0:
                        continue
                    if i == j:
                        continue
                    if j == 0:
                        continue
                    dist = abs(i - j) - 1
                    span = abs(i - j)
                    if dist > max_dist:
                        dist = max_dist
                    if self.dir_flag:
                        if i < j:
                            dir = 1
                        else:
                            dir = 0
                    else:
                        dir = 0
                    h_pos = h_entry.pos
                    m_pos = m_entry.pos
                    word = m_entry.norm
                    h_pos_id = self.pos.get(h_pos)
                    m_pos_id = self.pos.get(m_pos)
                    word_id = self.vocab.get(word)
                    self.lex_param[m_pos_id, :, word_id] += 1. / self.tag_num
                    self.recons_param[h_pos_id, :, m_pos_id, :, dist, dir] += 1. / (span * self.tag_num * self.tag_num)

        for i in range(pos_num):
            if i == root_idx:
                continue
            child_sum = np.sum(self.recons_param[i, :, :, :, :, :], axis=(1, 2))
            smoothing_child = np.empty((self.tag_num, pos_num, self.tag_num, self.dist_dim, dir_dim))
            smoothing_child.fill(smoothing)
            child_sum = child_sum + np.sum(smoothing_child, axis=(1, 2))
            self.recons_param[i, :, :, :, :, :] = (self.recons_param[i, :, :, :, :, :] + smoothing_child) / child_sum

        self.recons_param[:, :, root_idx, :, :, :] = 0
        lex_smoothing = np.empty((pos_num, self.tag_num, word_num))
        lex_smoothing.fill(smoothing)

        lex_sum = np.sum(self.lex_param, axis=2) + np.sum(lex_smoothing, axis=2)
        lex_sum = lex_sum.reshape(pos_num, self.tag_num, 1)
        self.lex_param = (self.lex_param + lex_smoothing) / lex_sum

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        return self._apply(lambda t: t.cuda(device_id))

    def init_simple_decoder_param(self,data):
        if self.dir_flag:
            dir_dim = 2
        else:
            dir_dim = 1
        pos_num = len(self.pos.keys())
        word_num = len(self.vocab.keys())
        self.recons_param = np.zeros(
            (pos_num, self.tag_num, pos_num, self.tag_num, dir_dim), dtype=np.float64)
        self.lex_param = np.zeros((pos_num, self.tag_num, word_num), dtype=np.float64)
        max_dist = self.dist_dim - 1
        smoothing = 0.00001
        root_idx = self.pos['ROOT-POS']


