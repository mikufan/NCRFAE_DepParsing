import numpy as np
import torch.nn as nn
import torch
from torch.nn.init import *


def get_scalar(var, index):
    if isinstance(var, Variable):
        numpy_var = var.data.numpy()
        flat_numpy_var = numpy_var.flat
    return flat_numpy_var[index]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def fire_feats(sentence, scores, crf_scores, feat_embedding, tag_table, flookup, recons_param, lex_param,vocab):
    max_dist = 5
    s = sentence.entries
    for i, h_entry in enumerate(s):
        for j, m_entry in enumerate(s):
            if j == 0:
                continue
            if i == j:
                continue
            h_pos = h_entry.pos
            m_pos = m_entry.pos
            m_word = m_entry.norm
            dist = abs(i - j)
            if dist > max_dist:
                dist = max_dist
            if i < j:
                dir = 1
            else:
                dir = 0
            h_tag_list = tag_table[h_pos]
            m_tag_list = tag_table[m_pos]
            for h_id in range(len(h_tag_list)):
                for m_id in range(len(m_tag_list)):
                    if scores[i, j, h_id, m_id] > -10000:
                        scores[i, j, h_id, m_id] += get_scalar(s[j].unary_potential, m_id)
                        crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + s[j].unary_potential[
                            0, m_id]
                    else:
                        scores[i, j, h_id, m_id] = get_scalar(s[j].unary_potential, m_id)
                        crf_scores[i, j, h_id, m_id] = s[j].unary_potential[0, m_id]
                    u_feat_h = (h_pos, None, None, h_id, None, dist, dir, 'h')
                    u_feat_m = (None, m_pos, None, None, m_id, dist, dir, 'm')
                    scores[i, j, h_id, m_id] += get_scalar(
                        feat_embedding(Variable(torch.LongTensor([flookup.find_id(u_feat_h)]))), 0)
                    scores[i, j, h_id, m_id] += get_scalar(
                        feat_embedding(Variable(torch.LongTensor([flookup.find_id(u_feat_m)]))), 0)
                    crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                        Variable(torch.LongTensor([flookup.find_id(u_feat_h)])))
                    crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                        Variable(torch.LongTensor([flookup.find_id(u_feat_m)])))
                    b_feat = (h_pos, m_pos, None, h_id, m_id, dist, dir, None)
                    scores[i, j, h_id, m_id] += get_scalar(
                        feat_embedding(Variable(torch.LongTensor([flookup.find_id(b_feat)]))), 0)
                    crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                        Variable(torch.LongTensor([flookup.find_id(b_feat)])))
                    if i - 1 > 0:
                        h_lc_pos = s[i - 1].pos
                        h_lc_feat = (h_pos, m_pos, h_lc_pos, h_id, m_id, dist, dir, 'h')
                        scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(h_lc_feat)]))), 0)
                        crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(h_lc_feat)])))
                    if i + 1 < len(s):
                        h_rc_pos = s[i + 1].pos
                        h_rc_feat = (h_pos, m_pos, h_rc_pos, h_id, m_id, dist, dir, 'h')
                        scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(h_rc_feat)]))), 0)
                        crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(h_rc_feat)])))
                    if j - 1 > 0:
                        m_lc_pos = s[j - 1].pos
                        m_lc_feat = (h_pos, m_pos, m_lc_pos, h_id, m_id, dist, dir, 'm')
                        scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(m_lc_feat)]))), 0)
                        crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(m_lc_feat)])))
                    if j + 1 < len(s):
                        m_rc_pos = s[j + 1].pos
                        m_rc_feat = (h_pos, m_pos, m_rc_pos, h_id, m_id, dist, dir, 'm')
                        scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(m_rc_feat)]))),0)
                        crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                            Variable(torch.LongTensor([flookup.find_id(m_rc_feat)])))
                    h_tag_id = h_tag_list[h_id]
                    m_tag_id = m_tag_list[m_id]
                    word_id = vocab.get(m_word)
                    scores[i][j][h_id][m_id]+= np.log(recons_param[h_tag_id][m_tag_id][dist][dir])
                    scores[i][j][h_id][m_id]+= np.log(lex_param[m_tag_id][word_id])
    return


def get_recons_score(sentence, tag_table, recons_param, max_tag_num):
    recons_score = np.zeros((len(sentence), len(sentence), max_tag_num, max_tag_num))
    if len(sentence) < 6:
        max_dist = len(sentence) - 1
    else:
        max_dist = 5
    for i, h_entry in enumerate(sentence):
        for j, m_entry in enumerate(sentence):
            if j == 0:
                continue
            if i == j:
                continue

            h_pos = h_entry.pos
            m_pos = m_entry.pos
            dist = abs(i - j)
            if dist > max_dist:
                dist = max_dist
            if i < j:
                dir = 1
            else:
                dir = 0
            h_tag_list = tag_table[h_pos]
            m_tag_list = tag_table[m_pos]
            for ih, h in enumerate(h_tag_list):
                for im, m in enumerate(m_tag_list):
                    recons_score[j][i][im][ih] = recons_param[m][h][dist][dir]
    return recons_param


def init_feat_param(feat_lookup):
    feat_param = np.zeros(feat_lookup.feat_num)
    feat_embedding = nn.Embedding(feat_lookup.feat_num, 1)
    for i in range(feat_lookup.feat_num):
        feat_param[i] = get_scalar(feat_embedding(Variable(torch.LongTensor([i]))), 0)
    return feat_param, feat_embedding

def get_lex_score(sentence, vocab, tag_table, lex_param, max_tag_num):
    lex_score = np.zeros((len(sentence), max_tag_num))
    for i, entry in enumerate(sentence):
        pos = entry.pos
        word = entry.norm
        word_id = vocab.get(word)
        tag_list = tag_table.get(pos)
        for j, t in enumerate(tag_list):
            lex_score[i][j] = lex_param[t][word_id]

    return lex_score


def counter_update(best_parse, recons_counter, lex_counter, sentence, tag_table, vocab):
    max_dist = 5
    s = sentence.entries
    for i, h in enumerate(best_parse[0]):
        dist = int(abs(i - h))
        if dist > max_dist:
            dist = max_dist
        if h < i:
            dir = 1
        else:
            dir = 0
        h = int(h)
        h_tag = s[h].pos
        m_tag = s[i].pos
        word = s[i].norm
        h_tag_id = int(best_parse[1][h])
        m_tag_id = int(best_parse[1][i])
        recons_counter[int(tag_table[h_tag][h_tag_id])][int(tag_table[m_tag][m_tag_id])][dist][dir] += 1
        lex_counter[int(tag_table[m_tag][m_tag_id])][vocab.get(word)] += 1
    return


def normalize(recons_counter, lex_counter, recons_param, lex_param, dist_dim):
    tag_num = len(recons_counter)
    word_num = lex_counter.shape[1]
    smoothing = 0.1
    for t in range(tag_num):
        for d in range(dist_dim):
            for i in range(2):
                tag_sum = np.sum(recons_counter[t, :, d, i])+smoothing*tag_num
                recons_param[t, :, d, i] = (recons_counter[t, :, d, i]+smoothing) / tag_sum
    for t in range(tag_num):
        tag_word_sum = np.sum(lex_counter[t, :])+smoothing*word_num
        lex_param[t, :] = (lex_counter[t, :]+smoothing) / tag_word_sum
    return

def init_param(data,vocab,tag_table,recons_param,lex_param):
    head = recons_param.shape[0]
    child = recons_param.shape[1]
    dist_dim =recons_param.shape[2]
    dir_dim = 2
    max_dist = 5
    v_num = len(vocab.keys())
    smoothing = 0.001
    root_idx = tag_table['ROOT-POS'][0]
    for i in range(child):
        if i == root_idx:
            continue
        for j in range(dist_dim):
                for k in range(dir_dim):
                    recons_param[root_idx][i][j][k] = 1./(dist_dim*dir_dim*child)
    for sentence in data:
        for i,h_entry in enumerate(sentence.entries):
            for j,m_entry in enumerate(sentence.entries):
                if i == 0:
                    continue
                if i == j:
                    continue
                if j == 0:
                    continue
                dist = abs(i - j)
                span = dist
                if dist > max_dist:
                    dist = max_dist
                if i < j:
                    dir = 1
                else:
                    dir = 0
                h_pos = h_entry.pos
                m_pos = m_entry.pos
                word = m_entry.norm
                word_id = vocab.get(word)
                h_tag_list = tag_table[h_pos]
                m_tag_list = tag_table[m_pos]
                for m_tag in m_tag_list:
                    lex_param[m_tag][word_id] += 1. / (len(m_tag_list))
                    for h_tag in h_tag_list:
                        recons_param[h_tag][m_tag][dist][dir]+=1./(span*len(h_tag_list)*len(m_tag_list))
    for i in range(head):
        for j in range(dist_dim):
            for k in range(dir_dim):
                sum = 0.0
                for c in range(child):
                    if i == root_idx:
                        continue
                    if c == root_idx:
                        continue
                    sum+=(recons_param[i][c][j][k] + smoothing)
                for c in range(child):
                    if i == root_idx:
                        continue
                    if c == root_idx:
                        continue
                    recons_param[i][c][j][k] = (recons_param[i][c][j][k]+smoothing)/sum
    for i in range(child):
        sum = 0.0
        for w in range(len(vocab.keys())):
            sum += (lex_param[i][w]+smoothing)
        for w in range(len(vocab.keys())):
            lex_param[i][w] = (lex_param[i][w]+smoothing)/sum
    return

def set_prior():
    return