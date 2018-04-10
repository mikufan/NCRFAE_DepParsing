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


def fire_feats(sentence, scores, crf_scores, feat_embedding, tag_table, flookup, recons_param, lex_param, vocab,
               distdim):
    max_dist = distdim - 1
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
                    # u_feat_h = (h_pos, None, h_id, None, dist, dir)
                    # u_feat_m = (None, m_pos, None, m_id, dist, dir)
                    # scores[i, j, h_id, m_id] += get_scalar(
                    #     feat_embedding(Variable(torch.LongTensor([flookup.find_id(u_feat_h)]))), 0)
                    # scores[i, j, h_id, m_id] += get_scalar(
                    #     feat_embedding(Variable(torch.LongTensor([flookup.find_id(u_feat_m)]))), 0)
                    # crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #     Variable(torch.LongTensor([flookup.find_id(u_feat_h)])))
                    # crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #     Variable(torch.LongTensor([flookup.find_id(u_feat_m)])))
                    b_feat = (h_pos, m_pos, h_id, m_id, dist, dir)
                    scores[i, j, h_id, m_id] += get_scalar(
                        feat_embedding(Variable(torch.LongTensor([flookup.find_id(b_feat)]))), 0)
                    crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                        Variable(torch.LongTensor([flookup.find_id(b_feat)])))
                    # if i - 1 > 0:
                    #     h_lc_pos = s[i - 1].pos
                    #     h_lc_feat = (h_pos, m_pos, h_lc_pos, h_id, m_id, dist, dir)
                    #     scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(h_lc_feat)]))), 0)
                    #     crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(h_lc_feat)])))
                    # if i + 1 < len(s):
                    #     h_rc_pos = s[i + 1].pos
                    #     h_rc_feat = (h_pos, m_pos, h_rc_pos, h_id, m_id, dist, dir)
                    #     scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(h_rc_feat)]))), 0)
                    #     crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(h_rc_feat)])))
                    # if j - 1 > 0:
                    #     m_lc_pos = s[j - 1].pos
                    #     m_lc_feat = (h_pos, m_pos, m_lc_pos, h_id, m_id, dist, dir)
                    #     scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(m_lc_feat)]))), 0)
                    #     crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(m_lc_feat)])))
                    # if j + 1 < len(s):
                    #     m_rc_pos = s[j + 1].pos
                    #     m_rc_feat = (h_pos, m_pos, m_rc_pos, h_id, m_id, dist, dir)
                    #     scores[i, j, h_id, m_id] += get_scalar(feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(m_rc_feat)]))), 0)
                    #     crf_scores[i, j, h_id, m_id] = crf_scores[i, j, h_id, m_id] + feat_embedding(
                    #         Variable(torch.LongTensor([flookup.find_id(m_rc_feat)])))
                    h_tag_id = h_tag_list[h_id]
                    m_tag_id = m_tag_list[m_id]
                    word_id = vocab.get(m_word)
                    scores[i][j][h_id][m_id] += np.log(recons_param[h_tag_id][m_tag_id][dist][dir])
                    scores[i][j][h_id][m_id] += np.log(lex_param[m_tag_id][word_id])
    return


def update_scores(pos_sentence, words_sentence, sidx, scores, crf_scores, recons_param, lex_param, distdim, use_lex,
                  prior_weight, prior_dict):
    max_dist = distdim - 1
    sentence_length, _, tag_num, _ = scores.shape
    dir_dim = recons_param.shape[4]
    for i in range(sentence_length):
        for j in range(sentence_length):
            if j == 0:
                continue
            if i == j:
                continue
            dist = int(abs(i - j)) - 1
            if dist > max_dist:
                dist = max_dist
            if dir_dim == 2:
                if i < j:
                    dir = 1
                else:
                    dir = 0
            else:
                dir = 0
            h_pos_idx = pos_sentence[i]
            m_pos_idx = pos_sentence[j]
            word_idx = words_sentence[j]
            scores[i, j, :, :] = crf_scores[i, j, :, :] + np.log(
                recons_param[h_pos_idx, :, m_pos_idx, dist, dir]).reshape(tag_num, 1)
            if use_lex:
                scores[i, j, :, :] += np.log(lex_param[m_pos_idx, :, word_idx].reshape(1, tag_num))


    return scores


def init_feat_param(feat_lookup):
    feat_embedding = nn.Embedding(feat_lookup.feat_num, 1)
    return feat_embedding


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


def counter_update(best_parses, best_tags, sentence_scores, recons_counter, lex_counter, pos_sentence, words_sentence,
                   distdim, partition, use_lex, sidx, prior_weight, prior_dict):
    max_dist = distdim - 1
    s_log_likelihood = 0.0
    if prior_weight > 0:
        sentence_scores -= prior_dict[sidx]
    for i, h in enumerate(best_parses):
        m_tag_id = int(best_tags[i])
        m_pos = pos_sentence[i]
        if use_lex:
            lex_counter[m_pos, m_tag_id, words_sentence[i]] += 1
        if h == -1:
            continue
        h = int(h)
        h_pos = pos_sentence[h]
        dist = int(abs(i - h)) - 1
        if dist > max_dist:
            dist = max_dist
        if recons_counter.shape[4] == 2:
            if h < i:
                dir = 1
            else:
                dir = 0
        else:
            dir = 0
        h_tag_id = int(best_tags[h])
        recons_counter[h_pos, h_tag_id, m_pos, dist, dir] += 1
        s_log_likelihood += sentence_scores[h, i, h_tag_id, m_tag_id]
    s_log_likelihood -= partition
    return s_log_likelihood


def normalize(recons_counter, lex_counter, recons_param, lex_param, root_idx, use_lex):
    pos_num, tag_num, _, dist_dim, dir_dim = recons_counter.shape
    if use_lex:
        _, _, word_num = lex_counter.shape
        word_sum = np.sum(lex_counter, axis=2).reshape(pos_num, tag_num, 1)
    smoothing = 1e-8
    child_sum = np.sum(recons_counter, axis=2).reshape(pos_num, tag_num, 1, dist_dim, dir_dim)
    smoothing_child = np.empty((pos_num, dist_dim, dir_dim))
    smoothing_child.fill(smoothing)
    smoothing_child[root_idx, :, :] = 0
    smoothing_child_sum = np.sum(smoothing_child, axis=0)
    for i in range(pos_num):
        if i == root_idx:
            recons_param[i, 0, :, :, :] = (recons_counter[i, 0, :, :, :] + smoothing_child) / (
                child_sum[i, 0] + smoothing_child_sum)
        else:
            recons_param[i, :, :, :, :] = (recons_counter[i, :, :, :, :] + smoothing_child) / (
                child_sum[i] + smoothing_child_sum)
    if use_lex:
        smoothing_word = np.empty(word_num)
        smoothing_word.fill(smoothing)
        smoothing_word_sum = np.sum(smoothing_word)
        for i in range(pos_num):
            if i == root_idx:
                continue
            lex_param[i, :, :] = (lex_counter[i] + smoothing_word) / (word_sum[i] + smoothing_word_sum)

    return


def init_param(data, vocab, tag_table, recons_param, lex_param, distdim):
    head = recons_param.shape[0]
    child = recons_param.shape[1]
    dist_dim = recons_param.shape[2]
    dir_dim = 2
    max_dist = distdim - 1
    smoothing = 0.001
    root_idx = tag_table['ROOT-POS'][0]
    for i in range(child):
        if i == root_idx:
            continue
        for j in range(dist_dim):
            for k in range(dir_dim):
                recons_param[root_idx][i][j][k] = 1. / (dist_dim * dir_dim * child)
    for sentence in data:
        for i, h_entry in enumerate(sentence.entries):
            for j, m_entry in enumerate(sentence.entries):
                if i == 0:
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
                        recons_param[h_tag][m_tag][dist][dir] += 1. / (span * len(h_tag_list) * len(m_tag_list))
    for i in range(head):
        for j in range(dist_dim):
            for k in range(dir_dim):
                sum = 0.0
                for c in range(child):
                    if i == root_idx:
                        continue
                    if c == root_idx:
                        continue
                    sum += (recons_param[i][c][j][k] + smoothing)
                for c in range(child):
                    if i == root_idx:
                        continue
                    if c == root_idx:
                        continue
                    recons_param[i][c][j][k] = (recons_param[i][c][j][k] + smoothing) / sum
    for i in range(child):
        sum = 0.0
        for w in range(len(vocab.keys())):
            sum += (lex_param[i][w] + smoothing)
        for w in range(len(vocab.keys())):
            lex_param[i][w] = (lex_param[i][w] + smoothing) / sum
    return


def set_prior(ruleType):
    prior_set = set()
    if ruleType == 'WSJ':
        prior_set.add(("ROOT-POS", "MD"))
        prior_set.add(("ROOT-POS", "VB"))

        prior_set.add(("VB", "NN"))
        prior_set.add(("VB", "WP"))
        prior_set.add(("VB", "PR"))
        prior_set.add(("VB", "RB"))
        prior_set.add(("VB", "VB"))

        prior_set.add(("MD", "VB"))
        prior_set.add(("NN", "JJ"))
        prior_set.add(("NN", "NN"))
        prior_set.add(("NN", "CD"))

        prior_set.add(("IN", "NN"))

        prior_set.add(("JJ", "RB"))
    return prior_set
