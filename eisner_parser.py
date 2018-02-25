# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/

import numpy as np
import torch
from torch.nn.init import *
import param
import utils


def parse_proj(scores):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc, nt1, nt2 = np.shape(scores)
    # if nr != nc:
    # raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1  # Number of words (excluding root).
    T = nt1  # Max number of subtags
    # Initialize CKY table.
    complete = np.zeros([N + 1, N + 1, T, 2])  # s, t, s_tag,t_tag,direction (right=1).
    incomplete = np.zeros([N + 1, N + 1, T, T, 2])  # s, t, s_tag,direction (right=1).
    complete_backtrack = -np.ones([N + 1, N + 1, T, 2, 2], dtype=int)  # s, t, direction,(right=1),q_tag.
    incomplete_backtrack = -np.ones([N + 1, N + 1, T, T, 2], dtype=int)  # s, t, direction (right=1).

    incomplete[0, :, :, :, 0] -= 10000

    best_parse = np.zeros((2, N + 1))

    # Loop from smaller items to larger items.
    for k in xrange(1, N + 1):
        for s in xrange(N - k + 1):
            t = s + k
            # First, create incomplete items.

            for s_tag in range(T):
                for t_tag in range(T):
                    # left tree

                    incomplete_vals0 = np.add(complete[s, s:t, s_tag, 1], complete[(s + 1):(t + 1), t, t_tag, 0]) + \
                                       scores[
                                           t, s, t_tag, s_tag]
                    m = np.max(incomplete_vals0)
                    incomplete[s, t, s_tag, t_tag, 0] = np.max(incomplete_vals0)
                    r = s + np.argmax(incomplete_vals0)
                    incomplete_backtrack[s, t, s_tag, t_tag, 0] = s + np.argmax(incomplete_vals0)

                    # right tree
                    incomplete_vals1 = complete[s, s:t, s_tag, 1] + complete[(s + 1):(t + 1), t, t_tag, 0] + scores[
                        s, t, s_tag, t_tag]
                    m = np.max(incomplete_vals1)
                    incomplete[s, t, s_tag, t_tag, 1] = np.max(incomplete_vals1)
                    r = s + np.argmax(incomplete_vals1)
                    incomplete_backtrack[s, t, s_tag, t_tag, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            for t_tag in range(T):
                max_complete_vals0 = -np.inf
                max_position = -1
                max_tag = -1
                for q_tag in range(T):
                    # left tree
                    complete_vals0 = np.add(complete[s, s:t, q_tag, 0], incomplete[s:t, t, q_tag, t_tag, 0])
                    if np.max(complete_vals0) > max_complete_vals0:
                        max_complete_vals0 = np.max(complete_vals0)
                        max_position = np.argmax(complete_vals0)
                        max_tag = q_tag
                complete[s, t, t_tag, 0] = max_complete_vals0
                complete_backtrack[s, t, t_tag, 0, 0] = s + max_position
                complete_backtrack[s, t, t_tag, 0, 1] = max_tag
            for s_tag in range(T):
                max_complete_vals1 = -np.inf
                max_position = -1
                max_tag = -1
                for q_tag in range(T):
                    # right tree
                    complete_vals1 = np.add(incomplete[s, (s + 1):(t + 1), s_tag, q_tag, 1],
                                            complete[(s + 1):(t + 1), t,
                                            q_tag, 1])
                    if np.max(complete_vals1) > max_complete_vals1:
                        max_complete_vals1 = np.max(complete_vals1)
                        max_position = np.argmax(complete_vals1)
                        max_tag = q_tag
                complete[s, t, s_tag, 1] = max_complete_vals1
                complete_backtrack[s, t, s_tag, 1, 0] = s + 1 + max_position
                complete_backtrack[s, t, s_tag, 1, 1] = max_tag
    heads = [-1 for _ in range(N + 1)]  # -np.ones(N+1, dtype=int)
    tags = [0 for _ in range(N + 1)]
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 0, 0, 1, 1, heads, tags)
    best_parse[0] = heads
    best_parse[1] = tags

    return best_parse


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, s_tag, t_tag, direction, complete, heads, tags):
    if s == t:
        return
    if complete:

        if direction == 0:
            r = complete_backtrack[s][t][t_tag][direction][0]
            q_tag = complete_backtrack[s][t][t_tag][direction][1]
            # print "r is" ,r
            # print "s is", s
            # print "t is", t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, q_tag, 0, 1, heads, tags)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, q_tag, t_tag, 0, 0, heads, tags)
            return
        else:
            r = complete_backtrack[s][t][s_tag][direction][0]
            # print "r is", r
            # print "s is", s
            # print "t is", t
            q_tag = complete_backtrack[s][t][s_tag][direction][1]
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, s_tag, q_tag, 1, 0, heads, tags)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, q_tag, 0, 1, 1, heads, tags)
            return
    else:
        if direction == 0:

            r = incomplete_backtrack[s][t][s_tag][t_tag][direction]
            heads[s] = t
            tags[s] = s_tag
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, s_tag, 0, 1, 1, heads, tags)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, t_tag, 0, 1, heads, tags)
            return
        else:
            r = incomplete_backtrack[s][t][s_tag][t_tag][direction]
            heads[t] = s
            tags[t] = t_tag
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, s_tag, 0, 1, 1, heads, tags)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r + 1, t, 0, t_tag, 0, 1, heads, tags)
            return


def partition_inside(scores):
    (nr, nc, nt1, nt2) = scores.shape
    scores = Variable(torch.FloatTensor(scores))
    N = nr - 1  # Number of words (excluding root).
    T = nt1  # Max number of subtags
    # Initialize CKY table.
    complete = Variable(torch.zeros(N + 1, N + 1, T, 2))  # s, t, s_tag,t_tag,direction (right=1).
    incomplete = Variable(torch.zeros(N + 1, N + 1, T, T, 2))  # s, t, s_tag,direction (right=1).
    for k in xrange(1, N + 1):
        for s in xrange(N - k + 1):
            t = s + k
            # First, create incomplete items.

            for s_tag in range(T):
                for t_tag in range(T):
                    # left tree

                    incomplete_vals0 = complete[s, s:t, s_tag, 1] + complete[(s + 1):(t + 1), t, t_tag, 0] + scores[
                        t, s, t_tag, s_tag].expand(1, t - s)
                    su = param.log_sum_exp(incomplete_vals0.view(1, -1))
                    incomplete[s, t, s_tag, t_tag, 0] = param.log_sum_exp(incomplete_vals0.view(1, -1))
                    # right tree
                    incomplete_vals1 = complete[s, s:t, s_tag, 1] + complete[(s + 1):(t + 1), t, t_tag, 0] + scores[
                        s, t, s_tag, t_tag].expand(1, t - s)
                    su = param.log_sum_exp(incomplete_vals1.view(1, -1))
                    incomplete[s, t, s_tag, t_tag, 1] = param.log_sum_exp(incomplete_vals1.view(1, -1))
            for t_tag in range(T):
                sum_complete_vals0 = []
                for q_tag in range(T):
                    # left tree
                    complete_vals0 = complete[s, s:t, q_tag, 0] + incomplete[s:t, t, q_tag, t_tag, 0]
                    sum_complete_vals0.append(param.log_sum_exp(complete_vals0.view(1, -1)))
                sum_complete_vals0 = torch.cat(sum_complete_vals0,0).view(1,-1)
                complete[s, t, t_tag, 0] = param.log_sum_exp(sum_complete_vals0)
            for s_tag in range(T):
                sum_complete_vals1 = []
                for q_tag in range(T):
                    # right tree
                    complete_vals1 = incomplete[s, (s + 1):(t + 1), s_tag, q_tag, 1] + complete[(s + 1):(t + 1), t,
                                                                                       q_tag, 1]
                    sum_complete_vals1.append(param.log_sum_exp(complete_vals1.view(1, -1)))
                sum_complete_vals1 = torch.cat(sum_complete_vals1,0).view(1,-1)
                complete[s, t, s_tag, 1] = param.log_sum_exp(sum_complete_vals1)
    sentence_score = complete[0, N, 0, 1]
    inside_score = (complete, incomplete)
    return sentence_score, inside_score


def partition_outside(inside_scores, scores):
    complete_inside = inside_scores[0]
    incomplete_inside = inside_scores[1]
    nr, nc, nt1, nt2 = np.shape(scores)
    N = nr - 1
    T = nt1
    scores = np.exp(scores)
    complete_outside = np.zeros([N + 1, N + 1, T, 2])  # s, t, s_tag,t_tag,direction (right=1).
    incomplete_outside = np.zeros([N + 1, N + 1, T, T, 2])  # s, t, s_tag,direction (right=1).
    complete_outside[0][N][0][1] = 1
    for l in xrange(N, 0, -1):
        for s in xrange(N - l + 1):
            t = s + l
            for s_tag in range(T):
                for q_tag in range(T):
                    for q in range(s + 1, t + 1):
                        incomplete_outside[s, q, s_tag, q_tag, 1] += complete_outside[s, t, s_tag, 1] * complete_inside[
                            q, t, q_tag, 1]
            for q_tag in range(T):
                for t_tag in range(T):
                    for q in range(s, t):
                        incomplete_outside[q, t, q_tag, t_tag, 0] += complete_outside[s, t, t_tag, 0] * complete_inside[
                            s, q, q_tag, 0]
            for s_tag in range(T):
                for t_tag in range(T):
                    for q in range(s, t):
                        complete_outside[s, q, s_tag, 1] += incomplete_outside[s, t, s_tag, t_tag, 1] * complete_inside[
                            q + 1, t, q_tag, 0] * scores[s, t, s_tag, t_tag]
                        # c = complete_outside[s, q, s_tag, 1]
                        # a = incomplete_outside[s,t,s_tag,t_tag,0]
                        # b = complete_inside[q+1,t,q_tag,0]
                        # r = scores[t,s,t_tag,s_tag]

                        complete_outside[s, q, s_tag, 1] += incomplete_outside[s, t, s_tag, t_tag, 0] * complete_inside[
                            q + 1, t, q_tag, 0] * scores[t, s, t_tag, s_tag]
                        # c = incomplete_outside[s, t, s_tag, t_tag, 0]
                        # a = incomplete_outside[s,t,s_tag,t_tag,1]
                        # b = complete_inside[s,q,s_tag,1]
                        # r = scores[s,t,s_tag,t_tag]

                        complete_outside[q + 1, t, t_tag, 0] += incomplete_outside[s, t, s_tag, t_tag, 1] * \
                                                                complete_inside[s, q, s_tag, 1] * scores[
                                                                    s, t, s_tag, t_tag]
                        # c = complete_outside[q + 1, t, t_tag, 0]
                        # a = incomplete_outside[s,t,s_tag,t_tag,0]
                        # b = complete_inside[s,q,s_tag,1]
                        # r = scores[t,s,t_tag,s_tag]

                        complete_outside[q + 1, t, t_tag, 0] += incomplete_outside[s, t, s_tag, t_tag, 0] * \
                                                                complete_inside[s, q, s_tag, 1] * scores[
                                                                    t, s, t_tag, s_tag]
                        # c = complete_outside[q + 1, t, t_tag, 0]

            for q_tag in range(T):
                for t_tag in range(T):
                    for q in range(s, t):
                        a = complete_outside[s, t, t_tag, 0]
                        b = incomplete_inside[q, t, q_tag, t_tag, 0]

                        complete_outside[s, q, q_tag, 0] += complete_outside[s, t, t_tag, 0] * incomplete_inside[
                            q, t, q_tag, t_tag, 0]
                        c = complete_outside[s, q, q_tag, 0]
            for s_tag in range(T):
                for q_tag in range(T):
                    for q in range(s + 1, t + 1):
                        a = complete_outside[s, t, t_tag, 1]
                        b = incomplete_inside[s, q, s_tag, q_tag, 1]

                        complete_outside[q, t, q_tag, 1] += complete_outside[s, t, s_tag, 1] * incomplete_inside[
                            s, q, s_tag, q_tag, 1]
                        c = complete_outside[q, t, q_tag, 1]

    outside_score = (complete_outside, incomplete_outside)
    return outside_score


def batch_parse(batch_scores):
    batch_size, sentence_length, _, tag_num, _ = batch_scores.shape
    # CYK table
    complete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num))
    incomplete_table = np.zeros((batch_size, sentence_length * sentence_length * 2, tag_num, tag_num))
    # backtrack table
    complete_backtrack = -np.ones((batch_size, sentence_length * sentence_length * 2, tag_num), dtype=int)
    incomplete_backtrack = -np.ones((batch_size, sentence_length * sentence_length * 2, tag_num, tag_num), dtype=int)
    # span index table, to avoid redundant iterations
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length)
    # initial basic complete spans
    for ii in basic_span:
        complete_table[:, ii, :] = 0.0

    for ij in ijss:
        (l, r, dir) = id_2_span[ij]
        num_ki = len(ikis[ij])
        ik_ci = complete_table[:, ikis[ij], :].reshape(batch_size, num_ki, tag_num, 1)
        kj_ci = complete_table[:, kjis[ij], :].reshape(batch_size, num_ki, 1, tag_num)
        # construct incomplete spans
        if dir == 0:
            span_i = ik_ci + kj_ci + batch_scores[:, r, l, :, :].swapaxes(1, 2).reshape(batch_size, 1, tag_num, tag_num)
        else:
            span_i = ik_ci + kj_ci + batch_scores[:, l, r, :, :].reshape(batch_size, 1, tag_num, tag_num)

        incomplete_table[:, ij, :, :] = np.max(span_i, axis=1)
        max_idx = np.argmax(span_i, axis=1)
        incomplete_backtrack[:, ij, :, :] = max_idx

        num_kc = len(ikcs[ij])
        if dir == 0:
            ik_cc = complete_table[:, ikcs[ij], :].reshape(batch_size, num_kc, tag_num, 1)
            kj_ic = incomplete_table[:, kjcs[ij], :, :].reshape(batch_size, num_kc, tag_num, tag_num)
            span_c = ik_cc + kj_ic
            span_c = span_c.reshape(batch_size, num_kc * tag_num, tag_num)
        else:
            ik_ic = incomplete_table[:, ikcs[ij], :, :].reshape(batch_size, num_kc, tag_num, tag_num)
            kj_cc = complete_table[:, kjcs[ij], :].reshape(batch_size, num_kc, 1, tag_num)
            span_c = ik_ic + kj_cc
            span_c = span_c.swapaxes(2,3).reshape(batch_size, num_kc * tag_num, tag_num)
        complete_table[:, ij, :] = np.max(span_c, axis=1)
        max_idx = np.argmax(span_c, axis=1)
        complete_backtrack[:, ij, :] = max_idx

    tags = np.zeros((batch_size, sentence_length)).astype(int)
    heads = -np.ones((batch_size, sentence_length))
    root_id = span_2_id[(0, sentence_length - 1, 1)]
    for s in range(batch_size):
        batch_backtracking(incomplete_backtrack, complete_backtrack, root_id, 0, 0, 1, tags, heads,
                           ikcs, ikis, kjcs, kjis, id_2_span,span_2_id,tag_num,s)

    return (heads, tags)


def batch_backtracking(incomplete_backtrack, complete_backtrack, span_id, l_tag, r_tag, complete, tags, heads,
                        ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,tag_num, sen_id):
    #print span_id
    (l, r, dir) = id_2_span[span_id]
    if l == r:
        return
    if complete:
        if dir == 0:
            k = complete_backtrack[sen_id, span_id, r_tag]
            #print 'k is ', k, ' complete left'
            k_span,k_tag = utils.get_index(tag_num,k)
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, 0, k_tag, 1, tags, heads,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, k_tag, r_tag, 0, tags, heads,
                                ikcs, ikis, kjcs, kjis, id_2_span, span_2_id,tag_num, sen_id)
            return
        else:
            num_k = len(ikcs[span_id])
            k = complete_backtrack[sen_id, span_id, l_tag]
            #print 'k is ', k, ' complete right'
            k_span, k_tag = utils.get_index(tag_num, k)
            left_span_id = ikcs[span_id][k_span]
            right_span_id = kjcs[span_id][k_span]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, k_tag, 0, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, k_tag, 0, 1, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
    else:
        if dir == 0:

            k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag]
            #print 'k is ', k, ' incomplete left'
            heads[sen_id, l] = r
            tags[sen_id, l] = l_tag
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, 0, 1, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, r_tag, 1, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
        else:
            k = incomplete_backtrack[sen_id, span_id, l_tag, r_tag]
            #print 'k is', k, ' incomplete right'
            heads[sen_id, r] = l
            tags[sen_id, r] = r_tag
            left_span_id = ikis[span_id][k]
            right_span_id = kjis[span_id][k]
            batch_backtracking(incomplete_backtrack, complete_backtrack, left_span_id, l_tag, 0, 1, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            batch_backtracking(incomplete_backtrack, complete_backtrack, right_span_id, 0, r_tag, 1, tags, heads,
                               ikcs, ikis, kjcs, kjis, id_2_span, span_2_id, tag_num, sen_id)
            return
