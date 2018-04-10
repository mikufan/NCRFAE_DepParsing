from collections import Counter
import re
from itertools import groupby
import torch.autograd as autograd
import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import *
import itertools


class ConllEntry:
    def __init__(self, id, form, lemma, cpos, pos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


class Feature:
    def __init__(self, id, head, modifier, head_id, mod_id, dist, dir):
        self.id = id
        self.head = head
        self.modifier = modifier
        self.head_id = head_id
        self.mod_id = mod_id
        self.dist = dist
        self.dir = dir


class FeatureLookUp:
    def __init__(self):
        self.feat_set = set()
        self.feat_map = {}
        self.id_map = {}
        self.feat_num = 0

    def update(self, feat):
        if feat not in self.feat_set:
            self.feat_set.add(feat)
            self.feat_num += 1
            id = self.feat_num - 1
            new_feat = Feature(id, feat[0], feat[1], feat[2], feat[3], feat[4], feat[5])
            self.feat_map[id] = new_feat
            self.id_map[feat] = id

    def find_id(self, feat):
        if feat not in self.feat_set:
            return None
        else:
            return self.id_map[feat]

    def find_feature_with_id(self, id):
        if id > self.feat_num - 1:
            return None
        else:
            return self.feat_map[id]

    def find_feature(self, feat):
        if feat not in self.feat_set:
            return None
        else:
            id = self.find_id(feat)
            return self.find_feature_with_id(id)


class data_sentence:
    def __init__(self, id, entry_list):
        self.id = id
        self.entries = entry_list
        self.size = len(entry_list)

    def set_data_list(self, words, pos):
        word_list = list()
        pos_list = list()
        for entry in self.entries:
            if entry.norm in words.keys():
                word_list.append(words[entry.norm])
            else:
                word_list.append(words['<UNKNOWN>'])
            if entry.pos in pos.keys():
                pos_list.append(pos[entry.pos])
            else:
                pos_list.append(pos['<UNKNOWN-POS>'])
        return word_list, pos_list

    def __str__(self):
        return '\t'.join([e for e in self.entries])


def traverse_feat(conll_path, tag_map, distdim):
    flookup = FeatureLookUp()
    sentence_id = 0
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            max_dist = distdim - 1
            for i, hnode in enumerate(sentence):
                for j, mnode in enumerate(sentence):
                    if isinstance(hnode, ConllEntry) and isinstance(mnode, ConllEntry):
                        if i == j:
                            continue
                        if j == 0:
                            continue
                        pos_feat_h = hnode.pos
                        pos_feat_m = mnode.pos
                        dist = abs(i - j)
                        if dist > max_dist:
                            dist = max_dist
                        if i < j:
                            dir = 1
                        else:
                            dir = 0
                        num_subtag_h = tag_map[pos_feat_h]
                        num_subtag_m = tag_map[pos_feat_m]
                        # for id_h in range(num_subtag_h):
                        #     u_feat_h = (pos_feat_h, None,id_h, None, dist, dir)
                        #     flookup.update(u_feat_h)
                        # for id_m in range(num_subtag_m):
                        #     u_feat_m = (None, pos_feat_m, None, id_m, dist, dir)
                        #     flookup.update(u_feat_m)
                        for id_h in range(num_subtag_h):
                            for id_m in range(num_subtag_m):
                                b_feat = (pos_feat_h, pos_feat_m, id_h, id_m, dist, dir)
                                flookup.update(b_feat)

                                # if i - 1 > 0:
                                #     pos_feat_h_lc = sentence[i - 1].pos
                                #     feat_h_lc = (
                                #         pos_feat_h, pos_feat_m, pos_feat_h_lc, id_h, id_m, dist, dir)
                                #     flookup.update(feat_h_lc)
                                # if i + 1 < len(sentence):
                                #     pos_feat_h_rc = sentence[i + 1].pos
                                #     feat_h_rc = (
                                #         pos_feat_h, pos_feat_m, pos_feat_h_rc, id_h, id_m, dist, dir)
                                #     flookup.update(feat_h_rc)
                                # if j - 1 > 0:
                                #     pos_feat_m_lc = sentence[j - 1].pos
                                #     feat_m_lc = (
                                #         pos_feat_h, pos_feat_m, pos_feat_m_lc, id_h, id_m, dist, dir)
                                #     flookup.update(feat_m_lc)
                                # if j + 1 < len(sentence):
                                #     pos_feat_m_rc = sentence[j + 1].pos
                                #     feat_m_rc = (
                                #         pos_feat_h, pos_feat_m, pos_feat_m_rc, id_h, id_m, dist, dir)
                                #     flookup.update(feat_m_rc)
            sentence_id += 1
    print 'number of features', len(flookup.feat_map)
    return flookup


def update_features(featureSet, data_sentence):
    entries = data_sentence.entries
    for i in range(data_sentence.size):
        for j in range(data_sentence.size):
            if i == j:
                continue
            if i > j:
                dir = 0
            else:
                dir = 1
            if abs(i - j) > 5:
                dist = 6
            else:
                dist = abs(i - j)

            if i > j:
                small = j
                large = i
            else:
                small = i
                large = j
            if small > 0:
                p_left = entries[small - 1].pos
            else:
                p_left = "STR"
            if large < len(entries) - 1:
                p_right = entries[large + 1].pos
            else:
                p_right = "END"
            if small < large - 1:
                p_left_right = entries[small + 1].pos
            else:
                p_left_right = "MID"
            if large > small + 1:
                p_right_left = entries[large - 1].pos
            else:
                p_right_left = "MID"
            left_unary = (entries[small].pos, dir, dist)
            right_unary = (entries[large].pos, dir, dist)
            binary = (entries[small].pos, entries[large].pos, dir, dist)
            h_left_trigram = (entries[small].pos, p_left, entries[large].pos, dir, dist)
            h_right_trigram = (entries[small].pos, p_left_right, entries[large].pos, dir, dist)
            m_left_trigram = (entries[small].pos, entries[large].pos, p_right_left, dir, dist)
            m_right_trigram = (entries[small].pos, entries[large].pos, p_right, dir, dist)
            featureSet.add(left_unary)
            featureSet.add(right_unary)
            featureSet.add(binary)
            featureSet.add(h_left_trigram)
            featureSet.add(h_right_trigram)
            featureSet.add(m_left_trigram)
            featureSet.add(m_right_trigram)


def read_data(conll_path, isPredict):
    sentences = []
    if not isPredict:
        wordsCount = Counter()
        posCount = Counter()
        s_counter = 0
        with open(conll_path, 'r') as conllFP:
            for sentence in read_conll(conllFP):
                wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        wordsCount['<UNKNOWN>'] = 0
        posCount['<UNKNOWN-POS>'] = 0
        posCount['<START>'] = 1
        posCount['<END>'] = 2
        return {w: i for i, w in enumerate(wordsCount.keys())}, {p: i for i, p in enumerate(
            posCount.keys())}, sentences
    else:
        with open(conll_path, 'r') as conllFP:
            s_counter = 0
            for sentence in read_conll(conllFP):
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                s_counter += 1
        return sentences


def read_sparse_data(conll_path, isPredict):
    sentences = []
    if not isPredict:
        wordsCount = Counter()
        posCount = Counter()
        featureSet = set()
        s_counter = 0
        with open(conll_path, 'r') as conllFP:
            for sentence in read_conll(conllFP):
                wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
                posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
                ds = data_sentence(s_counter, sentence)
                sentences.append(ds)
                update_features(featureSet, ds)
                s_counter += 1
        wordsCount['<UNKNOWN>'] = 0
        posCount['<UNKNOWN-POS>'] = 0
        featureSet.add('<UNKNOWN-FEATS>')
        return {w: i for i, w in enumerate(wordsCount.keys())}, {p: i for i, p in enumerate(
            posCount.keys())}, {f: i for i, f in enumerate(featureSet)}, sentences


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-CPOS', 'ROOT-POS', '_', -1, 'rroot', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5],
                                         int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def eval(predicted, gold, test_path, log_path, epoch):
    correct_counter = 0
    total_counter = 0
    for s in range(len(gold)):
        ps = predicted[s][0]
        gs = gold[s]
        for i, e in enumerate(gs.entries):
            if i == 0:
                continue
            if ps[i] == e.parent_id:
                correct_counter += 1
            h = gs.entries[e.parent_id].pos
            # if (h,e.pos) in prior_set and ps[i] != e.parent_id:
            #     print "Wrong prediction for "+ h +" " +e.pos
            total_counter += 1
    accuracy = float(correct_counter) / total_counter
    print 'UAS is ' + str(accuracy * 100) + '%'
    f_w = open(test_path, 'w')
    for s, sentence in enumerate(gold):
        for entry in sentence.entries:
            f_w.write(str(entry.norm) + ' ')
        f_w.write('\n')
        for entry in sentence.entries:
            f_w.write(str(entry.pos) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(sentence.entries[i].parent_id) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(int(predicted[s][1][i])) + ' ')
        f_w.write('\n')
        for i in range(len(sentence.entries)):
            f_w.write(str(int(predicted[s][0][i])) + ' ')
        f_w.write('\n')
        f_w.write('\n')
    f_w.close()
    if epoch == 0:
        log = open(log_path, 'w')
        # log.write("UAS for epoch " + str(epoch))
        # log.write('\n')
        # log.write('\n')
        log.write(str(accuracy))
        log.write('\n')
        log.write('\n')
    else:
        log = open(log_path, 'a')
        # log.write("UAS for epoch " + str(epoch))
        # log.write('\n')
        # log.write('\n')
        log.write(str(accuracy))
        log.write('\n')
        log.write('\n')


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence.entries:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


# Map each tag to the number of subtags
def round_tag(posCount, tag_level=0):
    tag_map = {}
    max_tag_num = 0
    for t in posCount.keys():
        c = posCount[t]
        if c > tag_level and t != 'ROOT-POS':
            tag_map[t] = 4
            max_tag_num = 4
        elif c > tag_level / 4 and t != 'ROOT-POS':
            tag_map[t] = 2
            if max_tag_num < 2:
                max_tag_num = 2
        else:
            tag_map[t] = 1
            if max_tag_num < 1:
                max_tag_num = 1

    return tag_map


def construct_batch_data(data_list, batch_size):
    data_list.sort(key=lambda x: len(x[0]))
    grouped = [list(g) for k, g in groupby(data_list, lambda s: len(s[0]))]
    batch_data = []
    for group in grouped:
        sub_batch_data = get_batch_data(group, batch_size)
        batch_data.extend(sub_batch_data)
    return batch_data

def construct_update_batch_data(data_list,batch_size):
    random.shuffle(data_list)
    batch_data = []
    len_datas = len(data_list)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1
    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(data_list[start_idx:end_idx])
    return batch_data


def get_batch_data(grouped_data, batch_size):
    batch_data = []
    len_datas = len(grouped_data)
    num_batch = len_datas // batch_size
    if not len_datas % batch_size == 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min(len_datas, (i + 1) * batch_size)
        batch_data.append(grouped_data[start_idx:end_idx])
    return batch_data


def list2Variable(list, gpu_flag):
    list = torch.LongTensor(list)
    if gpu_flag == -1:
        list_var = autograd.Variable(list)
    else:
        list_var = autograd.Variable(list).cuda()
    return list_var


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


def logaddexp(a, b):
    max_ab = torch.max(a, b)
    max_ab[~isfinite(max_ab)] = 0
    return torch.log(torch.add(torch.exp(a - max_ab), torch.exp(b - max_ab))) + max_ab


def isfinite(a):
    return (a != np.inf) & (a != -np.inf) & (a != np.nan) & (a != -np.nan)


def logsumexp(a, axis=None):
    a_max = amax(a, axis=axis, keepdim=True)
    a_max[~isfinite(a_max)] = 0
    res = torch.log(asum(torch.exp(a - a_max), axis=axis, keepdim=True)) + a_max
    if isinstance(axis, tuple):
        for x in reversed(axis):
            res.squeeze_(x)
    else:
        res.squeeze_(axis)
    return res


def amax(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a, index = a.max(x, keepdim=keepdim)
    else:
        a, index = a.max(axis, keepdim=True)
    return a


def asum(a, axis=None, keepdim=False):
    if isinstance(axis, tuple):
        for x in reversed(axis):
            a = a.sum(x, keepdim=keepdim)
    else:
        a = a.sum(axis, keepdim=keepdim)
    return a


@memoize
def constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    for left_idx in range(sentence_length):
        for right_idx in range(left_idx, sentence_length):
            for dir in range(2):
                id_2_span[counter_id] = (left_idx, right_idx, dir)
                counter_id += 1

    span_2_id = {s: id for id, s in id_2_span.items()}

    for i in range(sentence_length):
        if i != 0:
            id = span_2_id.get((i, i, 0))
            basic_span.append(id)
        id = span_2_id.get((i, i, 1))
        basic_span.append(id)

    ijss = []
    ikcs = [[] for _ in range(counter_id)]
    ikis = [[] for _ in range(counter_id)]
    kjcs = [[] for _ in range(counter_id)]
    kjis = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for dir in range(2):
                ids = span_2_id[(i, j, dir)]
                for k in range(i, j + 1):
                    if dir == 0:
                        if k < j:
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir + 1)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir)]
                            kjis[ids].append(idri)
                            # one complete span,one incomplete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                    else:
                        if k < j and ((not (i == 0 and k != 0) and not multiroot) or multiroot):
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir - 1)]
                            kjis[ids].append(idri)
                        if k > i:
                            # one incomplete span,one complete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                ijss.append(ids)

    return span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span


def get_index(b, id):
    id_a = id // b
    id_b = id % b
    return (id_a, id_b)


def use_external_embedding(extrn_emb, vocab):
    to_augment = {}
    extrn_dim = 0
    for line in extrn_emb:
        line = line.split(' ')
        word = line[0]
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        extrn_dim = len(vector)
        if word in vocab.keys():
            to_augment[word] = vector
    return extrn_dim, to_augment


def build_new_emb(original_emb, to_augment, vocab):
    augmented = np.copy(original_emb)
    for w in to_augment.keys():
        w_idx = vocab[w]
        augmented[w_idx] = to_augment[w]
    return augmented


def construct_prior(prior_set, sentence, pos, tag_num, prior_weight):
    sentence_length = sentence.size
    s_prior = np.zeros((sentence_length, sentence_length, tag_num, tag_num))
    for i in range(sentence_length):
        for j in range(sentence_length):
            if i == j:
                continue
            if j == 0:
                continue
            h_pos = sentence.entries[i].pos
            m_pos = sentence.entries[j].pos
            tag_tuple = (h_pos, m_pos)
            if tag_tuple in prior_set:
                s_prior[i, j, :, :] = float(prior_weight) / sentence_length
                #s_prior[i, j, :, :] = float(prior_weight)
    return s_prior


def compute_trans(feat_type, batch_size, sentence_length, tag_num, feat_emb):
    if feat_type == 'sentence':
        feat_emb_h = feat_emb.unsqueeze(2)
        feat_emb_m = feat_emb_h.permute(0, 2, 1, 3)
        feat_emb_h = feat_emb_h.repeat(1, 1, sentence_length, 1)
        feat_emb_m = feat_emb_m.repeat(1, sentence_length, 1, 1)
        feat_emb_h = feat_emb_h.unsqueeze(3)
        feat_emb_h = feat_emb_h.unsqueeze(4)
        feat_emb_m = feat_emb_m.unsqueeze(3)
        feat_emb_m = feat_emb_m.unsqueeze(4)
        feat_emb_h = feat_emb_h.repeat(1, 1, 1, tag_num, tag_num, 1)
        feat_emb_m = feat_emb_m.repeat(1, 1, 1, tag_num, tag_num, 1)
        return feat_emb_h, feat_emb_m
    if feat_type == 'tag':
        feat_emb_h = feat_emb.unsqueeze(1)
        feat_emb_m = feat_emb_h.permute(1, 0, 2)
        feat_emb_h = feat_emb_h.repeat(1, tag_num, 1)
        feat_emb_m = feat_emb_m.repeat(tag_num, 1, 1)
        feat_emb_h = feat_emb_h.unsqueeze(0)
        feat_emb_h = feat_emb_h.unsqueeze(0)
        feat_emb_h = feat_emb_h.unsqueeze(0)
        feat_emb_m = feat_emb_m.unsqueeze(0)
        feat_emb_m = feat_emb_m.unsqueeze(0)
        feat_emb_m = feat_emb_m.unsqueeze(0)
        feat_emb_h = feat_emb_h.repeat(batch_size, sentence_length, sentence_length, 1, 1, 1)
        feat_emb_m = feat_emb_m.repeat(batch_size, sentence_length, sentence_length, 1, 1, 1)
        return feat_emb_h, feat_emb_m
    if feat_type == 'global':
        feat_emb = feat_emb.unsqueeze(2)
        feat_emb = feat_emb.unsqueeze(3)
        feat_emb = feat_emb.repeat(1, 1, tag_num, tag_num, 1)
        feat_emb = feat_emb.unsqueeze(0)
        feat_emb = feat_emb.repeat(batch_size, 1, 1, 1, 1, 1)
        return feat_emb
    if feat_type == 'trans':
        feat_emb = feat_emb.unsqueeze(3)
        feat_emb = feat_emb.unsqueeze(4)
        feat_emb = feat_emb.repeat(1, 1, 1, tag_num, tag_num, 1)
        return feat_emb


def init_weight(layer):
    if isinstance(layer, nn.Linear):
        xavier_uniform(layer.weight.data, 0.1)
        constant(layer.bias, 0)
    if isinstance(layer, nn.Embedding):
        xavier_uniform(layer.weight.data, 0.1)
    if isinstance(layer, nn.LSTM):
        for p in layer.parameters():
            if len(p.data.shape) > 1:
                xavier_uniform(p.data, 0.1)
            else:
                constant(p, 0)


def construct_trigram(s_pos, pos):
    trigram_list = list()
    for i in range(len(s_pos)):
        if i == 0:
            trigram = (pos['<START>'], s_pos[0], pos['<START>'])
        elif i == 1:
            if len(s_pos) > 2:
                right_gram = s_pos[2]
            else:
                right_gram = pos['<END>']
            trigram = (s_pos[0], s_pos[1], right_gram)
        elif i == len(s_pos) - 1:
            trigram = (s_pos[i - 1], s_pos[i], pos['<END>'])
        else:
            trigram = (s_pos[i - 1], s_pos[i], s_pos[i + 1])
        trigram_list.append(trigram)
    return trigram_list


def construct_feats(feats, s):
    feature_list = []
    entries = s.entries
    unknown_idx = feats['<UNKNOWN-FEATS>']
    for i in range(len(entries)):
        head_feature = []
        for j in range(len(entries)):
            single_feature = []
            # if j == 0:
            #     for k in range(7):
            #         single_feature.append(unknown_idx)
            #     head_feature.append(single_feature)
            #     continue
            if i == j:
                for k in range(7):
                    single_feature.append(unknown_idx)
                head_feature.append(single_feature)
                continue
            if i > j:
                dir = 0
            else:
                dir = 1
            if abs(i - j) > 5:
                dist = 6
            else:
                dist = abs(i - j)
            if i > j:
                small = j
                large = i
            else:
                small = i
                large = j
            if small > 0:
                p_left = entries[small - 1].pos
            else:
                p_left = "STR"
            if large < len(entries) - 1:
                p_right = entries[large + 1].pos
            else:
                p_right = "END"
            if small < large - 1:
                p_left_right = entries[small + 1].pos
            else:
                p_left_right = "MID"
            if large > small + 1:
                p_right_left = entries[large - 1].pos
            else:
                p_right_left = "MID"
            left_unary = (entries[small].pos, dir, dist)
            right_unary = (entries[large].pos, dir, dist)
            binary = (entries[small].pos, entries[large].pos, dir, dist)
            h_left_trigram = (entries[small].pos, p_left, entries[large].pos, dir, dist)
            h_right_trigram = (entries[small].pos, p_left_right, entries[large].pos, dir, dist)
            m_left_trigram = (entries[small].pos, entries[large].pos, p_right_left, dir, dist)
            m_right_trigram = (entries[small].pos, entries[large].pos, p_right, dir, dist)
            h_unary_idx = feats.get(left_unary, unknown_idx)
            m_unary_idx = feats.get(right_unary, unknown_idx)
            binary_idx = feats.get(binary, unknown_idx)
            h_left_trigram_idx = feats.get(h_left_trigram, unknown_idx)
            h_right_trigram_idx = feats.get(h_right_trigram, unknown_idx)
            m_left_trigram_idx = feats.get(m_left_trigram, unknown_idx)
            m_right_trigram_idx = feats.get(m_right_trigram, unknown_idx)
            single_feature = [h_unary_idx, m_unary_idx, binary_idx, h_left_trigram_idx, h_right_trigram_idx,
                              m_left_trigram_idx, m_right_trigram_idx]
            head_feature.append(single_feature)
        feature_list.append(head_feature)
    return feature_list


def get_state_code(i, j, k):
    return i * 4 + j * 2 + k


def is_valid_signature(sig):
    left_index, _, right_index, _, bL, bR, is_simple = sig
    if left_index == 0 and bL:
        return 0
    if right_index - left_index == 1:
        if is_simple:
            if bL and bR:
                return 0
            else:
                return 1
        else:
            return 0

    if bL and bR and is_simple:
        return 0
    return (bL == is_simple) or (bR == is_simple)


@memoize
def constituent_indexes(sent_len, is_multi_root=True):
    seed_spans = []
    base_left_spans = []
    crt_id = 0
    id_span_map = {}

    for left_index in range(sent_len):
        for right_index in range(left_index + 1, sent_len):
            for c in range(8):
                id_span_map[crt_id] = (left_index, right_index, c)
                crt_id += 1

    span_id_map = {v: k for k, v in id_span_map.items()}

    for i in range(1, sent_len):
        ids = span_id_map[(i - 1, i, get_state_code(0, 0, 1))]
        seed_spans.append(ids)

        ids = span_id_map[(i - 1, i, get_state_code(0, 1, 1))]
        base_left_spans.append(ids)

    base_right_spans = []
    for i in range(2, sent_len):
        ids = span_id_map[(i - 1, i, get_state_code(1, 0, 1))]
        base_right_spans.append(ids)

    ijss = []
    ikss = [[] for _ in range(crt_id)]
    kjss = [[] for _ in range(crt_id)]

    left_spans = set()
    right_spans = set()

    for length in range(2, sent_len):
        for i in range(1, sent_len - length + 1):
            j = i + length
            for (bl, br) in list(itertools.product([0, 1], repeat=2)):
                ids = span_id_map[(i - 1, j - 1, get_state_code(bl, br, 0))]

                for (b, s) in list(itertools.product([0, 1], repeat=2)):
                    for k in range(i + 1, j):
                        sig_left = (i - 1, None, k - 1, None, bl, b, 1)
                        sig_right = (k - 1, None, j - 1, None, 1 - b, br, s)

                        if is_valid_signature(sig_left) and is_valid_signature(sig_right):
                            if is_multi_root or ((i > 1) or (
                                                    (i == 1) and (j == sent_len) and (bl == 0) and (b == 1) and (
                                                br == 1)) or ((i == 1) and (k == 2) and (bl == 0) and (b == 0) and (
                                        br == 0))):
                                ids1 = span_id_map[(i - 1, k - 1, get_state_code(bl, b, 1))]
                                ikss[ids].append(ids1)
                                ids2 = span_id_map[(k - 1, j - 1, get_state_code(1 - b, br, s))]
                                kjss[ids].append(ids2)
                if len(ikss[ids]) >= 1:
                    ijss.append(ids)

            ids = span_id_map[(i - 1, j - 1, get_state_code(0, 1, 1))]
            ijss.append(ids)
            left_spans.add(ids)
            if i != 1:
                ids = span_id_map[(i - 1, j - 1, get_state_code(1, 0, 1))]
                ijss.append(ids)
                right_spans.add(ids)

    return seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map


def decoding_batch(weights, is_multi_root=True):
    batch_size, sentence_len, tags_dim, _, _ = weights.shape
    inside_table = np.empty(
        (batch_size, sentence_len * sentence_len * 8, tags_dim, tags_dim),
        dtype=np.float64)
    inside_table.fill(-np.inf)
    seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = constituent_indexes(
        sentence_len, is_multi_root)
    kbc = np.empty_like(inside_table, dtype=int)

    for ii in seed_spans:
        inside_table[:, ii, :, :] = 0.0
        kbc[:, ii, :, :] = -1

    for ii in base_right_spans:
        (l, r, c) = id_span_map[ii]
        swap_weights = np.swapaxes(weights, 2, 4)
        inside_table[:, ii, :, :] = swap_weights[:, r, :, l, :]
        kbc[:, ii, :, :] = -1

    for ii in base_left_spans:
        (l, r, c) = id_span_map[ii]
        inside_table[:, ii, :, :] = weights[:, l, :, r, :]
        kbc[:, ii, :, :] = -1

    for ij in ijss:
        (l, r, c) = id_span_map[ij]
        if ij in left_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            prob = inside_table[:, ids, :, :] + weights[:, l, :, r, :]
            inside_table[:, ij, :, :] = np.logaddexp(inside_table[:, ij, :, :], prob)
        elif ij in right_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            swap_weights = np.swapaxes(weights, 2, 4)
            prob = inside_table[:, ids, :, :] + swap_weights[:, r, :, l, :]
            inside_table[:, ij, :, :] = np.logaddexp(inside_table[:, ij, :, :], prob)
        else:
            num_k = len(ikss[ij])
            beta_ik, beta_kj = inside_table[:, ikss[ij], :, :], inside_table[:, kjss[ij], :, :]
            probs = beta_ik.reshape(batch_size, num_k, tags_dim, tags_dim, 1) + \
                    beta_kj.reshape(batch_size, num_k, 1, tags_dim, tags_dim)

            probs = probs.transpose(0, 2, 4, 1, 3).reshape(batch_size, tags_dim, tags_dim, -1)
            inside_table[:, ij, :, :] = np.max(probs, axis=3)
            kbc[:, ij, :, :] = np.argmax(probs, axis=3)

    id1 = span_id_map.get((0, sentence_len - 1, get_state_code(0, 1, 0)), -1)
    id2 = span_id_map.get((0, sentence_len - 1, get_state_code(0, 1, 1)), -1)

    score1 = inside_table[:, id1, 0, :].reshape(batch_size, 1, tags_dim)
    score2 = inside_table[:, id2, 0, :].reshape(batch_size, 1, tags_dim)

    root_prob1 = np.max(score1, axis=2)
    root_prob2 = np.max(score2, axis=2)

    best_latest_tag1 = np.argmax(score1, axis=2)
    best_latest_tag2 = np.argmax(score2, axis=2)

    root_prob = np.maximum(root_prob1, root_prob2)
    mask = np.equal(root_prob, root_prob1)
    best_latest_tag = best_latest_tag2
    best_latest_tag[mask] = best_latest_tag1[mask]

    root_id = np.empty((batch_size, 1), dtype=int)
    root_id.fill(id2)
    root_id[mask] = id1

    best_tags = back_trace_batch(kbc, best_latest_tag, root_id, sentence_len, tags_dim, is_multi_root)

    return root_prob, best_tags


def back_trace_batch(kbc, best_latest_tag, root_id, sentence_len, tags_dim, is_multi_root=True):
    seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = constituent_indexes(
        sentence_len, is_multi_root)
    batch_size, num_ctt, _, _ = kbc.shape
    in_tree = np.full((batch_size, num_ctt), -1)
    best_tags = np.full((batch_size, sentence_len), -1)
    best_tags[:, sentence_len - 1] = best_latest_tag[:, 0]
    best_tags[:, 0] = 0  # ROOT can only be 0
    for sent_id in range(batch_size):
        current_sent_root_id = root_id[sent_id][0]
        in_tree[sent_id, current_sent_root_id] = 1
        for ij in reversed(ijss):
            non = in_tree[sent_id, ij]
            (l, r, c) = id_span_map[ij]
            if non != -1:
                if ij in left_spans:
                    ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
                    in_tree[sent_id, ids] = 1
                elif ij in right_spans:
                    ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
                    in_tree[sent_id, ids] = 1
                else:
                    num_k = len(ikss[ij])
                    iks, kjs = ikss[ij], kjss[ij]
                    k, tag = np.unravel_index(kbc[sent_id, ij, best_tags[sent_id, l], best_tags[sent_id, r]],
                                              (num_k, tags_dim))
                    in_tree[sent_id, iks[k]] = 1
                    in_tree[sent_id, kjs[k]] = 1
                    (ll, lr, lc) = id_span_map[iks[k]]
                    best_tags[sent_id, lr] = tag
    return best_tags
