from collections import Counter
import re
from itertools import groupby
import torch.autograd as autograd
import torch
import numpy as np


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

# def find_trans(sentences,pos,tag_num):
#     pos_num = len(pos.keys())
#     trans_map = np.zeros((pos_num*tag_num,pos_num*tag_num),dtype = int)
#     for i in range((pos_num*tag_num)*(pos_num*tag_num)):
#         trans_map[i/(pos_num*tag_num)][i%(pos_num*tag_num)] = i
#     trans_list = list()
#     for sentence in sentences:
#         s_trans = list()
#         for i,h_entry in enumerate(sentence):
#             h_trans_list = list()
#             for j,m_entry in enumerate(sentence):
#                 h_tag_num = pos[h_entry.pos]
#                 m_tag_num = pos[m_entry.pos]
#                 trans_idx = h_tag_num
#
#     return


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


def eval(predicted, gold, test_path,prior_set):
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
def constituent_index(sentence_length):
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
                        if k < j:
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

def use_external_embedding(extrn_emb,vocab):
    to_augment = {}
    extrn_dim = 0
    for line in extrn_emb:
        line = line.split(' ')
        word = line[0]
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        extrn_dim = len(vector)
        if word in vocab.keys():
            to_augment[word] = vector
    return extrn_dim,to_augment

def build_new_emb(original_emb,to_augment,vocab):
    augmented = np.copy(original_emb)
    for w in to_augment.keys():
        w_idx = vocab[w]
        augmented[w_idx] = to_augment[w]
    return augmented

def construct_prior(prior_set,sentence,pos,tag_num,prior_weight):
    sentence_length = sentence.size
    s_prior = np.zeros((sentence_length,sentence_length,tag_num,tag_num))
    for i in range(sentence_length):
        for j in range(sentence_length):
            if i == j:
                continue
            if j == 0:
                continue
            h_pos = sentence.entries[i].pos
            m_pos = sentence.entries[j].pos
            tag_tuple = (h_pos,m_pos)
            if tag_tuple in prior_set:
                s_prior[i,j,:,:] = prior_weight
    return s_prior

def compute_trans(feat_type,batch_size,sentence_length,tag_num,feat_emb):
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
        return feat_emb_h,feat_emb_m
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
        return feat_emb_h,feat_emb_m
    if feat_type == 'global':
        feat_emb = feat_emb.unsqueeze(2)
        feat_emb = feat_emb.unsqueeze(3)
        feat_emb = feat_emb.repeat(1, 1, tag_num, tag_num, 1)
        feat_emb = feat_emb.unsqueeze(0)
        feat_emb = feat_emb.repeat(batch_size, 1, 1, 1, 1, 1)
        return feat_emb


