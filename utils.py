from collections import Counter
import re


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
    def __init__(self, id, head, modifier, context, head_id, mod_id, dist, dir, location):
        self.id = id
        self.head = head
        self.modifier = modifier
        self.context = context
        self.head_id = head_id
        self.mod_id = mod_id
        self.dist = dist
        self.dir = dir
        self.location = location
        self.weight = None


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
            new_feat = Feature(id, feat[0], feat[1], feat[2], feat[3], feat[4], feat[5], feat[6], feat[7])
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
    def __init__(self,id,entry_list):
        self.id = id
        self.entries = entry_list
        self.size = len(entry_list)

def traverse_feat(conll_path, tag_map):
    flookup = FeatureLookUp()
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            if len(sentence) < 6:
                max_dist = len(sentence) - 1
            else:
                max_dist = 5
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
                        for id_h in range(num_subtag_h):
                            u_feat_h = (pos_feat_h, None, None, id_h, None, dist, dir, 'h')
                            flookup.update(u_feat_h)
                        for id_m in range(num_subtag_m):
                            u_feat_m = (None, pos_feat_m, None, None, id_m, dist, dir, 'm')
                            flookup.update(u_feat_m)
                        for id_h in range(num_subtag_h):
                            for id_m in range(num_subtag_m):
                                b_feat = (pos_feat_h, pos_feat_m, None,id_h, id_m, dist, dir, None)
                                flookup.update(b_feat)
                                if i - 1 > 0:
                                    pos_feat_h_lc = sentence[i - 1].pos
                                    feat_h_lc = (
                                        pos_feat_h, pos_feat_m, pos_feat_h_lc, id_h, id_m, dist, dir,
                                        'h')
                                    flookup.update(feat_h_lc)
                                if i + 1 < len(sentence):
                                    pos_feat_h_rc = sentence[i + 1].pos
                                    feat_h_rc = (
                                        pos_feat_h, pos_feat_m, pos_feat_h_rc, id_h, id_m, dist, dir,
                                        'h')
                                    flookup.update(feat_h_rc)
                                if j - 1 > 0:
                                    pos_feat_m_lc = sentence[j - 1].pos
                                    feat_m_lc = (
                                        pos_feat_h, pos_feat_m, pos_feat_m_lc, id_h, id_m, dist, dir, 'm')
                                    flookup.update(feat_m_lc)
                                if j + 1 < len(sentence):
                                    pos_feat_m_rc = sentence[j + 1].pos
                                    feat_m_rc = (
                                        pos_feat_h, pos_feat_m, pos_feat_m_rc, id_h, id_m, dist, dir, 'm')
                                    flookup.update(feat_m_rc)
    return flookup


def read_data(conll_path):
    wordsCount = Counter()
    posCount = Counter()
    sentences = []
    s_counter = 0
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            ds = data_sentence(s_counter,sentence)
            sentences.append(ds)
            s_counter+=1
    return wordsCount, {w: i for i, w in enumerate(wordsCount.keys())}, posCount.keys(), posCount, sentences


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


def write_conll(fn, conll_gen):
    with open(fn, 'w') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + '\n')
            fh.write('\n')


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


# Map each tag to the number of subtags
def round_tag(posCount, tag_level= 0):
    tag_map = {}
    for t in posCount.keys():
        c = posCount[t]
        if c > tag_level and t != 'ROOT-POS':
            tag_map[t] = 4
        elif c > tag_level / 4 and t != 'ROOT-POS':
            tag_map[t] = 2
        else:
            tag_map[t] = 1
    return tag_map
