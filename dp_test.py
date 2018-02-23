import eisner_parser
import numpy as np
import eisner_layer as EL
from torch.nn.init import *
import torch.autograd
import utils
import dt_pl_model
import pickle
import torch.autograd as autograd
import itertools


def construct_mask(batch_size, sentence_length,tag_num):
    masks = np.zeros((batch_size, sentence_length, sentence_length, tag_num, tag_num))
    masks[:, :, 0, :, :] = 1
    for i in range(sentence_length):
        masks[:, i, i, :, :] = 1
    masks = masks.astype(int)
    mask_var = Variable(torch.ByteTensor(masks))
    return mask_var



def diff(outside_table,inside_table,batch_size,sentence_length,tag_num,partition_score):

    counts = inside_table[1] + outside_table[1]
    pseudo_count = torch.DoubleTensor(batch_size, sentence_length, sentence_length, tag_num,
                                      tag_num)
    pseudo_count.fill_(0.0)
    span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(sentence_length)

    for l in range(sentence_length):
        for r in range(sentence_length):
            for dir in range(2):
                span_id = span_2_id.get((l, r, dir))
            if span_id is not None:
                if dir == 0:
                    pseudo_count[:, r, l, :, :] = counts[:, span_id, :, :]
                else:
                    pseudo_count[:, l, r, :, :] = counts[:, span_id, :, :]
    mius = pseudo_count - partition_score.contiguous().view(batch_size, 1, 1, 1, 1)
    diff = torch.exp(mius)
    #if mask is not None:
        #diff = diff.masked_fill_(mask, 0.0)
    return diff

def backward(output,outside_table,inside_table,batch_size,sentence_length,tag_num):
    mius = diff(outside_table,inside_table,batch_size,sentence_length,tag_num,output)
    grad_output = output.contiguous().view(batch_size, 1, 1, 1, 1)
    gradient = mius * grad_output
    batch_size, sent_len, _, tag_dim, _ = gradient.size()

    gradient[:, 0, :, 1:, :].fill_(0.0)
    gradient[:, :, 0, :, :].fill_(0.0)
    for i in range(sentence_length):
        gradient[:, i, i, :, :].fill_(0.0)
    return gradient

def  check_gradient(delta, inputs, sentence_len, batch_size=1):
    #with open('output/params.pickle', 'r') as paramsfp:
        #w2i, pos, stored_opt = pickle.load(paramsfp)
    #dtpl = dt_pl_model.dt_paralell_model(sentence_len, 2, batch_size, True)
    #llh = dt_pl_model(inputs)
    eisner = EL.eisner_layer(sentence_length, 2, batch_size)
    partition = eisner(inputs)
    print partition
    prev_grad = np.ones((batch_size, 1, 1, 1, 1))
    prev_grad = torch.DoubleTensor([prev_grad])
    grad = eisner.backward(prev_grad)
    for batch_id in range(batch_size):
        for i in range(sentence_len):
            for j in range(sentence_len):
                for k in range(2):
                    for l in range(2):
                        weight1 = inputs.clone()
                        weight2 = inputs.clone()
                        weight1[batch_id][i][j][k][l].data += delta
                        weight2[batch_id][i][j][k][l].data -= delta
                        llh1 = eisner(weight1).data.numpy()
                        llh2 = eisner(weight2).data.numpy()
                        real_grad = (llh1[batch_id] - llh2[batch_id]) / (2 * delta) * 1
                        grad_cal = grad[batch_id, i, j, k, l]

                        if math.fabs(real_grad - grad_cal) > 1e-6:
                            print("[{0}, {1}, {2}, {3}, {4}] : {5} {6} {7} ".format(batch_id, i, j, k, l, real_grad, grad_cal, math.fabs(real_grad - grad_cal)))
                            #exit(-1)
    print("PASS")

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


def test_constituent_indexes(sent_len, is_multi_root=True):
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


def dp_inside_batch(batch_size,sentence_len,tags_dim,weights):
    inside_table = torch.DoubleTensor(batch_size, sentence_len * sentence_len * 8, tags_dim, tags_dim)
    inside_table.fill_(-np.inf)
    if torch.cuda.is_available():
        inside_table = inside_table.cuda()
    m = sentence_len
    seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = test_constituent_indexes(
            m, True)

    for ii in seed_spans:
        inside_table[:, ii, :, :] = 0.0

    for ii in base_right_spans:
        (l, r, c) = id_span_map[ii]
        swap_weights = weights.permute(0, 1, 4, 3, 2)
        inside_table[:, ii, :, :] = swap_weights[:, r, :, l, :]

    for ii in base_left_spans:
        (l, r, c) = id_span_map[ii]
        inside_table[:, ii, :, :] = weights[:, l, :, r, :]

    for ij in ijss:
        (l, r, c) = id_span_map[ij]
        if ij in left_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            prob = inside_table[:, ids, :, :] + weights[:, l, :, r, :]
            inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], prob)
        elif ij in right_spans:
            ids = span_id_map.get((l, r, get_state_code(0, 0, 0)), -1)
            swap_weights = weights.permute(0, 1, 4, 3, 2)
            prob = inside_table[:, ids, :, :] + swap_weights[:, r, :, l, :]
            inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], prob)
        else:
            num_k = len(ikss[ij])
            beta_ik, beta_kj = inside_table[:, ikss[ij], :, :], inside_table[:, kjss[ij], :, :]
            probs = beta_ik.contiguous().view(batch_size, num_k, tags_dim, tags_dim, 1) +\
                        beta_kj.contiguous().view(batch_size, num_k, 1, tags_dim, tags_dim)
            probs = utils.logsumexp(probs, axis=(1, 3))
            inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], probs)

    id1 = span_id_map.get((0, m - 1, get_state_code(0, 1, 0)), -1)
    id2 = span_id_map.get((0, m - 1, get_state_code(0, 1, 1)), -1)

    score1 = inside_table[:, id1, 0, :].contiguous().view(batch_size, 1, tags_dim)
    score2 = inside_table[:, id2, 0, :].contiguous().view(batch_size, 1, tags_dim)
    ll = utils.logaddexp(utils.logsumexp(score1, axis=2), utils.logsumexp(score2, axis=2))
    return inside_table, ll


scores = np.zeros((4, 4, 2, 2))
scores[0, 1, 0, 0] = 0.3
scores[0, 1, 0, 1] = 0.2
scores[0, 2, 0, 0] = 0.6
scores[0, 2, 0, 1] = 0.7
scores[0, 3, 0, 0] = 0.4
scores[0, 3, 0, 1] = 0.3
scores[1, 2, 0, 0] = 0.4
scores[1, 2, 0, 1] = 0.5
scores[1, 3, 0, 0] = 0.6
scores[1, 3, 0, 1] = 0.5
scores[1, 2, 1, 0] = 0.3
scores[1, 2, 1, 1] = 0.4
scores[1, 3, 1, 0] = 0.5
scores[1, 3, 1, 1] = 0.4
scores[2, 1, 0, 0] = 0.7
scores[2, 1, 0, 1] = 0.6
scores[2, 3, 0, 0] = 0.7
scores[2, 3, 0, 1] = 0.6
scores[2, 1, 1, 0] = 0.8
scores[2, 1, 1, 1] = 0.7
scores[2, 3, 1, 0] = 0.8
scores[2, 3, 1, 1] = 0.7
scores[3, 1, 0, 0] = 0.6
scores[3, 1, 0, 1] = 0.5
scores[3, 2, 0, 0] = 0.4
scores[3, 2, 0, 1] = 0.5
scores[3, 1, 1, 0] = 0.5
scores[3, 1, 1, 1] = 0.4
scores[3, 2, 1, 0] = 0.3
scores[3, 2, 1, 1] = 0.4

#best_parse = eisner_parser.parse_proj(scores)
#sentence_score, inside_scores = eisner_parser.partition_inside(Variable(torch.FloatTensor(scores)))
#outside_score = eisner_parser.partition_outside(inside_scores, scores)
best_parse = eisner_parser.parse_proj(scores)
print best_parse
sentence_score,inside_score = eisner_parser.partition_inside(scores)
print sentence_score
#partition_out = eisner_parser.partition_outside(inside_score,scores)
trans_score = scores.transpose(1,0,2,3)
print eisner_parser.parse_proj(trans_score)
new_score = []
new_score.append(scores)
new_score.append(trans_score)
new_score = np.array(new_score)



#scores = np.tile(scores,(2,1,1,1,1))

#best_parse = eisner_parser.parse_proj(scores)
best_parse = eisner_parser.batch_parse(new_score)
#batch_size,sentence_length,_,tag_num,_ = scores.shape
#el = EL.eisner_layer(sentence_length,tag_num,batch_size)
#scores = torch.FloatTensor(scores)
#inside_table,sentence_score = el.batch_inside(scores)
#print sentence_score
#outside_table = el.batch_outside(inside_table,scores)
#diff(outside_table,inside_table,batch_size,sentence_length,tag_num,sentence_score)
#grad  = backward(sentence_score,outside_table,inside_table,batch_size,sentence_length,tag_num)
#print grad
new_score = torch.DoubleTensor(new_score)
weight = new_score.permute(0,1,3,2,4)
batch_size,sentence_length,_,_,_ = new_score.shape
mask = construct_mask(batch_size,sentence_length,2)
new_score = autograd.Variable(new_score)
#new_score = new_score.masked_fill(mask, -np.inf)
check_gradient(1e-6,new_score,sentence_length,batch_size)
weights = torch.DoubleTensor(scores).permute(0,2,1,3)
weights = weights.unsqueeze(0)
inside_table,ll = dp_inside_batch(1,sentence_length,2,weights)
print ll
short_score = np.zeros((3,3,2,2))
short_score[0,1,0,0] = 0.3
short_score[0,1,0,1] = 0.2
short_score[0,2,0,0] = 0.6
short_score[0,2,0,1] = 0.7
short_score[1,2,0,0] = 0.4
short_score[1,2,0,1] = 0.5
short_score[1,2,1,0] = 0.3
short_score[1,2,1,1] = 0.4
short_score[2,1,0,0] = 0.7
short_score[2,1,0,1] = 0.6
short_score[2,1,1,0] = 0.8
short_score[2,1,1,1] = 0.7
el = EL.eisner_layer(3,2,1)
short_score = short_score.reshape(1,3,3,2,2)
short_score = torch.DoubleTensor(short_score)
inside_table,sentence_score = el.batch_inside(short_score)
print sentence_score
weights = np.zeros((3,2,3,2))
weights[0,0,1,0] = 0.3
weights[0,0,1,1] = 0.2
weights[0,0,2,0] = 0.6
weights[0,0,2,1] = 0.7
weights[1,0,2,0] = 0.4
weights[1,0,2,1] = 0.5
weights[1,1,2,0] = 0.3
weights[1,1,2,1] = 0.4
weights[2,0,1,0] = 0.7
weights[2,0,1,1] = 0.6
weights[2,1,1,0] = 0.8
weights[2,1,1,1] = 0.7
weights = torch.DoubleTensor(weights)
weights = weights.unsqueeze(0)
inside_table,ll = dp_inside_batch(1,3,2,weights)
print ll






