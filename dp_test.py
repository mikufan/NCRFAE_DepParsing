import eisner_parser
import numpy as np
import eisner_layer as EL
from torch.nn.init import *
import torch.autograd
import utils
import dt_pl_model
import pickle
import torch.autograd as autograd


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

def check_gradient(delta, inputs, sentence_len, batch_size=1):
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
                            exit(-1)
    print("PASS")


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
batch_size,sentence_length,_,_,_ = new_score.shape
mask = construct_mask(batch_size,sentence_length,2)
new_score = autograd.Variable(new_score)
#new_score = new_score.masked_fill(mask, -np.inf)
check_gradient(1e-6,new_score,sentence_length,batch_size)





