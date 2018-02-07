import eisner_parser
import numpy as np
import eisner_layer
from torch.nn.init import *
import torch.autograd

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
scores = scores.reshape((1,4,4,2,2))

scores = np.tile(scores,(2,1,1,1,1))

#best_parse = eisner_parser.parse_proj(scores)
best_parse = eisner_parser.batch_parse(scores)
batch_size,sentence_length,_,tag_num,_ = scores.shape
el = eisner_layer.eisner_layer(sentence_length,tag_num,batch_size)
scores = torch.FloatTensor(scores)
inside_table,sentence_score = el.batch_inside(scores)
print sentence_score
outside = el.batch_outside(inside_table,sentence_score)
