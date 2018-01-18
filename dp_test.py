import eisner_parser
import numpy as np
from torch.nn.init import *
scores = np.zeros((4,4,2,2))
scores[0,1,0,0] = 0.3
scores[0,1,0,1] = 0.2
scores[0,2,0,0] = 0.6
scores[0,2,0,1] = 0.7
scores[0,3,0,0] = 0.4
scores[0,3,0,1] = 0.3
scores[1,2,0,0] = 0.4
scores[1,2,0,1] = 0.5
scores[1,3,0,0] = 0.6
scores[1,3,0,1] = 0.5
scores[1,2,1,0] = 0.3
scores[1,2,1,1] = 0.4
scores[1,3,1,0] = 0.5
scores[1,3,1,1] = 0.4
scores[2,1,0,0] = 0.7
scores[2,1,0,1] = 0.6
scores[2,3,0,0] = 0.7
scores[2,3,0,1] = 0.6
scores[2,1,1,0] = 0.8
scores[2,1,1,1] = 0.7
scores[2,3,1,0] = 0.8
scores[2,3,1,1] = 0.7
scores[3,1,0,0] = 0.6
scores[3,1,0,1] = 0.5
scores[3,2,0,0] = 0.4
scores[3,2,0,1] = 0.5
scores[3,1,1,0] = 0.5
scores[3,1,1,1] = 0.4
scores[3,2,1,0] = 0.3
scores[3,2,1,1] = 0.4
best_parse = eisner_parser.parse_proj(scores)
sentence_score,inside_scores = eisner_parser.partition_inside(Variable(torch.FloatTensor(scores)))
outside_score = eisner_parser.partition_outside(inside_scores,scores)