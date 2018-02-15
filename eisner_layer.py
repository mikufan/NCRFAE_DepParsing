import torch.autograd as autograd
import torch
from utils import memoize
import utils
import numpy as np

LOGZERO = -np.inf


class eisner_layer(autograd.Function):
    def __init__(self, sentence_length, tag_num, batch_size):
        super(eisner_layer, self).__init__()
        self.sentence_length = sentence_length
        self.tag_num = tag_num
        self.batch_size = batch_size

    def forward(self, crf_scores):

        self.inside_table, self.partition_score = self.batch_inside(crf_scores)
        self.crf_scores = crf_scores
        return self.partition_score

    def backward(self, output):
        mius = self.diff()
        grad_output = output.contiguous().view(self.batch_size, 1, 1, 1, 1)
        gradient = mius * grad_output
        batch_size, sent_len, _, tag_dim, _ = gradient.size()

        gradient[:, 0, :, 1:, :].fill_(0.0)
        gradient[:, :, 0, :, :].fill_(0.0)
        for i in range(self.sentence_length):
            gradient[:, i, i, :, :].fill_(0.0)
        return gradient

    def diff(self):
        self.outside_table = self.batch_outside(self.inside_table, self.crf_scores)

        counts = self.inside_table[1] + self.outside_table[1]
        pseudo_count = torch.DoubleTensor(self.batch_size, self.sentence_length, self.sentence_length, self.tag_num,
                                         self.tag_num)
        pseudo_count.fill_(LOGZERO)
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(self.sentence_length)

        for l in range(self.sentence_length):
            for r in range(self.sentence_length):
                for dir in range(2):
                    span_id = span_2_id.get((l, r, dir))
                    if span_id is not None:
                        if dir == 0:
                            pseudo_count[:, r, l, :, :] = counts[:, span_id, :, :].permute(0,2,1)
                        else:
                            pseudo_count[:, l, r, :, :] = counts[:, span_id, :, :]
        if torch.cuda.is_available():
            pseudo_count = pseudo_count.cuda()
        mius = pseudo_count - self.partition_score.contiguous().view(self.batch_size, 1, 1, 1, 1)
        diff = torch.exp(mius)
        # if self.mask is not None:
        #     diff = diff.masked_fill_(self.mask, 0.0)
        return diff

    def batch_inside(self, crf_scores):
        inside_complete_table = torch.DoubleTensor(self.batch_size, self.sentence_length * self.sentence_length * 2,
                                                  self.tag_num)
        inside_incomplete_table = torch.DoubleTensor(self.batch_size, self.sentence_length * self.sentence_length * 2,
                                                    self.tag_num, self.tag_num)
        if torch.cuda.is_available():
            inside_complete_table = inside_complete_table.cuda()
            inside_incomplete_table = inside_incomplete_table.cuda()
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(self.sentence_length)

        inside_complete_table.fill_(LOGZERO)
        inside_incomplete_table.fill_(LOGZERO)

        for ii in basic_span:
            inside_complete_table[:, ii, :] = 0.0

        for ij in ijss:
            (l, r, dir) = id_2_span[ij]
            # two complete span to form an incomplete span
            num_ki = len(ikis[ij])
            inside_ik_ci = inside_complete_table[:, ikis[ij], :].contiguous().view(self.batch_size, num_ki,
                                                                                   self.tag_num, 1)
            inside_kj_ci = inside_complete_table[:, kjis[ij], :].contiguous().view(self.batch_size, num_ki, 1,
                                                                                   self.tag_num)
            if dir == 0:
                span_inside_i = inside_ik_ci + inside_kj_ci + crf_scores[:, r, l, :, :] \
                    .permute(0, 2, 1).contiguous().view(self.batch_size, 1, self.tag_num, self.tag_num)
                # swap head-child to left-right position
            else:
                span_inside_i = inside_ik_ci + inside_kj_ci + crf_scores[:, l, r, :, :].contiguous().view(
                    self.batch_size, 1, self.tag_num, self.tag_num)
            inside_incomplete_table[:, ij, :, :] = utils.logsumexp(span_inside_i, axis=1)

            # one complete span and one incomplete span to form bigger complete span
            num_kc = len(ikcs[ij])
            if dir == 0:
                inside_ik_cc = inside_complete_table[:, ikcs[ij], :].contiguous().view(self.batch_size, num_kc,
                                                                                       self.tag_num, 1)
                inside_kj_ic = inside_incomplete_table[:, kjcs[ij], :, :].contiguous().view(self.batch_size, num_kc,
                                                                                            self.tag_num, self.tag_num)
                span_inside_c = inside_ik_cc + inside_kj_ic
                span_inside_c = span_inside_c.contiguous().view(self.batch_size, num_kc * self.tag_num, self.tag_num)
                inside_complete_table[:, ij, :] = utils.logsumexp(span_inside_c, axis=1)
            else:
                inside_ik_ic = inside_incomplete_table[:, ikcs[ij], :, :].contiguous().view(self.batch_size, num_kc,
                                                                                            self.tag_num, self.tag_num)
                inside_kj_cc = inside_complete_table[:, kjcs[ij], :].contiguous().view(self.batch_size, num_kc,
                                                                                       1, self.tag_num)
                span_inside_c = inside_ik_ic + inside_kj_cc
                span_inside_c = span_inside_c.permute(0, 1, 3, 2).contiguous().view(self.batch_size,
                                                                                    num_kc * self.tag_num, self.tag_num)
                # swap the left-right position since the left tags are to be indexed
                inside_complete_table[:, ij, :] = utils.logsumexp(span_inside_c, axis=1)

        final_id = span_2_id[(0, self.sentence_length - 1, 1)]
        partition_score = inside_complete_table[:, final_id, 0]

        return (inside_complete_table, inside_incomplete_table), partition_score

    def batch_outside(self, inside_table, crf_score):
        inside_complete_table = inside_table[0]
        inside_incomplete_table = inside_table[1]
        outside_complete_table = torch.DoubleTensor(self.batch_size, self.sentence_length * self.sentence_length * 2,
                                                   self.tag_num)
        outside_incomplete_table = torch.DoubleTensor(self.batch_size, self.sentence_length * self.sentence_length * 2,
                                                     self.tag_num, self.tag_num)
        if torch.cuda.is_available():
            outside_complete_table = outside_complete_table.cuda()
            outside_incomplete_table = outside_incomplete_table.cuda()
        span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span = utils.constituent_index(self.sentence_length)
        outside_complete_table.fill_(LOGZERO)
        outside_incomplete_table.fill_(LOGZERO)

        root_id = span_2_id.get((0, self.sentence_length - 1, 1))
        outside_complete_table[:, root_id, 0] = 0.0

        complete_span_used = set()
        incomplete_span_used = set()
        complete_span_used.add(root_id)
        root_flag = False
        for ij in reversed(ijss):
            (l, r, dir) = id_2_span[ij]
            # complete span consists of one incomplete span and one complete span
            num_kc = len(ikcs[ij])
            if dir == 0:
                outside_ij_cc = outside_complete_table[:, ij, :].contiguous().view(self.batch_size, 1, 1, self.tag_num)
                inside_kj_ic = inside_incomplete_table[:, kjcs[ij], :, :].contiguous().view(self.batch_size, num_kc,
                                                                                            self.tag_num, self.tag_num)
                inside_ik_cc = inside_complete_table[:, ikcs[ij], :].contiguous().view(self.batch_size, num_kc,
                                                                                       self.tag_num, 1)
                outside_ik_cc = (outside_ij_cc + inside_kj_ic).permute(0, 1, 3, 2)
                # swap left-right position since right tags are to be indexed
                outside_kj_ic = outside_ij_cc + inside_ik_cc
                for i in range(num_kc):
                    ik = ikcs[ij][i]
                    kj = kjcs[ij][i]
                    outside_ik_cc_i = utils.logsumexp(outside_ik_cc[:, i, :, :], axis=1)
                    if ik in complete_span_used:
                        outside_complete_table[:, ik, :] = utils.logaddexp(
                            outside_complete_table[:, ik, :], outside_ik_cc_i)
                    else:
                        outside_complete_table[:, ik, :] = outside_ik_cc_i.clone()
                        complete_span_used.add(ik)

                    if kj in incomplete_span_used:
                        outside_incomplete_table[:, kj, :, :] = utils.logaddexp(outside_incomplete_table[:, kj, :, :],
                                                                                outside_kj_ic[:, i, :, :])
                    else:
                        outside_incomplete_table[:, kj, :, :] = outside_kj_ic[:, i, :, :]
                        incomplete_span_used.add(kj)
            else:
                outside_ij_cc = outside_complete_table[:, ij, :].contiguous().view(self.batch_size, 1, self.tag_num, 1)
                inside_ik_ic = inside_incomplete_table[:, ikcs[ij], :, :].contiguous().view(self.batch_size, num_kc,
                                                                                            self.tag_num, self.tag_num)
                inside_kj_cc = inside_complete_table[:, kjcs[ij], :].contiguous().view(self.batch_size, num_kc, 1,
                                                                                       self.tag_num)
                outside_kj_cc = outside_ij_cc + inside_ik_ic
                outside_ik_ic = outside_ij_cc + inside_kj_cc
                for i in range(num_kc):
                    kj = kjcs[ij][i]
                    ik = ikcs[ij][i]
                    outside_kj_cc_i = utils.logsumexp(outside_kj_cc[:, i, :, :], axis=1)
                    if kj in complete_span_used:
                        outside_complete_table[:, kj, :] = utils.logaddexp(outside_complete_table[:, kj, :],
                                                                           outside_kj_cc_i)
                    else:
                        outside_complete_table[:, kj, :] = outside_kj_cc_i.clone()
                        complete_span_used.add(kj)

                    if ik in incomplete_span_used:
                        outside_incomplete_table[:, ik, :, :] = utils.logaddexp(outside_incomplete_table[:, ik, :, :],
                                                                                outside_ik_ic[:, i, :, :])
                    else:
                        outside_incomplete_table[:, ik, :, :] = outside_ik_ic[:, i, :, :]
                        incomplete_span_used.add(ik)

            # incomplete span consists of two complete spans
            num_ki = len(ikis[ij])

            outside_ij_ii = outside_incomplete_table[:, ij, :, :].contiguous().view(self.batch_size, 1, self.tag_num,
                                                                                    self.tag_num)
            inside_ik_ci = inside_complete_table[:, ikis[ij], :].contiguous().view(self.batch_size, num_ki,
                                                                                   self.tag_num, 1)
            inside_kj_ci = inside_complete_table[:, kjis[ij], :].contiguous().view(self.batch_size, num_ki, 1,
                                                                                   self.tag_num)

            if dir == 0:
                outside_ik_ci = outside_ij_ii + inside_kj_ci + crf_score[:, r, l, :, :]. \
                    permute(0, 2, 1).contiguous().view(self.batch_size, 1, self.tag_num, self.tag_num)

                outside_kj_ci = outside_ij_ii + inside_ik_ci + crf_score[:, r, l, :, :]. \
                    permute(0, 2, 1).contiguous().view(self.batch_size, 1, self.tag_num, self.tag_num)
            else:
                outside_ik_ci = outside_ij_ii + inside_kj_ci + crf_score[:, l, r, :, :].contiguous().view(
                    self.batch_size, 1, self.tag_num, self.tag_num)
                outside_kj_ci = outside_ij_ii + inside_ik_ci + crf_score[:, l, r, :, :].contiguous().view(
                    self.batch_size, 1, self.tag_num, self.tag_num)

            for i in range(num_ki):
                ik = ikis[ij][i]
                kj = kjis[ij][i]

                outside_ik_ci_i = utils.logsumexp(outside_ik_ci[:, i, :, :], axis=2)
                outside_kj_ci_i = utils.logsumexp(outside_kj_ci[:, i, :, :], axis=1)
                if ik in complete_span_used:
                    outside_complete_table[:, ik, :] = utils.logaddexp(outside_complete_table[:, ik, :],
                                                                       outside_ik_ci_i)
                else:
                    outside_complete_table[:, ik, :] = outside_ik_ci_i.clone()
                    complete_span_used.add(ik)
                if kj in complete_span_used:
                    outside_complete_table[:, kj, :] = utils.logaddexp(outside_complete_table[:, kj, :],
                                                                       outside_kj_ci_i)
                else:
                    outside_complete_table[:, kj, :] = outside_kj_ci_i.clone()
                    complete_span_used.add(kj)

        return (outside_complete_table, outside_incomplete_table)
