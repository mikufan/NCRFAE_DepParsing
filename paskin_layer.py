import utils
import numpy
import torch
import torch.autograd as autograd
import itertools
import numpy as np
from numba import njit

LOGZERO = -numpy.inf


class PASKIN(torch.autograd.Function):
    """ PLDFM
    args:
        hidden_dim : input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
    """

    def __init__(self, sentence_len, tags_dim, batch_size, is_multi_root=True):
        super(PASKIN, self).__init__()
        self.sentence_len, self.tags_dim = sentence_len, tags_dim
        self.is_multi_root = is_multi_root
        self.batch_size = batch_size

    def forward(self, weights, mask=None):
        """
        args:
            weights (batch_size, sentence_len, tag_dim, sentence_len, tag_dim) : input score from previous layers
        return:
            output from pldfm layer (batch_size, 1)
        """
        self.inside_table, self.score = self.dp_inside_batch(weights)
        self.weights = weights
        self.mask = mask
        return self.score

    def diff(self):
        outside_table = self.dp_outside_batch(self.inside_table, self.weights)

        counts = self.inside_table + outside_table# shape is batch_size * len_ijspan * tags_dim * tags_dim
        part_count = torch.DoubleTensor(self.batch_size, self.sentence_len, self.tags_dim, self.sentence_len, self.tags_dim)
        part_count.fill_(0.0)
        seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = utils.constituent_indexes(
            self.sentence_len, self.is_multi_root)

        for left_index in range(self.sentence_len):
            for right_index in range(self.sentence_len):
                span_id1 = span_id_map.get((left_index, right_index, utils.get_state_code(0, 1, 1)))
                if span_id1 is not None:
                    part_count[:, left_index, :, right_index, :] = counts[:, span_id1, :, :]

        for left_index in range(self.sentence_len):
            for right_index in range(self.sentence_len):
                span_id2 = span_id_map.get((left_index, right_index, utils.get_state_code(1, 0, 1)))
                if span_id2 is not None:
                    swap_count = counts[:, span_id2, :, :].permute(0, 2, 1)
                    part_count[:, right_index, :, left_index, :] = swap_count

        if torch.cuda.is_available():
            part_count = part_count.cuda()
        alpha = part_count - self.score.contiguous().view(self.batch_size, 1, 1, 1, 1)
        diff = torch.exp(alpha)
        if self.mask is not None:
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
            diff = diff.masked_fill_(self.mask, 0.0)
        return diff

    def backward(self, grad_output):
        alpha = self.diff()
        grad_output = grad_output.contiguous().view(self.batch_size, 1, 1, 1, 1)
        gradient = alpha * grad_output
        batch_size, sent_len, tag_dim, _, _ = gradient.size()

        #gradient[:, 0, 1:, :, :].fill_(0.0)
        #gradient[:, :, :, 0, 1:].fill_(0.0)
        for i in range(self.sentence_len):
            gradient[:, i, :, i, :].fill_(0.0)
        return gradient, None

    def dp_inside_batch(self, weights):
        inside_table = torch.DoubleTensor(self.batch_size, self.sentence_len * self.sentence_len * 8, self.tags_dim, self.tags_dim)
        inside_table.fill_(LOGZERO)
        if torch.cuda.is_available():
            inside_table = inside_table.cuda()
        m = self.sentence_len
        seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = utils.constituent_indexes(
            m, self.is_multi_root)

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
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                prob = inside_table[:, ids, :, :] + weights[:, l, :, r, :]
                inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], prob)
            elif ij in right_spans:
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                swap_weights = weights.permute(0, 1, 4, 3, 2)
                prob = inside_table[:, ids, :, :] + swap_weights[:, r, :, l, :]
                inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], prob)
            else:
                num_k = len(ikss[ij])
                beta_ik, beta_kj = inside_table[:, ikss[ij], :, :], inside_table[:, kjss[ij], :, :]
                probs = beta_ik.contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim, 1) +\
                        beta_kj.contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim)
                probs = utils.logsumexp(probs, axis=(1, 3))
                inside_table[:, ij, :, :] = utils.logaddexp(inside_table[:, ij, :, :], probs)

        id1 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 0)), -1)
        id2 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 1)), -1)

        score1 = inside_table[:, id1, 0, :].contiguous().view(self.batch_size, 1, self.tags_dim)
        score2 = inside_table[:, id2, 0, :].contiguous().view(self.batch_size, 1, self.tags_dim)
        ll = utils.logaddexp(utils.logsumexp(score1, axis=2), utils.logsumexp(score2, axis=2))
        return inside_table, ll

    def dp_outside_batch(self, inside_table, weights):
        outside_table = torch.DoubleTensor(self.batch_size, self.sentence_len * self.sentence_len * 8, self.tags_dim, self.tags_dim)
        outside_table.fill_(LOGZERO)
        if torch.cuda.is_available():
            outside_table = outside_table.cuda()
        m = self.sentence_len
        seed_spans, base_left_spans, base_right_spans, left_spans, right_spans, ijss, ikss, kjss, id_span_map, span_id_map = utils.constituent_indexes(
            m, self.is_multi_root)
        id1 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 0)), -1)
        id2 = span_id_map.get((0, m - 1, utils.get_state_code(0, 1, 1)), -1)
        outside_table[:, id1, :, :] = 0.0
        outside_table[:, id2, :, :] = 0.0

        for ij in reversed(ijss):
            (l, r, c) = id_span_map[ij]
            if ij in left_spans:
                assert c == utils.get_state_code(0, 1, 1)
                prob = outside_table[:, ij, :, :] + weights[:, l, :, r, :]
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                outside_table[:, ids, :, :] = utils.logaddexp(outside_table[:, ids, :, :], prob)
            elif ij in right_spans:
                assert c == utils.get_state_code(1, 0, 1)
                swap_weights = weights.permute(0, 1, 4, 3, 2)
                prob = outside_table[:, ij, :, :] + swap_weights[:, r, :, l, :]
                ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 0)), -1)
                outside_table[:, ids, :, :] = utils.logaddexp(outside_table[:, ids, :, :], prob)
            else:
                num_k = len(ikss[ij])
                if l == 0:
                    # ROOT's value can only be 0
                    alpha_ij = outside_table[:, ij, 0, :].contiguous().view(self.batch_size, 1, 1, 1, self.tags_dim)
                    beta_left = inside_table[:, ikss[ij], [0], :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, 1)
                    beta_right = inside_table[:, kjss[ij], :, :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim)
                    new_left = alpha_ij + beta_right
                    new_left = utils.logsumexp(new_left, axis=4).contiguous().view(self.batch_size, num_k, 1, self.tags_dim, 1)
                    new_right = alpha_ij + beta_left
                    if len(list(set(ikss[ij]))) == num_k:
                        outside_table[:, ikss[ij], [0], :] = utils.logaddexp(
                            outside_table[:, ikss[ij], [0], :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, 1), new_left).contiguous().view(self.batch_size, num_k, self.tags_dim)
                    outside_table[:, kjss[ij], :, :] = utils.logaddexp(
                            outside_table[:, kjss[ij], :, :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim),
                            new_right).contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim)
                else:
                    alpha_ij = outside_table[:, ij, :, :].contiguous().view(self.batch_size, 1, self.tags_dim, 1, self.tags_dim)
                    beta_left = inside_table[:, ikss[ij], :, :].contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim, 1)
                    beta_right = inside_table[:, kjss[ij], :, :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim)
                    new_left = alpha_ij + beta_right
                    new_left = utils.logsumexp(new_left, axis=4).contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim, 1)
                    new_right = alpha_ij + beta_left
                    new_right = utils.logsumexp(new_right, axis=2).contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim)
                    if len(list(set(ikss[ij]))) == num_k:
                        outside_table[:, ikss[ij], :, :] = utils.logaddexp(
                            outside_table[:, ikss[ij], :, :].contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim, 1),
                            new_left).contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim)

                    outside_table[:, kjss[ij], :, :] = utils.logaddexp(outside_table[:, kjss[ij], :, :].contiguous().view(self.batch_size, num_k, 1, self.tags_dim, self.tags_dim), new_right).contiguous().view(self.batch_size, num_k, self.tags_dim, self.tags_dim)

                if len(list(set(ikss[ij]))) == num_k:
                    # Already done in the above
                    pass
                else:
                    # TODO make this vectorized, the problem is the id in ikss is not unique
                    for i in range(num_k):
                        ik = ikss[ij][i]
                        kj = kjss[ij][i]
                        if l == 0:
                            alpha_ij = outside_table[:, ij, 0, :].contiguous().view(self.batch_size, 1, 1, 1, self.tags_dim)
                            beta_right = inside_table[:, kj, :, :].contiguous().view(self.batch_size, 1, 1, self.tags_dim, self.tags_dim)
                            new_left = alpha_ij + beta_right
                            new_left = utils.logsumexp(new_left, axis=4).contiguous().view(self.batch_size, 1, 1, self.tags_dim, 1)
                            outside_table[:, ik, 0, :] = utils.logaddexp(
                                outside_table[:, ik, 0, :].contiguous().view(self.batch_size, 1, 1, self.tags_dim, 1), new_left).contiguous().view(self.batch_size, self.tags_dim, )
                        else:
                            alpha_ij = outside_table[:, ij, :, :].contiguous().view(self.batch_size, 1, self.tags_dim, 1, self.tags_dim)
                            beta_right = inside_table[:, kj, :, :].contiguous().view(self.batch_size, 1, 1, self.tags_dim, self.tags_dim)
                            new_left = alpha_ij + beta_right
                            new_left = utils.logsumexp(new_left, axis=4).contiguous().view(self.batch_size, 1, self.tags_dim, self.tags_dim, 1)
                            outside_table[:, ik, :, :] = utils.logaddexp(outside_table[:, ik, :, :].contiguous().view(self.batch_size, 1, self.tags_dim, self.tags_dim, 1), new_left).contiguous().view(self.batch_size, self.tags_dim, self.tags_dim)

        for ij in base_left_spans:
            (l, r, c) = id_span_map[ij]
            prob = outside_table[:, ij, :, :] + weights[:, l, :, r, :]
            ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 1)), -1)
            outside_table[:, ids, :, :] = utils.logaddexp(outside_table[:, ids, :, :], prob)

        for ij in base_right_spans:
            (l, r, c) = id_span_map[ij]
            swap_weights = weights.permute(0, 1, 4, 3, 2)
            prob = outside_table[:, ij, :, :] + swap_weights[:, r, :, l, :]
            ids = span_id_map.get((l, r, utils.get_state_code(0, 0, 1)), -1)
            outside_table[:, ids, :, :] = utils.logaddexp(outside_table[:, ids, :, :], prob)

        return outside_table

    def decoding(self, weights):
        best_score, decoded_tags = utils.decoding_batch(weights.data.cpu().numpy(), self.is_multi_root)
        return decoded_tags


class PLDFMRepack:
    """Packer for word level model

    args:
        tagset_size: target_set_size
        if_cuda: whether use GPU
    """

    def __init__(self, tagset_size, if_cuda):

        self.tagset_size = tagset_size
        self.if_cuda = if_cuda


    def repack_vb(self, feature, target):
        """packer for viterbi loss

        args:
            feature (Seq_len, Batch_size): input feature
            target (Seq_len, Batch_size): output target
        return:
            feature (Seq_len, Batch_size), target (Seq_len, Batch_size)
        """
        feature = torch.LongTensor(feature)
        target = torch.LongTensor(target)

        if self.if_cuda:
            fea_v = autograd.Variable(feature.transpose(0, 1)).cuda()
            tg_v = autograd.Variable(target.transpose(0, 1)).unsqueeze(2).cuda()
        else:
            fea_v = autograd.Variable(feature.transpose(0, 1))
            tg_v = autograd.Variable(target.transpose(0, 1)).contiguous().unsqueeze(2)
        return fea_v, tg_v


    def convert_for_eval(self, target):
        """convert target to original decoding

        args:
            target: input labels used in training
        return:
            output labels used in test
        """
        return target % self.tagset_size