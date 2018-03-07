import torch
import torch.autograd as autograd
from optparse import OptionParser
import utils
import dt_pl_model
from tqdm import tqdm
import sys
import random
import param
import numpy as np
import os
import pickle

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train", help="train file", metavar="FILE", default="data/toy_test")
    parser.add_option("--dev", dest="dev", help="dev file", metavar="FILE", default="data/wsj10_d")

    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--batch", type="int", dest="batchsize", default=100)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dim", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dim", default=25)
    parser.add_option("--hidden", type="int", dest="hidden_dim", default=100)
    parser.add_option("--nLayer", type="int", dest="n_layer", default=1)
    parser.add_option("--epochs", type="int", dest="epochs", default=10)
    parser.add_option("--tag_num", type="int", dest="tag_num", default=4)
    parser.add_option("--tag_dim", type="int", dest="tag_dim", default=5)
    parser.add_option("--use_dir", action="store_true", dest="dir_flag", default=False)
    parser.add_option("--dist_num", type="int", dest="dist_num", default=5)

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.01)
    parser.add_option("--outdir", type="string", dest="output", default="output")
    parser.add_option("--l2",type="float",dest="l2",default=0.0)
    parser.add_option("--sample_idx",type="int",dest="sample_idx",default=1000)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--distdim", type="int", dest="dist_dim", default=1)
    parser.add_option("--dropout", type="float", dest="dropout_ratio", default=0.25)
    parser.add_option("--trans_hidden", type="int", dest="trans_hidden", default=50)
    parser.add_option("--use_lex", action="store_true", dest="use_lex", default=False)
    parser.add_option("--use_context", action="store_true", dest="use_context", default=False)
    parser.add_option("--prior_weight", type="float", dest="prior_weight", default=0.0)
    parser.add_option("--rule_type",type="string",dest="rule_type",default="WSJ")
    parser.add_option("--use_gold",action="store_true",dest="use_gold",default=False)
    parser.add_option("--use_initial",action="store_true",dest="use_initial",default=False)
    parser.add_option("--do_eval",action="store_true",dest="do_eval",default= False)
    parser.add_option("--log",dest="log",help="log file", metavar="FILE", default="output/log")

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)

    parser.add_option("--e_pass", type="int", dest="e_pass", default=2)
    parser.add_option("--d_pass", type="int", dest="d_pass", default=2)

    parser.add_option("--paramdec", dest="paramdec", help="Decoder parameters file", metavar="FILE",
                      default="paramdec.pickle")

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)

    w2i, pos, sentences = utils.read_data(options.train, False)
    print 'Data read'
    with open(os.path.join(options.output, options.params+'_'+str(options.sample_idx)), 'w') as paramsfp:
        pickle.dump((w2i, pos, options), paramsfp)
    print 'Parameters saved'
    data_list = list()
    sen_idx = 0
    if options.prior_weight > 0:
        prior_dict = {}
        prior_set = param.set_prior(options.rule_type)
    if options.use_gold:
        gold_dict = {}
    for s in sentences:
        s_word, s_pos = s.set_data_list(w2i, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append([sen_idx])
        data_list.append(s_data_list)
        if options.prior_weight > 0:
            s_prior = utils.construct_prior(prior_set,s,pos,options.tag_num,options.prior_weight)
            prior_dict[sen_idx] = s_prior
        if options.use_gold:
            s_gold = list(map(lambda e:e.parent_id,s.entries))
            gold_dict[sen_idx] = s_gold
        sen_idx += 1
    # utils.find_trans(sentences,pos,options.tag_num)
    batch_data = utils.construct_batch_data(data_list, options.batchsize)
    print 'Batch data constructed'

    dependencyTaggingPl_model = dt_pl_model.dt_paralell_model(w2i, pos, options)
    if options.prior_weight > 0:
        dependencyTaggingPl_model.prior_dict = prior_dict
    if options.use_gold:
        dependencyTaggingPl_model.gold_dict = gold_dict
    print 'Model constructed'
    dependencyTaggingPl_model.init_decoder_param(sentences)
    print 'Decoder parameters initialized'
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        dependencyTaggingPl_model.cuda(options.gpu)

    for epoch in range(options.epochs):
        print 'Starting epoch', epoch
        print 'To train encoder'
        dependencyTaggingPl_model.train()
        if epoch == 0 and options.use_initial:
            dependencyTaggingPl_model.initial_Flag = True
        for n in range(options.e_pass):
            iter_loss = 0.0
            training_likelihood = 0.0
            print 'Encoder training iteration ', n
            tot_batch = len(batch_data)
            #random.shuffle(batch_data)
            for batch_id, one_batch in tqdm(
                    enumerate(batch_data), mininterval=2,
                    desc=' -Tot it %d (epoch %d)' % (tot_batch, 0), leave=False, file=sys.stdout):
                batch_words, batch_pos, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                    [s[2][0] for s in one_batch]
                batch_words_v = utils.list2Variable(batch_words, options.gpu)
                batch_pos_v = utils.list2Variable(batch_pos, options.gpu)

                batch_loss, batch_likelihood = dependencyTaggingPl_model(batch_words_v, batch_pos_v, None,
                                                                         batch_sen)
                training_likelihood += batch_likelihood
                batch_loss.backward()
                dependencyTaggingPl_model.trainer.step()
                dependencyTaggingPl_model.trainer.zero_grad()
                iter_loss += param.get_scalar(batch_loss.cpu(), 0)
            iter_loss /= tot_batch
            print ' loss for this iteration ', iter_loss

            print 'likelihood for this iteration ', training_likelihood

            if dependencyTaggingPl_model.initial_Flag:
                dependencyTaggingPl_model.initial_Flag = False
        print 'To train decoder'
        if options.dir_flag:
            dir_dim = 2
        else:
            dir_dim = 1
        for n in range(options.d_pass):
            print 'Decoder training iteration ', n
            training_likelihood = 0.0
            recons_counter = np.zeros(
                (len(pos.keys()), options.tag_num, len(pos.keys()), options.dist_dim, dir_dim))
            if options.use_lex:
                lex_counter = np.zeros((len(pos.keys()), options.tag_num, len(w2i.keys())))
            else:
                lex_counter = None
            for batch_id in range(len(batch_data)):
                one_batch = batch_data[batch_id]
                batch_words, batch_pos, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                    [s[2][0] for s in one_batch]
                batch_likelihood = dependencyTaggingPl_model.hard_em_e(batch_pos, batch_words, batch_sen,
                                                                       recons_counter, lex_counter)
                training_likelihood += batch_likelihood
            print 'Likelihood for this iteration', training_likelihood
            dependencyTaggingPl_model.hard_em_m(batch_data, recons_counter, lex_counter)
        with open(os.path.join(options.output, options.paramdec) + str(epoch + 1)+'_'+str(options.sample_idx), 'w') as paramdec:
            pickle.dump((dependencyTaggingPl_model.recons_param, dependencyTaggingPl_model.lex_param), paramdec)
        dependencyTaggingPl_model.save(
            os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)+'_'+str(options.sample_idx)))
        if options.do_eval:
            print "===================================="
            print 'Do evaluation on development set'
            eval_sentences = utils.read_data(options.dev, True)
            dependencyTaggingPl_model.eval()
            sen_idx = 0
            eval_data_list = list()
            devpath = os.path.join(options.output, 'test_pred'+str(epoch+1)+'_'+str(options.sample_idx))
            for s in eval_sentences:
                s_word, s_pos = s.set_data_list(w2i, pos)
                s_data_list = list()
                s_data_list.append(s_word)
                s_data_list.append(s_pos)
                s_data_list.append([sen_idx])
                eval_data_list.append(s_data_list)
                sen_idx += 1
            eval_batch_data = utils.construct_batch_data(eval_data_list, options.batchsize)
            for batch_id, one_batch in enumerate(eval_batch_data):
                eval_batch_words, eval_batch_pos, eval_batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                                    [s[2][0] for s in one_batch]
                eval_batch_words_v = utils.list2Variable(eval_batch_words, options.gpu)
                eval_batch_pos_v = utils.list2Variable(eval_batch_pos, options.gpu)
                dependencyTaggingPl_model(eval_batch_words_v, eval_batch_pos_v, None, eval_batch_sen)
            test_res = dependencyTaggingPl_model.parse_results
            utils.eval(test_res, eval_sentences, devpath,options.log+'_'+str(options.sample_idx),epoch)
            print "===================================="
    print 'Training finished'
