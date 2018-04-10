from optparse import OptionParser
import os
import pickle
import utils
import dt_pl_model
import torch
import param
import numpy as np

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--test", dest="test", help="test file", metavar="FILE", default="data/wsj10_te")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--batch", type="int", dest="batchsize", default=100)

    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/neuralfirstorder.model")

    parser.add_option("--outdir", type="string", dest="output", default="output")

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)

    parser.add_option("--paramdec", dest="paramdec", help="Decoder parameters file", metavar="FILE",
                      default="paramdec.pickle")

    parser.add_option("--log", dest="log", help="log file", metavar="FILE", default="output/log")

    parser.add_option("--idx", type="int", dest="model_idx", default=1)

    parser.add_option("--sample_idx", type="int", dest="sample_idx", default=1000)
    parser.add_option("--use_trigram", action="store_true", dest="use_trigram", default=False)

    parser.add_option("--gpu", type="int", dest="gpu", default=-1, help='gpu id, set to -1 if use cpu mode')

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        print 'To use gpu' + str(options.gpu)

    with open(os.path.join(options.output, options.params+'_'+str(options.sample_idx)), 'r') as paramsfp:
        w2i, pos, stored_opt = pickle.load(paramsfp)
        # stored_opt.external_embedding = options.external_embedding
    sentences = utils.read_data(options.test, True)
    dependencyTaggingPl_model = dt_pl_model.dt_paralell_model(w2i, pos, stored_opt)
    dependencyTaggingPl_model.load(options.model + str(options.model_idx)+'_'+str(options.sample_idx))
    if not stored_opt.use_gold:
        with open(
                os.path.join(options.output, options.paramdec + str(options.model_idx) + '_' + str(options.sample_idx)),
                'r') as paramdec:
            dependencyTaggingPl_model.recons_param, dependencyTaggingPl_model.lex_param = pickle.load(paramdec)
    else:
        dependencyTaggingPl_model.use_gold = False
    print 'Model loaded'
    dependencyTaggingPl_model.eval()
    testpath = os.path.join(options.output, 'test_pred' + '_' + str(options.sample_idx))
    sen_idx = 0
    data_list = list()
    for s in sentences:
        s_word, s_pos = s.set_data_list(w2i, pos)
        s_data_list = list()
        s_data_list.append(s_word)
        s_data_list.append(s_pos)
        s_data_list.append([sen_idx])
        if options.use_trigram:
            s_trigram = utils.construct_trigram(s_pos,pos)
            s_data_list.append(s_trigram)
        data_list.append(s_data_list)
        sen_idx += 1
    batch_data = utils.construct_batch_data(data_list, options.batchsize)
    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(options.gpu)
        dependencyTaggingPl_model.cuda(options.gpu)
    dependencyTaggingPl_model.prior_weight = 0
    for batch_id, one_batch in enumerate(batch_data):
        batch_words, batch_pos, batch_sen = [s[0] for s in one_batch], [s[1] for s in one_batch], \
                                            [s[2][0] for s in one_batch]
        batch_words_v = utils.list2Variable(batch_words, options.gpu)
        batch_pos_v = utils.list2Variable(batch_pos, options.gpu)
        if options.use_trigram:
            batch_trigram = [s[3] for s in one_batch]
            batch_trigram_v = utils.list2Variable(batch_trigram, options.gpu)
        else:
            batch_trigram_v = None
        dependencyTaggingPl_model(batch_words_v, batch_pos_v, None, batch_sen,batch_trigram_v)
    test_res = dependencyTaggingPl_model.parse_results
    utils.eval(test_res, sentences, testpath,options.log,0)
