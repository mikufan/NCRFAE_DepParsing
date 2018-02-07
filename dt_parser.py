from optparse import OptionParser
import torch
import utils
from dt_model import dependency_tagging_model
import os, os.path
import pickle
import resource
import sys
import profile

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_CORE,(100,100))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="data/wsj10_tr")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="../data/en-universal-dev.conll.ptb")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="data/to_test")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--paramdec", dest="paramdec", help="Decoder parameters file", metavar="FILE",
                      default="paramdec.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="output/neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--tembedding", type="int", dest="tembedding_dims", default=50)

    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--numthread", type="int", dest="nthreads", default=1)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="output")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--distdim", type="int", dest="dist_dim", default=1)
    parser.add_option("--batch", type="int", dest="batchsize", default=100)
    parser.add_option("--tag_num",type="int",dest="tag_num",default = 10)

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--taglevel", type="int", dest="tag_level", default=50)
    parser.add_option("--fdim", type="int", dest="feat_param_dims", default=1)

    parser.add_option("--mode", action="store_false", dest="subtag_use", default=True)
    parser.add_option("--encoder_pass", type="int", dest="e_pass", default=1)
    parser.add_option("--decoder_pass", type="int", dest="d_pass", default=4)

    parser.add_option("--gpu", type="int", dest="gpu",default=-1, help='gpu id, set to -1 if use cpu mode')

    (options, args) = parser.parse_args()

    if options.gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    print 'Using external embedding:', options.external_embedding

    memory_limit()

    try:

        if options.predictFlag:
            with open(os.path.join(options.output, options.params), 'r') as paramsfp:
                words, pos, w2i, tag_map, flookup, stored_opt = pickle.load(paramsfp)
            stored_opt.external_embedding = options.external_embedding
            sentences = utils.read_data(options.conll_test, True)
            dep_tagging_model = dependency_tagging_model(words, pos, w2i, tag_map, flookup, options)
            dep_tagging_model.load(options.model)
            with open(os.path.join(options.output, options.paramdec), 'r') as paramdec:
                dep_tagging_model.recons_param,dep_tagging_model.lex_param = pickle.load(paramdec)
            testpath = os.path.join(options.output, 'test_pred.conll')
            test_res = dep_tagging_model.test(sentences)
            utils.eval(test_res,sentences)


        else:
            words, w2i, pos, tagCount, sentences = utils.read_data(options.conll_train, False)
            print 'Data read'
            tag_map = utils.round_tag(tagCount, options.tag_level)
            flookup = utils.traverse_feat(options.conll_train, tag_map,options.dist_dim)
            print 'Feature traversed'
            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump((words, pos, w2i, tag_map, flookup, options), paramsfp)
            print 'Parameters saved'
            dep_tagging_model = dependency_tagging_model(words, pos, w2i, tag_map, flookup, options)
            best_trees = {}
            tree_params = {}
            print 'Model building completed'
            for epoch in xrange(options.epochs):
                print 'Starting epoch', epoch
                print 'To train encoder'
                for n in range(options.e_pass):
                    dep_tagging_model.crf_train(sentences, best_trees, tree_params)
                print 'To train decoder'
                for n in range(options.d_pass):
                    dep_tagging_model.hard_em_train(sentences, tree_params)
                dep_tagging_model.save(os.path.join(options.output, os.path.basename(options.model) + str(epoch + 1)))
            with open(os.path.join(options.output,options.paramdec) + str(epoch + 1), 'w') as paramdec:
                pickle.dump((dep_tagging_model.recons_param,dep_tagging_model.lex_param),paramdec)
            print 'Training finished'
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)