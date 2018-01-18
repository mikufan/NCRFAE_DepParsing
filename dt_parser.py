from optparse import OptionParser
import torch
import utils
from dt_model import dependency_tagging_model




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE",
                      default="../data/en-universal-train.conll.ptb")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE",
                      default="../data/en-universal-dev.conll.ptb")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE",
                      default="../data/en-universal-test.conll.ptb")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE",
                      default="neuralfirstorder.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--tembedding",type="int", dest="tembedding_dims",default=50)

    parser.add_option("--epochs", type="int", dest="epochs", default=30)
    parser.add_option("--numthread", type="int", dest="nthreads", default=1)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)

    parser.add_option("--optim", type="string", dest="optim", default='adam')
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=125)
    parser.add_option("--distdim",type = "int",dest="dist_dim",default = 6)
    parser.add_option("--batch",type = "int",dest="batchsize",default = 10)

    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--taglevel", type="int",dest="tag_level", default =100)
    parser.add_option("--fdim",type="int",dest="feat_param_dims",default = 1)

    parser.add_option("--mode",action="store_false",dest="subtag_use",default = True)

    (options, args) = parser.parse_args()
    torch.set_num_threads(options.nthreads)
    print torch.get_num_threads()

    print 'Using external embedding:', options.external_embedding

    if options.predictFlag:
        None

    else:
        words, w2i, pos ,tagCount,sentences = utils.read_data(options.conll_train)
        tag_map = utils.round_tag(tagCount,options.tag_level)
        flookup = utils.traverse_feat(options.conll_train,tag_map)
        dep_tagging_model = dependency_tagging_model(words,pos,w2i,tag_map,flookup,sentences,options)
        best_trees = {}
        for epoch in xrange(options.epochs):
            print 'Starting epoch', epoch
            dep_tagging_model.train(sentences,best_trees)