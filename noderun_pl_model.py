import subprocess
from multiprocessing import Pool
import os
import numpy as np
import sys


def Thread(arg):
    print(arg)
    file = open('output/' + str(0) + '.log', 'w')
    subprocess.call(arg, shell=True, stdout=file)


def main():
    seed = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    batch = np.array([10, 50, 100, 200])
    # batch = batch.repeat(9)
    batch = np.tile(batch, 9)
    hidden = np.array([25, 50, 100])
    hidden = hidden.repeat(4)
    hidden = np.tile(hidden, 3)
    optim = {0: 'adam', 1: 'adagrad', 2: 'adadelta', 3: 'sgd'}
    op_idx = np.array([0, 1, 2, 3])
    op_idx = op_idx.repeat(12)
    lr = np.array([0.1, 0.01, 0.001])
    ed_pass = np.array([4, 8, 10])

    idx = [x for x in range(36)]

    arglist = []
    st = int(sys.argv[1])
    print(st)
    end = int(sys.argv[2])
    print(end)

    for i in range(st, end):
        opt_st = optim[op_idx[i]]
        pcmd = "python dt_pl_parser.py  --train data/wsj10_tr --tag_num 1 --hidden " + str(
            hidden[i]) + " " + "--batch " + str(
            batch[i]) + " " + "--optim " + opt_st + " " + "--do_eval --use_trigram " + "--sample_idx " + str(idx[i])
        arglist.append(pcmd)
        print(pcmd)

    p = Pool(4)
    p.map(Thread, arglist, chunksize=1)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
