import os
import sys

a = "\' cd dependency_Tagging && python noderun_pl_model.py " + str(
    sys.argv[1]) + " " + str(sys.argv[2]) + "\'"
os.system("ssh node" + str(sys.argv[3]) + " "+ a)