'''
Example script to prepare model weights for Blitz engine.
When you call pull_model, it will load the model weights from the local path to host memory
'''

import sys
import blitz_lib

if __name__ == "__main__":
    model = sys.argv[1]
    tp_size = int(sys.argv[2])
    blitz_lib.pull_model(model, tp_size, tp_size, 1)
