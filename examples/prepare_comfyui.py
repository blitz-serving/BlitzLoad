'''
Example script to prepare model weights for Blitz engine.
When you call pull_diffusion_model, it will load the model weights from the local directory to host memory
'''

import sys
import blitz_lib

if __name__ == "__main__":
    models_dir = sys.argv[1]
    blitz_lib.pull_diffusion_model(models_dir)
