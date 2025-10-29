'''
Example script to prepare model weights for Blitz engine.
When you call pull_model, it will load the model weights from the local path to host memory
'''

import blitz_lib

model_path = "your-local-model-path"
tp_size = your-tp-size
blitz_lib.pull_model(model_path, tp_size, tp_size, 1)
