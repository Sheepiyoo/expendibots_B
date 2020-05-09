import subprocess
import logging
import ok_boomer_nntdleaf.neural_network as nn
import numpy as np
import os

FORMAT = '%(asctime)s: %(levelname)s: %(message)s'

formatter = logging.Formatter(FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='game-training.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

num_tests = 0
MAX_TESTS = 2000

logger.debug("------------------- New Training Session ----------------")
swap = False

if not os.path.exists("NN_i-h.csv") or not os.path.exists ("NN_h-o.csv"):
    nn.reset_weights("NN_i-h.csv", "NN_h-o.csv", 64, 10, 1)

while num_tests < MAX_TESTS:

    black = "ok_boomer_alphabeta"
    white =  "ok_boomer_nntdleaf"#"ok_boomer_greedy" 
    
    if swap:
        temp = black
        black = white
        white = temp

    referee = "referee_elon_musk"
    p = subprocess.Popen(["python", "-m", referee, white, black])
    p.wait()

    # Update weights
    nn.wrapper_update()

    num_tests += 1
    swap = not swap

                                                                