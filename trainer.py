import subprocess
import logging
import ok_boomer_tdleaf.machine_learning as ml
import numpy as np

FORMAT = '%(asctime)s: %(levelname)s: %(message)s'

formatter = logging.Formatter(FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='game-training.log', mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

num_tests = 0
MAX_TESTS = 20

logger.debug("------------------- New Training Session ----------------")
swap = False

while num_tests < MAX_TESTS:

    white = "ok_boomer_tdleaf"
    black = "ok_boomer_alphabeta" #"ok_boomer_greedy" 
    
    if swap:
        temp = black
        black = white
        white = temp

    referee = "referee_elon_musk"
    p = subprocess.Popen(["python", "-m", referee, white, black])
    p.wait()

    # Update weights
    weights = ml.load_weights()

    rewards, features = ml.load_data('training/data.log')
    new_weights = ml.update_weights(weights, rewards, features, ml.LR, ml.DECAY)
    logger.debug(new_weights)
    np.savetxt(ml.WEIGHT_FILE, new_weights, delimiter=',')

    num_tests += 1
    swap = not swap

                                                                