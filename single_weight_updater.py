import numpy as np
import ok_boomer_tdleaf.machine_learning as ml


weights = ml.load_weights()
rewards, features = ml.load_data('training/data.log')
new_weights = ml.update_weights(weights, rewards, features, ml.LR, ml.DECAY)
np.savetxt(ml.WEIGHT_FILE, new_weights, delimiter=',')