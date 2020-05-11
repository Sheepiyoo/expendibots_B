import numpy as np
import glob

DECAY = 0.7
LR = 0.05
WEIGHT_FILE = 'weights.csv'

def load_data(filename):
    """
    Load rewards and feature data from log file
    """
    data = np.loadtxt(filename, delimiter=',')
    rewards, features = np.split(data, [1], axis=1)
    return rewards, features


def update_weights(weights, rewards, features, lr, gamma):
    """
    Update feature weights
    """
    num_steps = len(rewards)
    accumulator = np.zeros(weights.shape)
    r1 = rewards[:num_steps-1]
    r2 = rewards[1:]
    differences = r2 - r1
    #print(differences)
    print(rewards)
    
    #Iterate through each step made in game
    for i in range(0, num_steps-1):
        decay_term = 0
        coefficient = 1
        for j in range(i, num_steps):
            decay_term += differences[i] * coefficient
            coefficient*=gamma
            
        accumulator += (1+rewards[i]**2)*features[i]*decay_term
    
    #Update weights
    weights = weights + lr * accumulator
        
    return weights

def load_weights():
    return np.loadtxt(WEIGHT_FILE, delimiter=',')

if __name__ == "__main__":
    print("Updating weights")
    weights = np.loadtxt(WEIGHT_FILE, delimiter=',')

    print(weights.shape)

    files = glob.glob('training/data.log')
    r, f = load_data(files[0])
    print(r.shape)
    print(f.shape)

    print(weights)
    new_weights = update_weights(weights, r, f, LR, DECAY)
    print(new_weights)

    np.savetxt(WEIGHT_FILE, new_weights, delimiter=',')

