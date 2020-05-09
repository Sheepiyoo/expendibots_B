import numpy as np
from scipy.special import expit
import sys

LR = 0.1
DECAY = 0.995
WEIGHT_FILE = 'training/nn_data.log'

class neuralNetwork:
    
    def __init__(self, inodes, hnodes, onodes, learningrate, gamma, ihweights, howeights):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.ihpath = ihweights
        self.hopath = howeights

        self.wih = np.loadtxt(ihweights, delimiter=',')
        self.who = np.loadtxt(howeights, delimiter=',')

        self.lr = learningrate
        self.gamma = gamma
        
        self.activationfunction = expit
        self.activationfunction2 = np.tanh
        
        pass
    
    def sigmoid(self, x):
        return 1/(1+expit(-x))

    def sigmoid_deriv(self, x):
        return expit(x)*(1 - expit(x))
    
    def update_weights(self, features, rewards):

        num_steps = len(rewards)

        wih_accumulator = np.zeros(self.wih.shape)
        who_accumulator = np.zeros(self.who.shape)
        
        differences = rewards[1:]- rewards[:num_steps-1]

        for i in range(0, num_steps-1):

            # Forward propagation
            inputs = np.array(features[i], ndmin=1).T
            hinputs = np.dot(self.wih, inputs)
            houtputs = self.activationfunction(hinputs)
            finputs = np.dot(self.who, houtputs)               # Evaluation function: E = w.hx
            foutputs = self.activationfunction2(finputs)       # Reward function    : tanh(E)

            decay_term = 0
            coefficient = 1
            

            for j in range(i, num_steps):
                decay_term += differences[i] * coefficient
                coefficient*=self.gamma
            
            for r in range(0, self.wih.shape[0]):
                for c in range(0, self.wih.shape[1]):
                    delta = (1+rewards[r]**2)*self.who[r]*self.sigmoid_deriv(hinputs[r])*features[i][c]
                    #print(delta)
                    wih_accumulator[r][c] += delta

            wih_accumulator *= decay_term
        
            #print(who_accumulator.shape)
            #test = (1+rewards[i]**2)*houtputs[i]*decay_term
        
            who_accumulator += (1+rewards[i]**2)*houtputs*decay_term

        #Update weights
        self.who += self.lr * who_accumulator
        self.wih += self.lr * wih_accumulator
    
    def predict(self, inputslist):
        inputs = np.array(inputslist, ndmin=1).T

        # Forward propagation
        hinputs = np.dot(self.wih, inputs)
        houtputs = self.activationfunction(hinputs)
        
        finputs = np.dot(self.who, houtputs)               # Evaluation function: E = w.hx
        foutputs = self.activationfunction2(finputs)       # Reward function    : tanh(E)
        
        return foutputs
    
    def save_weights(self):
        np.savetxt(self.ihpath, self.wih, delimiter=',')
        np.savetxt(self.hopath, self.who, delimiter = ',')

def load_data(filename):
    """
    Load rewards and feature data from log file
    """
    data = np.loadtxt(filename, delimiter=',')
    rewards, features = np.split(data, [1], axis=1)
    return rewards, features

def wrapper_predict(features):
    nn = neuralNetwork(64, 10, 1, LR, DECAY, "NN_i-h.csv", "NN_h-o.csv")
    eval = nn.predict(np.array(features))
    return float(eval)

def wrapper_update():
    nn = neuralNetwork(64, 10, 1, LR, DECAY, "NN_i-h.csv", "NN_h-o.csv")
    rewards, features = load_data(WEIGHT_FILE)
    nn.update_weights(features, rewards)
    print("Saving weights")
    nn.save_weights()

def reset_weights(ih, ho, inodes, hnodes, onodes):
    wih = np.random.normal(0.0, 1.0, (hnodes, inodes))
    woh = np.random.normal(0.0, 1.0, (onodes, hnodes))
    np.savetxt(ih, wih, delimiter=',')
    np.savetxt(ho, woh, delimiter=',')


if __name__ == "__main__":
    nn = neuralNetwork(64, 10, 1, LR, DECAY, "NN_i-h.csv", "NN_h-o.csv")
    rewards, features = load_data(WEIGHT_FILE)
    nn.update_weights(features, rewards)
    print("Saving weights")
    nn.save_weights()