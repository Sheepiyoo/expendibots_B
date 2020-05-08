import numpy as np

class neuralNetwork:
    
    def __init__(self, inodes, hnodes, onodes, learningrate, gamma):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate
        self.gamma = gamma
        
        self.activationfunction = sigmoid
        self.activationfunction2 = np.tanh
        
        pass
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv(self, x):
        return sigmoid(x)*(1 - sigmoid(x))
    
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
                coefficient*=gamma
                
            who_accumulator += (1+rewards[i]**2)*features[i]*decay_term

        #Update weights
        self.who = self.lr * who_accumulator
        self.wih = self.lr * wih_accumulator


        
        self.who += self.lr * np.dot((oerrors * (foutputs > 0)*1.0), np.transpose(houtputs))
        self.wih += self.lr * np.dot((herrors * (houtputs > 0)*1.0), np.transpose(inputs)) 
        
        pass
    
    
    def predict(self, inputslist, reward):
        inputs = np.array(inputslist, ndmin=1).T
        targets = np.array(reward, ndmin=1).T 
        
        # Forward propagation
        hinputs = np.dot(self.wih, inputs)
        houtputs = self.activationfunction(hinputs)
        
        finputs = np.dot(self.who, houtputs)               # Evaluation function: E = w.hx
        foutputs = self.activationfunction2(finputs)       # Reward function    : tanh(E)
        
        return foutputs