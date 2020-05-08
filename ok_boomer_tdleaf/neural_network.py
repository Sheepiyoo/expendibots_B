import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))

class neuralNetwork:
    
    def __init__(self, inodes, hnodes, onodes, learningrate):
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate
        
        self.activationfunction = lambda x: scipy.special.expit(x)
        
        pass
    
    
    def train(self, inputslist, targetslist):
        inputs = numpy.array(inputslist, ndmin=2).T
        targets = numpy.array(targetslist, ndmin=2).T
        
        hinputs = numpy.dot(self.wih, inputs)
        houtputs = self.activationfunction(hinputs)
        
        finputs = numpy.dot(self.who, houtputs)
        foutputs = self.activationfunction(finputs)
        
        oerrors = targets - foutputs
        herrors = numpy.dot(self.who.T, oerrors) 
        
        self.who += self.lr * numpy.dot((oerrors * (foutputs > 0)*1.0), numpy.transpose(houtputs))
        self.wih += self.lr * numpy.dot((herrors * (houtputs > 0)*1.0), numpy.transpose(inputs)) 
        
        pass
    
    
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hinputs = numpy.dot(self.wih, inputs)
        houtputs = self.activationfunction(hinputs)
        
        finputs = numpy.dot(self.who, houtputs)
        foutputs = self.activationfunction(finputs)
        
        return foutputs

def training(datalist, network):
    for record in datalist:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01        
        targets = numpy.zeros(network.onodes) + 0.01
        targets[int(all_values[0])] = 0.99
        network.train(inputs, targets)
        pass
    pass


nineimages = []
fourimages = []
def testing(datalist, network, scorecard, disease, num):
    wronganswers = [0,0,0,0,0,0,0,0,0,0]
    for record in datalist:
        all_values = record.split(',')
        answer = int(all_values[0])
        if(disease == "pigmentosa"):
            inputs = pigmentosa(all_values[1:], num)
        if(disease == "macula"):
            inputs = maculadensa(all_values[1:], num)
        if(disease == "line"):
            inputs = removelines(all_values[1:], num)
        if(disease == "keep"):
            inputs = keeplines(all_values[1:], num)
        outputs = network.query(inputs)
        
        label = numpy.argmax(outputs)
        if (label == answer):
            scorecard.append(1)
            #if(label ==4):
                #fourimages.append(inputs)
        else:
            scorecard.append(0)
            # This commented out code was used to investigate unexpected results - see Discussion
            #if(label == 9 and num == 5):
                #ninelist[answer] += 1
                #if(answer == 4 and num == 5):
                    #nineimages.append(inputs)
            #wronganswers[answer] += 1
            pass    
        pass
    #print(num, wronganswers)
    #print("ninelist is", ninelist)
    pass
