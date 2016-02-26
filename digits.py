from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import cPickle
import os
from scipy.io import loadmat


    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = tanh_layer(L0, W1, b1)
    #L1 = dot(W1.T, L0) + b1 if you don't want tanh at the top layer
    output = softmax(L1)
    return L0, L1, output
    


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    #dCdW1 =  dot(L0, dCdL1.T ) if you don't want the nonlinearity at the top layer
    dCdW1 =  dot(L0, ((1- L1**2)*dCdL1).T )
    


class Assignment:

    def __init__(self):
        self.matFileName = "mnist_all.mat"
        self.numbers = self.loadData()

    def loadData(self):
        return loadmat(self.matFileName)

    def getDataByKey(self, numberKey, count):
        resultDataSet = []
        requestedDataSet = self.numbers[numberKey]
        for index in range(count): 
            resultDataSet.append(requestedDataSet[index]/float(255))
        return resultDataSet

    def loadAllData(self):
        trainingCount = 60
        validationCount = 20
        testCount = 40
        inputs_train = []
        inputs_valid = []
        inputs_test = []
        target_train = []
        target_valid = []
        target_test = []

        for digitIndex in range(10):
            numberKey = "train" + str(digitIndex)
            totalDigitData = self.getDataByKey(numberKey, 120)
            inputs_train += totalDigitData[0:60]
            inputs_valid += totalDigitData[60:80]
            inputs_test += totalDigitData[80:120]
            for i in range(trainingCount):
                target_train.append(digitIndex)
            for j in range(validationCount):
                target_valid.append(digitIndex)
            for k in range(testCount):
                target_test.append(digitIndex)


        inputs_train = np.asarray(inputs_train).T
        inputs_valid = np.asarray(inputs_valid).T
        inputs_test = np.asarray(inputs_test).T
        target_train = np.asarray([target_train])
        target_valid = np.asarray([target_valid])
        target_test = np.asarray([target_test])

        return inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test

    def partOne(self):
        f, axarr = plt.subplots(10, 10)
        for i in range(10):
            numberKey = "train" + str(i)
            data = self.getDataByKey(numberKey, 10)
            for j in range(len(data)):
                axarr[i, j].imshow(data[j].reshape(28,28),  cmap=cm.gray)
                axarr[i, j].get_yaxis().set_visible(False)
                axarr[i, j].get_xaxis().set_visible(False)

        plt.show()

    def forwardPropagation(self, w, x, b):
        alpha = np.dot(w.T, x) + b.T
        prediction = 1 / (1 + np.exp(-alpha))
        return prediction

    def softmax(self, y):
        '''Return the output of the softmax function for the matrix of output y. y
        is an NxM matrix where N is the number of outputs for a single case, and M
        is the number of cases'''
        return exp(y)/tile(sum(exp(y),0), (len(y),1))

    def partTwo(self):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        x = np.asarray([inputs_train.T[0]]).T
        w = 0.01 * np.random.randn(inputs_train.shape[0], 10)
        b = np.zeros((target_train.shape[0], 10))
        predictionPercentage = self.forwardPropagation(w, x, b)
        resultPercentageList = self.softmax(predictionPercentage).T[0]
        prediction = argmax(resultPercentageList)

    def calculateCost(self, prediction, target):
        return -sum(target*log(prediction))

    def getPercentageListFromTarget(self, target): 
        result = []
        for targetInstance in target[0]:
            instancePercentage = [0] * 10
            instancePercentage[targetInstance] = 1
            result.append(instancePercentage)

        return np.asarray(result).T

    def calculateGradientDecent(self, w, x, b, target_train):
        # Forward propagation
        prediction = self.forwardPropagation(w, x, b)    
        prediction = self.softmax(prediction)    
        target = self.getPercentageListFromTarget(target_train)
        # Compute Deriv
        dEbydlogit = prediction - target
        dEbydw = np.dot(x, dEbydlogit.T)
        dEbydb = np.sum(dEbydlogit, axis=1).reshape(-1, 1).T
        return dEbydw, dEbydb
    
    def partThree(self):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        x = inputs_train
        w = 0.01 * np.random.randn(inputs_train.shape[0], 10)
        b = np.zeros((target_train.shape[0], 10))
        dEbydw, dEbydb = self.calculateGradientDecent(w, x, b, target_train)


    def f(self, x, y, theta):
        x = vstack( (ones((1, x.shape[1])), x))
        return sum( (y - dot(theta.T,x)) ** 2)

    def df(self, x, y, theta):
        x = vstack( (ones((1, x.shape[1])), x))
        # RuntimeWarning: overflow encountered in multiply
        return -2*dot(x, (y-dot(theta.T, x).T))
        
    def calculateGradientDecentByFiniteDifference(self, f, df, x, y, init_t, alpha):
        EPS = 1e-5   #EPS = 10**(-5)
        prev_t = init_t-10*EPS
        t = init_t.copy()
        
        while norm(t - prev_t) >  EPS:
            prev_t = t.copy()
            t -= alpha*self.df(x, y, t)
        return t

    def partFour(self):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        x = inputs_train
        w = 0.01 * np.random.randn(inputs_train.shape[0], 10)
        b = np.zeros((target_train.shape[0], 10))
        dEbydw, dEbydb = self.calculateGradientDecent(w, x, b, target_train)

        y = self.getPercentageListFromTarget(target_train).T
        theta0 = vstack((dEbydw, dEbydb))
        theta = self.calculateGradientDecentByFiniteDifference(self.f, self.df, x, y, theta0, 0.0000010)

    def trainSimpleNN(self, learningRate, momentum, num_epochs):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        x = inputs_train
        w = 0.01 * np.random.randn(inputs_train.shape[0], 10)
        b = np.zeros((target_train.shape[0], 10))
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)

        train_error = []
        valid_error = []
        num_train_cases = inputs_train.shape[1]
        for epoch in xrange(num_epochs):
            # Forward propagation
            prediction = self.forwardPropagation(w, x, b)    
            prediction = self.softmax(prediction)    
            target = self.getPercentageListFromTarget(target_train)
            
            # Compute Deriv
            dEbydlogit = prediction - target
            dEbydw = np.dot(x, dEbydlogit.T)
            dEbydb = np.sum(dEbydlogit, axis=1).reshape(-1, 1).T

            #%%%% Update the weights at the end of the epoch %%%%%%
            dw = momentum * dw - (learningRate / num_train_cases) * dEbydw
            db = momentum * db - (learningRate / num_train_cases) * dEbydb

            w = w + dw
            b = b + db



        print self.EvaluateSimpleNN(inputs_train, target_train, w, b)
        print self.EvaluateSimpleNN(inputs_valid, target_valid, w, b)
        print self.EvaluateSimpleNN(inputs_test, target_test, w, b)
        return w, b

    def EvaluateSimpleNN(self, inputs_train, target_train, w, b):
        # Forward propagation
        predictionPercentage = self.forwardPropagation(w, inputs_train, b)    
        predictionPercentage = self.softmax(predictionPercentage)   
        predictionPercentageList =  predictionPercentage.T
        prediction = []
        for predictionInstance in predictionPercentageList:
            prediction.append(argmax(predictionInstance))

        target = target_train[0]
        totalCount = len(target)
        correctCount = 0
        for index in range(totalCount):
            if target[index] == prediction[index]:
                correctCount+=1

        return float(correctCount)/totalCount

    def EvaluateSimpleNNEntroy(self, inputs_train, target_train, w, b):
        # Forward propagation
        predictionPercentage = self.forwardPropagation(w, inputs_train, b)    
        predictionPercentage = self.softmax(predictionPercentage)   
        predictionPercentageList =  predictionPercentage.T
        prediction = []
        for predictionInstance in predictionPercentageList:
            prediction.append(argmax(predictionInstance))

        target = target_train[0]
        totalCount = len(target)
        correctCount = 0
        for index in range(totalCount):
            if target[index] == prediction[index]:
                correctCount+=1

        return float(correctCount)/totalCount

        # h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
        # h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
        # logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
        # prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
        # CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  


    def partFive(self):
        learningRate = 0.01
        momentum = 0.5
        num_epochs = 1000
        w, b = self.trainSimpleNN(learningRate, momentum, num_epochs)


if __name__ == "__main__":
    hw = Assignment()
    # hw.partOne()
    # hw.partTwo()
    # hw.partThree()

    #PART 4: RuntimeWarning: overflow encountered in multiply
    #hw.partFour()

    hw.partFive()


    # #Load sample weights for the multilayer neural network
    # snapshot = cPickle.load(open("snapshot50.pkl"))
    # W0 = snapshot["W0"]
    # b0 = snapshot["b0"].reshape((300,1))
    # W1 = snapshot["W1"]
    # b1 = snapshot["b1"].reshape((10,1))
    # print W0.shape
    # print b0.shape

# #Load one example from the training set, and run it through the
# #neural network
# x = M["train5"][148:149].T    
# L0, L1, output = forward(x, W0, b0, W1, b1)
# #get the index at which the output is the largest
# y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################