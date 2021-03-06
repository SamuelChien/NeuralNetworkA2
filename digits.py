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
from IPython import embed

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
        prediction = np.dot(w.T, x) + b.T
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
        print x.shape
        print prediction

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
        print dEbydw.shape, dEbydb.shape
        print dEbydw, dEbydb


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

            if (epoch + 1) % 100 == 0:
                print "--------------- Set " + str(epoch + 1) + "------------------"
                print "Correction Rate For Train: " + str(self.EvaluateSimpleNNCorrection(inputs_train, target_train, w, b))
                print "Correction Rate For Test: " + str(self.EvaluateSimpleNNCorrection(inputs_test, target_test, w, b))
                print "Negative Log For Train: " + str(self.EvaluateSimpleNNNegativeLog(inputs_train, target_train, w, b))
                print "Negative Log For Test: " + str(self.EvaluateSimpleNNNegativeLog(inputs_test, target_test, w, b))

        return w, b

    def EvaluateSimpleNNCorrection(self, inputs_train, target_train, w, b):
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

    def EvaluateSimpleNNNegativeLog(self, inputs_train, target_train, w, b):
        # Forward propagation
        predictionPercentage = self.softmax(self.forwardPropagation(w, inputs_train, b)   )
        targetPercentage = self.getPercentageListFromTarget(target_train)
        return -np.mean(targetPercentage * np.log(predictionPercentage) + (1 - targetPercentage) * np.log(1 - predictionPercentage))


    def plotCorrectionRateGraph(self):
        trainLine, = plt.plot([2, 5, 10, 15, 20, 40, 60, 80, 100], [0.176, 0.358, 0.55, 0.622, 0.655, 0.718, 0.736, 0.758, 0.776], label='Train')
        testLine, = plt.plot([2, 5, 10, 15, 20, 40, 60, 80, 100], [0.165, 0.29, 0.45, 0.528, 0.555, 0.615, 0.645, 0.652, 0.652], label='Test')
        plt.ylabel('Correctness(%)')
        plt.xlabel('Learning Times')
        plt.legend(handles=[trainLine, testLine])
        plt.show()

    def plotNegativeLogRateGraph(self):
        trainLine, = plt.plot([2, 5, 10, 15, 20, 40, 60, 80, 100], [0.324, 0.322, 0.319, 0.316, 0.313, 0.303, 0.296, 0.291, 0.288], label='Train')
        testLine, = plt.plot([2, 5, 10, 15, 20, 40, 60, 80, 100], [0.324, 0.323, 0.320, 0.318, 0.315, 0.306, 0.301, 0.295, 0.292], label='Test')
        plt.ylabel('Negative Log')
        plt.xlabel('Learning Times')
        plt.legend(handles=[trainLine, testLine])
        plt.show()


    def getCorrectFaces(self, w, b):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        predictionPercentage = self.forwardPropagation(w, inputs_test, b)
        predictionPercentage = self.softmax(predictionPercentage)
        predictionPercentageList =  predictionPercentage.T
        prediction = []
        for predictionInstance in predictionPercentageList:
            prediction.append(argmax(predictionInstance))


        target = target_test[0]
        totalCount = len(target)
        correctCount = 0
        f, axarr = plt.subplots(4, 5)
        for index in range(totalCount):
            if target[index] == prediction[index] and correctCount < 20:
                axarr[int(correctCount)/5, correctCount%5].imshow(inputs_test.T[index].reshape(28,28),  cmap=cm.gray)
                axarr[correctCount/5, correctCount%5].get_yaxis().set_visible(False)
                axarr[correctCount/5, correctCount%5].get_xaxis().set_visible(False)
                correctCount+=1

        plt.show()


    def getIncorrectFaces(self, w, b):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        predictionPercentage = self.forwardPropagation(w, inputs_test, b)
        predictionPercentage = self.softmax(predictionPercentage)
        predictionPercentageList =  predictionPercentage.T
        prediction = []
        for predictionInstance in predictionPercentageList:
            prediction.append(argmax(predictionInstance))


        target = target_test[0]
        totalCount = len(target)
        correctCount = 0
        f, axarr = plt.subplots(2, 5)
        for index in range(totalCount):
            if target[index] != prediction[index] and correctCount < 10:
                axarr[int(correctCount)/5, correctCount%5].imshow(inputs_test.T[index].reshape(28,28),  cmap=cm.gray)
                axarr[correctCount/5, correctCount%5].get_yaxis().set_visible(False)
                axarr[correctCount/5, correctCount%5].get_xaxis().set_visible(False)
                correctCount+=1

        plt.show()

    def partFive(self):
        learningRate = 0.01
        momentum = 0.5
        num_epochs = 3000
        w, b = self.trainSimpleNN(learningRate, momentum, num_epochs)
        #self.plotCorrectionRateGraph()
        #self.plotNegativeLogRateGraph()
        #self.getCorrectFaces(w, b)
        #self.getIncorrectFaces(w, b)

    def partSix(self):
        learningRate = 0.01
        momentum = 0.5
        num_epochs = 1000
        w, b = self.trainSimpleNN(learningRate, momentum, num_epochs)


        #Code for displaying a feature from the weight matrix mW
        fig = figure(1)
        ax = fig.gca()
        heatmap = ax.imshow(w.T[0].reshape((28,28)), cmap = cm.coolwarm)
        fig.colorbar(heatmap, shrink = 0.5, aspect=5)
        show()
    def EvaluateTanhNNCorrection(self, inputs_train, target_train, w0, b0, w1, b1):
        # Forward propagation
        L0 = np.tanh(np.dot(w0.T, inputs_train) + b0) #zj
        L1 = np.tanh(np.dot(w1.T, L0) + b1)
        output = self.softmax(L1)
        predictionPercentageList =  output.T
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


    def trainTanhNN(self, hidden_num, learningRate, momentum, num_epochs, batch_size=-1):
        inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = self.loadAllData()
        # embed()
        x = inputs_train
        num_train_cases = x.shape[1]
        target = self.getPercentageListFromTarget(target_train)

        #Load sample weights for the multilayer neural network
        snapshot = cPickle.load(open("snapshot50.pkl"))
        w0 = snapshot["W0"]
        b0 = snapshot["b0"].reshape((300,1))
        w1 = snapshot["W1"]
        b1 = snapshot["b1"].reshape((10,1))

        dw0 = np.zeros(w0.shape)
        db0 = np.zeros(b0.shape)
        dw1 = np.zeros(w1.shape)
        db1 = np.zeros(b1.shape)

        if batch_size < 1:
            batch_size = num_train_cases
        # embed()
        train_error = []
        valid_error = []
        def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
            assert len(inputs) == len(targets)
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batchsize]
                else:
                    excerpt = slice(start_idx, start_idx + batchsize)
                yield inputs[excerpt], targets[excerpt]
        # embed()
        for epoch in xrange(num_epochs):
            # Forward propagation with tanh
            counter = 0
            for batch in iterate_minibatches(x.T, target.T, batch_size, shuffle=True):
                # print(counter)
                counter += 1
                inputs, targets = batch
                inputs = inputs.T
                targets = targets.T
                # embed()
                L0 = np.tanh(np.dot(w0.T, inputs) + b0) #zj
                L1 = np.dot(w1.T, L0) + b1
                output = self.softmax(L1)
                # print(output.shape)
                # Compute Deriv
                dCdL1 = output - targets #softmax
                dCdw1 =  np.dot(L0, dCdL1.T ) #layer1
                dCdb1 = np.sum(dCdL1, axis=1).reshape(-1,1)
                dCdw0 = np.dot(inputs, ((1- L0**2)*np.dot(w1, dCdL1)).T)
                dCdb0 = np.sum(((1- L0**2)*np.dot(w1, dCdL1)), axis=1).reshape(-1,1)

                #%%%% Update the weights at the end of the epoch %%%%%%
                dw0 = momentum * dw0 - (learningRate / batch_size) * dCdw0
                db0 = momentum * db0 - (learningRate / batch_size) * dCdb0
                dw1 = momentum * dw1 - (learningRate / batch_size) * dCdw1
                db1 = momentum * db1 - (learningRate / batch_size) * dCdb1

                w0 = w0 + dw0
                b0 = b0 + db0
                w1 = w1 + dw1
                b1 = b1 + db1
            if (epoch + 1) % 100 == 0:
                print "--------------- Set " + str(epoch + 1) + "------------------"
                print "Correction Rate For Train: " + str(self.EvaluateTanhNNCorrection(inputs_train, target_train, w0, b0, w1, b1))
                print "Correction Rate For Test: " + str(self.EvaluateTanhNNCorrection(inputs_test, target_test, w0, b0, w1, b1))
                # print "Negative Log For Train: " + str(self.EvaluateSimpleNNNegativeLog(inputs_train, target_train, w, b))
                # print "Negative Log For Test: " + str(self.EvaluateSimpleNNNegativeLog(inputs_test, target_test, w, b))

        return w0, b0, w1, b1

    def partSeven(self):
        learningRate = 0.001
        momentum = 0.5
        num_epochs = 4000
        hidden_num = 300
        w0, b0, w1, b1 = self.trainTanhNN(hidden_num, learningRate, momentum, num_epochs)
    
    def partEight(self):
        print "hello world"

    def partNine(self):
        learningRate = 0.001
        momentum = 0.5
        num_epochs = 2000
        hidden_num = 300
        batch_size = 50
        w0, b0, w1, b1 = self.trainTanhNN(hidden_num, learningRate, momentum, num_epochs, batch_size)


    def partTen(self):
        print "hello world"

if __name__ == "__main__":
    hw = Assignment()
    #hw.partOne()
    #hw.partTwo()
    #hw.partThree()
    #PART 4: RuntimeWarning: overflow encountered in multiply
    #hw.partFour()
    #hw.partFive()
    #hw.partSix()
    #hw.partSeven()
    #hw.partEight()
    #hw.partNine()
    #hw.partTen()
