
# coding: utf-8

# In[58]:

import theano
# theano.config.optimizer = 'None' 
# theano.config.exception_verbosity='high'
print theano.config.optimizer
print theano.config.exception_verbosity
import theano.tensor as T
import numpy as np


# In[59]:

# DIMENSIONS = 5


# In[60]:

# ETA = 0.5


# In[ ]:




# In[61]:

# def setEigenSoFar(self,idx, w, v):
#     self.i = idx
#     for j in range(self.i):
#         self.eigenValues[j] = w[j]
#         self.eigenVectors[:,j] = v[:,j]

# In[65]:

class GradientDescent():
    def __init__(self):
        return

    def __init__(self, m=5, n=5 , eta=0.5, numEig=5):
        self.DIMENSION_M = m
        self.DIMENSION_N = n
        self.ETA = eta
        self.NUM_EIG = numEig
        self.i = 0
        self.eigenValues = np.zeros(self.NUM_EIG, dtype=np.float32)
        self.eigenVectors = np.zeros((self.DIMENSION_N,self.NUM_EIG), dtype=np.float32)
        print 'initiliased eigenValues of shape', self.eigenValues.shape
        print 'initiliased eigenVectors of shape', self.eigenVectors.shape
        #         self.i = 0
        #         self.eigenValues = theano.shared( np.zeros(self.DIMENSIONS, dtype=np.float32) )
        #         self.eigenVectors = theano.shared( np.zeros((self.DIMENSIONS,self.DIMENSIONS), dtype=np.float32) )

    def compileTrainer(self):
        self.v = theano.shared( np.empty(self.DIMENSION_N, dtype=np.float32), name="v")
        self.X = T.fmatrix(name='X')
        #     self.X=X

        X_dot_v = T.dot(self.X,self.v)
        print 'X_dot_v', # theano.pp(X_dot_v)

        #         previousEigenContribution = \
        #         np.sum(
        #             [self.eigenValues[j]
        #              * T.dot(self.eigenVectors[j], self.v)
        #              * T.dot(self.eigenVectors[j], self.v)
        #                          for j in xrange(self.DIMENSIONS)])
        #         eigv_dot_v = T.dot(self.eigenVectors,self.v)
        #         previousEigenContribution = T.dot(self.eigenValues,(eigv_dot_v*eigv_dot_v))
        previousEigenContribution = np.sum( self.eigenValues[j]
                                           *T.dot(self.eigenVectors[:,j], self.v)
                                           *T.dot(self.eigenVectors[:,j], self.v)
                                               for j in xrange(self.NUM_EIG) )
        print 'previousEigenContribution', # theano.pp(previousEigenContribution)

        # np.sum(gd.eigenVectors[:,j]* gd.v for j in range(gd.i)).eval()

        self.objectiveFunction = T.dot(X_dot_v.T, X_dot_v) - previousEigenContribution
        print 'self.objectiveFunction', # theano.pp(self.objectiveFunction)

        grad_v = T.grad(self.objectiveFunction, self.v)
        print 'grad_v', # theano.pp(grad_v)

        y = self.v + self.ETA * grad_v
        updateFunction = y / y.norm(2)
        print 'y', # theano.pp(y)
        print 'updateFunction', # theano.pp(updateFunction)

        self.train = theano.function(
            inputs=[self.X],
            outputs=[updateFunction, grad_v, self.objectiveFunction, previousEigenContribution],
            updates=((self.v, updateFunction),)
        )
        print 'self.train trained'#, theano.pp(self.train)
        
    def getNextEigenValueAndVector(self, X_data, epsilon=1e-100, max_iteraions = 20, validateEpsilon=1):
        initVector = np.random.rand(self.DIMENSION_N).astype('float32')
        self.v.set_value(initVector)
        print 'Getting eigen value and vector #', self.i #, ' for: X = \n', X_data
        print 'Shape of X_data: ', X_data.shape
        # print 'Starting with a random vector\n', v.get_value()
        last_v_star = self.v.get_value()
        checkEpsilon = validateEpsilon
        for i in range(max_iteraions):
            (update, grad, objective, contrib) = self.train(X_data)
            # print update
            #             print eigv_dot_v
            # print contrib
            # print update, grad, objective, contrib
            # self.eigenVectors[:,self.i] = update
            rms = sum(update - last_v_star)**2
            # print 'rms difference: ', rms
            last_v_star = update
            if(rms < epsilon):
                checkEpsilon-=1
                if(checkEpsilon<=0):
                    break
            else:
                checkEpsilon=validateEpsilon
        print 'Number of iterations done: ', i
        self.eigenValues[self.i] = self.objectiveFunction.eval({self.X:X_data})
        #         currEigenValues = self.eigenValues.get_value()
        #         currEigenValues[self.i] = self.objectiveFunction.eval({self.X:X_data})
        #         self.eigenValues.set_value( currEigenValues )

        self.eigenVectors[:,self.i] = update
        #         currEigenVectors = self.eigenVectors.get_value()
        #         currEigenVectors[:,self.i] = last_v_star
        #         self.eigenVectors.set_value( currEigenVectors )

        #         print "Eigen Value:", currEigenValues[self.i]
        #         print "Eigen Vector:\n", currEigenVectors[:,self.i]
        #         self.i+=1
        #         return currEigenValues[self.i-1], currEigenVectors[:,self.i-1]

        print "Eigen Value:", self.eigenValues[self.i]
        print "Eigen Vector:\n", self.eigenVectors[:,self.i]
        self.i+=1
        return self.eigenValues[self.i-1], self.eigenVectors[:,self.i-1]

    def getAllEigenValuesAndVectors(self, X_data, epsilon=1e-100, max_iteraions = 20, validateEpsilon=1):
        for e in range(self.NUM_EIG):
            self.compileTrainer()
            self.getNextEigenValueAndVector(X_data, epsilon=epsilon, max_iteraions=max_iteraions, validateEpsilon=validateEpsilon)
        return self.eigenValues, self.eigenVectors
