
# coding: utf-8

# In[1]:

import theano
# theano.config.optimizer = 'None' 
# theano.config.exception_verbosity='high'
print theano.config.optimizer
print theano.config.exception_verbosity
import theano.tensor as T
import numpy as np


# In[2]:

# DIMENSIONS = 5


# In[3]:

# ETA = 0.5


# In[ ]:




# In[4]:

# def setEigenSoFar(self,idx, w, v):
#     self.i = idx
#     for j in range(self.i):
#         self.eigenValues[j] = w[j]
#         self.eigenVectors[:,j] = v[:,j]


# In[5]:

X_data = np.array(
[[ 0.41939831,  0.73336226,  0.81437075,  0.72610056,  0.76305652,],
 [ 0.66616774,  0.60886431,  0.92339832,  0.1030873 ,  0.31538147,],
 [ 0.04194576,  0.29987422,  0.7813136,   0.55106729,  0.58304852,],
 [ 0.05681328,  0.09614379,  0.83366364,  0.37384903,  0.06315704,],
 [ 0.97372508,  0.61315203,  0.2581135,   0.44524547,  0.21198589,],]
, dtype=np.float32
)
print X_data


# In[10]:

class GradientDescent():
    def __init__(self):
        return

    def __init__(self, dim=5, eta=0.5, numEig=5):
        self.DIMENSIONS = dim
        self.ETA = eta
        self.NUM_EIG = numEig
        self.i = 0
        self.eigenValues = np.zeros(self.NUM_EIG, dtype=np.float32)
        self.eigenVectors = np.zeros((self.DIMENSIONS,self.NUM_EIG), dtype=np.float32)
        #         self.i = 0
        #         self.eigenValues = theano.shared( np.zeros(self.DIMENSIONS, dtype=np.float32) )
        #         self.eigenVectors = theano.shared( np.zeros((self.DIMENSIONS,self.DIMENSIONS), dtype=np.float32) )

    def compileTrainer(self):
        self.v = theano.shared( np.empty(self.DIMENSIONS, dtype=np.float32), name="v")
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
                                           *T.dot(self.eigenVectors[j], self.v)
                                           *T.dot(self.eigenVectors[j], self.v)
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
            outputs=[updateFunction, grad_v, self.objectiveFunction],
            updates=((self.v, updateFunction),)
        )
        print 'self.train trained'#, theano.pp(self.train)

    def getNextEigenValueAndVector(self, X_data, epsilon=1e-100, max_iteraions = 20, validateEpsilon=1, initVector = np.array(0).astype('float32')):
        if( (initVector.size)<=1 ):
            initVector = np.random.rand(self.DIMENSIONS).astype('float32')
        self.v.set_value(initVector)
        print 'Getting eigen value and vector #', self.i, ' for: X = \n', X_data
        # print 'Starting with a random vector\n', v.get_value()
        last_v_star = self.v.get_value()
        checkEpsilon = validateEpsilon
        for i in range(max_iteraions):
            (update, grad, objective) = self.train(X_data)
            #             print update
            #             print eigv_dot_v
            #             print contrib
            # print update, grad, objective, contrib
            # self.eigenVectors[:,self.i] = update
            rms = sum(update - last_v_star)**2
            print 'rms difference: ', rms
            last_v_star = update
            if(rms < epsilon):
                checkEpsilon-=1
                if(checkEpsilon<=0):
                    break
            else:
                checkEpsilon=validateEpsilon
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



# In[11]:

gd1 = GradientDescent( eta=0.5, numEig=3)


# In[12]:

gd1.compileTrainer()


# In[13]:

gd1.getNextEigenValueAndVector(X_data, max_iteraions=100, epsilon=1e-50, validateEpsilon=10)


# In[14]:

print gd1.i
print gd1.eigenValues
print gd1.eigenVectors
# print gd1.eigenValues.get_value()
# print gd1.eigenVectors.get_value()


# In[15]:

gd1.compileTrainer()
gd1.getNextEigenValueAndVector(X_data, max_iteraions=100, epsilon=1e-50, validateEpsilon=10)


# In[ ]:

print gd1.i
print gd1.eigenValues
print gd1.eigenVectors
# print gd1.eigenValues.get_value()
# print gd1.eigenVectors.get_value()


# In[ ]:

# w_np,v_np = np.linalg.eigh( np.dot(X_data.T,X_data) )
print X_data
w_np,v_np = np.linalg.eigh(np.dot(X_data.T,X_data))
idx = np.argsort(-w_np)
w_np = w_np[idx]
v_np = v_np[:,idx]
# for i in range(DIMENSIONS):
#     print w_np[i]
#     print v_np[:,i]
print w_np.astype('float32')
print v_np.astype('float32')


# In[ ]:

# gd=gd1


# In[ ]:

# temp = gd.eigenVectors.eval()
# temp[:,1] = 0
# temp
# gd.eigenVectors.set_value(temp)


# In[ ]:

# print gd.v.eval()
# print gd.eigenVectors.eval()
# print np.sum(gd.eigenVectors[:,j]* gd.v for j in range(gd.i)).eval()
# print gd.i
# print
# print (gd.eigenVectors*gd.v).eval()
# print ((gd.eigenVectors*gd.v)).eval()


# In[ ]:

# print (T.sum(gd.eigenVectors*gd.v)).eval()
# print (T.sum(gd.eigenVectors*gd.v, axis=0)).eval()
# print (T.sum(gd.eigenVectors*gd.v, axis=1)).eval()


# In[ ]:

# print T.dot(gd.eigenVectors[1], gd.v).eval()
# print gd.v.eval()
# print gd.eigenVectors[1].eval()
# print (T.dot(gd.eigenVectors[1], gd.v)**2).eval()
# print (gd.eigenValues[1]*(T.dot(gd.eigenVectors[1], gd.v)**2)).eval()
# print ((T.dot(gd.eigenVectors[1], gd.v))).eval()
# print (T.dot(gd.eigenVectors,gd.v)).eval()
# print ((T.dot(gd.eigenVectors[1], gd.v)**2)).eval()
# print (T.dot(gd.eigenVectors,gd.v)**2).eval()
# print (gd.eigenValues[1]*(T.dot(gd.eigenVectors[1], gd.v)**2)).eval()
# print (gd.eigenValues*(T.dot(gd.eigenVectors,gd.v)**2)).eval()
# print T.dot(gd.eigenValues,(T.dot(gd.eigenVectors,gd.v)**2)).eval()
# print (gd.eigenValues(T.dot(gd.eigenVectors[1], gd.v)**2)).eval()
# # T.sum?


# In[ ]:




# In[ ]:




# ---------------------------------------------------------------------------

# In[ ]:




# In[ ]:




# In[ ]:



