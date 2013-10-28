import numpy as np
from vigra import superfeatures as sf 
from tools import writeH5
#
# Generate a dipole vector field for the regression task 
#
from numpy import sqrt ,exp
np.random.seed(33)

x=np.linspace(-5,5,100)
y=np.linspace(-5,5,100)

vx=-y/sqrt(x**2+y**2)*exp(-(x**2+y**2)) + np.random.randn(100)*0.1
vy= x/sqrt(x**2+y**2)*exp(-(x**2+y**2))+ np.random.randn(100)*.1

feat_train=np.vstack((x,y)).T
label_train=np.vstack((vx,vy)).T

rf=sf.RandomRegressionForest(10)


xx=np.linspace(-5,5,1000)
yy=np.linspace(-5,5,1000)

vxx=-yy/sqrt(xx**2+yy**2)*exp(-(xx**2+yy**2))
vyy= xx/sqrt(xx**2+yy**2)*exp(-(xx**2+yy**2))

feat_test=np.vstack((xx,yy)).T
label_test=np.vstack((vxx,vyy)).T


rf.learnRF(feat_train,label_train,seed=42)
pred=rf.predict(feat_test)

vxx_pred=pred.T[0]
vyy_pred=pred.T[1]


writeH5(filename="regressionND_noise.h5",path="features_train",data=feat_train.swapaxes(0,1),delete=True)

writeH5(filename="regressionND_noise.h5",path="labels_train",data=label_train.swapaxes(0,1))

writeH5(filename="regressionND_noise.h5",path="features_test",data=feat_test.swapaxes(0,1))

writeH5(filename="regressionND_noise.h5",path="reference_prediction",data=pred.swapaxes(0,1))
writeH5(filename="regressionND_noise.h5",path="seed",data=np.array([42]))


import pylab
pylab.plot(xx,vxx)
pylab.plot(xx,vxx_pred)

pylab.show()