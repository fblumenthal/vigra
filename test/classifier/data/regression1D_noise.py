import numpy as np
from vigra import superfeatures as sf 
from tools import writeH5
#
# Generate a simple 1D regression task 
#


rf=sf.RandomRegressionForest(10)

x=np.random.randn(100)*2*np.pi
y=0.3*x*np.sin(x)

x=x.reshape(-1,1)
y=y.reshape(-1,1)

rf.learnRF(x,y,seed=42)


xx=np.linspace(0,2*np.pi,500).reshape(-1,1)

yy=rf.predict(xx)


writeH5(filename="regression1D_noise.h5",path="features_test",data=xx.swapaxes(0,1),delete=True)
writeH5(filename="regression1D_noise.h5",path="features_train",data=x.swapaxes(0,1))
writeH5(filename="regression1D_noise.h5",path="labels_train",data=y.swapaxes(0,1))
writeH5(filename="regression1D_noise.h5",path="reference_prediction",data=yy.swapaxes(0,1))
writeH5(filename="regression1D_noise.h5",path="seed",data=np.array([42]))


rf2=sf.RandomRegressionForest(10)
rf2.learnRF(x,y,seed=42)
yy2=rf2.predict(xx)

np.testing.assert_array_equal(yy,yy2)




yt=.3*xx*np.sin(xx)
import pylab
# pylab.plot(xx,yy)
# pylab.plot(xx,yt)

# pylab.show()