#!/usr/bin/env python

import sys
sys.path.append('/home/pos/WEAVE/python/')
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class Zmap():
    def __init__(self,filename):
        self.zmapdata = np.loadtxt(filename)
        self.gridx,self.gridy=np.mgrid[-200.:200.:5,-200.:200.:5]
        self.XX = self.gridx.flatten()
        self.YY = self.gridy.flatten()

    def fit(self):
        self.A=np.c_[np.ones(self.zmapdata.shape[0]),self.zmapdata[:,:2],np.prod(self.zmapdata[:,:2],axis=1),self.zmapdata[:,:2]**2]
        self.C,_,_,_ = linalg.lstsq(self.A,self.zmapdata[:,2])
        return

    def plot(self):
        self.Z=np.dot(np.c_[np.ones(self.XX.shape),self.XX,self.YY,self.XX*self.YY,self.XX**2,self.YY**2], self.C).reshape(self.gridx.shape)
        plt.imshow(self.Z)
        plt.show()
        return

    def zvalue(self,x,y):
        zval = np.dot(np.c_[np.ones(np.size(x)),x,y,x*y,x**2,y**2],self.C)
        return zval
