from scipy.optimize import leastsq
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from SSM import SSM
import h5py

import sys
sys.path.append("..")
#from Index.produce_index import produce_index
from I_E_term.I_E_equation import I_E

class value_of_index(object):

    def __init__(self,I_E_form = 'no_seiffert'):
        self.I_E_form = I_E_form

    def read_file(self):
        #freq = np.array([0.022, 0.045, 0.408, 1.42, 3.20, 3.41, 7.98, 8.33, 9.72, 10.49, 29.5, 31, 90, 250])
        #T_obs = np.array([13268, 2843, 10.80, 3.181, 2.777, 2.761, 2.761, 2.742, 2.730, 2.738, 2.529, 2.573, 2.706, 2.725]) - 2.725
        #error = np.array([5229, 512, 3.53, 0.526, 0.01, 0.008, 0.013,0.015,0.005,0.006, 0.155, 0.076, 0.019, 0.001])
	#freq = np.array([0.408,1.419,3.150,3.410,7.970,8.330,9.720,10.490])
        #T_obs = np.array([15.2,3.276,2.788,2.768,2.764,2.741,2.731,2.731]) - 2.725
        #error = np.array([2.37,0.167,0.045,0.045,0.060,0.062,0.062,0.065])
        freq = np.array([0.022,0.040,0.046,0.050,0.060,0.070,0.080,0.408],dtype=np.float64)
        T_obs = np.array([19212,5792,4090,3443,2363,1505,1188,15.20],dtype=np.float64) - 2.725
        error = np.array([4095,963,691,526,365,208,112,2.37],dtype=np.float64)
        
        return freq,T_obs,error


    def I_E(self,v):
        
        f = I_E(v,self.I_E_form)
        result = f.I_E() 
        print 'I_E_result',result
        return result
    
    def func(self,p0, freq):
        #T_0, beta_0, beta_1 = p0
        T_0, beta_0 = p0
        #result = T_0 * (freq/0.31)**(beta_0 + beta_1*np.log10(freq/0.31))
        result = T_0 * (freq/0.408)**(beta_0)
        return result
    
    def error(self, p0, freq,T_obs,error):
        result = (self.func(p0,freq) - T_obs)/(error)
        return result

    def func2(self,p0, freq):
        T_0, beta_0, beta_1 = p0
        result = T_0 * (freq/0.408)**(beta_0 +  beta_1*np.log10(freq/0.408))
        return result
    
    def error2(self, p0, freq,T_obs,error):
        result = (self.func2(p0,freq) - T_obs)/(error)
        return result

    def Plot(self):

        freq,T_obs,error = self.read_file()
        #beta = np.array([18,1,1])
        beta = np.array([1,1],dtype=np.float64)

        Para_constant=leastsq(self.error,beta,args=(freq,T_obs,error))
        print 'T_0, beta_0',Para_constant[0] 

        beta = np.array([1,1,1])
        Para_0=leastsq(self.error2,beta,args=(freq,T_obs,error))
        print 'T_0, beta_0, beta_1',Para_0[0]
        
        plt.figure(2)
        X = np.arange(1*1e-3,408*1e-3,0.001)
        Y = Para_0[0][0]*(X/0.408)** ((Para_0[0][1]+Para_0[0][2]*np.log10(X/0.408)))
        YY = Para_constant[0][0]*(X/0.408)**(Para_constant[0][1])
        plt.plot(X,Y,label='beta_0+beta1*log10(v/0.408.)')
        plt.plot(X,YY,label='beta')
        plt.legend(loc='best')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('frequency(GHz)')
        plt.ylabel('brightness temperature(K)')
        plt.title('extragalactic diffused emission')
        plt.savefig('./output/'+'temperatue_1_to_408_new.eps',format='eps')
        plt.savefig('./output/'+'temperatue_1_to_408_new.jpg',ppi = 300)

        return Para_0[0]

    
if __name__ == '__main__':
    f = value_of_index()

    f.Plot() 
    #f.index_with_freq()               
     
     
