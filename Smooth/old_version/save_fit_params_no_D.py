#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import sin,cos,pi
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import scipy.constants as C
import healpy as hp
import h5py
import scipy.optimize as optimize
from scipy.integrate import quad

#from matplotlib import cm
from pylab import cm
#from mpi4py import MPI
import time
print np.zeros(5)

from how_to_smooth_SSM import smooth


class free_free(object):
    def __init__(self, v, nside):
        self.v = v#20Mhz frequency in hz
        self.nside = nside       

    def plot_mollview(self,total,filename = ''):
        cool_cmap = cm.jet
        cool_cmap.set_under("w") # sets background to white
        plt.figure(1)
        if True:
            m = np.log10(total)
            Min = None
            Max = None
        hp.mollview(m,title="The frequency in"+' '+str(self.v)+'MHz', min = Min, max = Max,cmap = cool_cmap)
        plt.savefig(str(self.v)+'MHz'+ filename +'.eps',format = 'eps')

    
    
    def produce_xyz(self):
        #v in Mhz
        result = smooth(self.nside,self.v).add_5()
        return result

    def diffuse_raw(self):
        diffuse_x = smooth(self.nside, self.v).produce_data()
        return diffuse_x
 
    def fun_for_curvefit(self, xyz, R_0, alpha, a, B, c, g, form = 'n_HII'):
        #produce for input for curve_fit
        result = []
        for l,b in xyz:
            self.l = l * np.pi / 180.0
            self.b = b * np.pi / 180.0

            def fun_inte(r):
                x = r * np.sin(np.pi/2. - self.b) * np.cos(self.l)
                y = r * np.sin(np.pi/2. - self.b) * np.sin(self.l)
                z = r * np.cos(np.pi/2. - self.b)

                x_1 = x - 8.5
                y_1 = y
                z_1 = z

                r_1 = np.sqrt(np.square(x_1) + np.square(y_1) + np.square(z_1))
                b_1 = np.pi/2.0 - np.arccos(z_1/r_1)
                l_1 = np.arctan(y_1/x_1)

                R = r_1
                Z = r_1 * np.sin(b_1)
                #integrate along the sight direction
                #return h * np.square(a * np.exp(-np.abs(Z)/b - np.square(R/c)) + d * np.exp(-np.abs(Z)/e - np.square(R/f - g)))
                #return a * (R/b)**c * np.exp(-d*(R-e/e) - np.abs(Z)/f)
                #integrate along sight direction
                #return e * np.square(a * np.exp(-R/b) * np.square(2/(np.exp((Z-d)/c)+ np.exp(-(Z-d)/c))))
                #ne = f * np.exp(-np.abs(d)/1e3 - (r_1/(2e4*A +1))**2) + g * np.exp(-np.abs(d)/(0.15*1e3) - (r_1/(2e3*B+1) - 2)**2)
                #ne = f * np.exp(-np.abs(d)/(1e3*B+1) - (r_1/(2e4*A +1))**2) 
                #ne = f * np.exp(-np.abs(d)/(1e3*B+1) - (r_1/(2e4*A +1))**4) 
                #ne = np.square(a * np.exp(-R/b) * np.square(2.0/(np.exp(Z/(1e3*c+1))+ np.exp(-Z/(1e3*c+1)))))
            	#ne = a * np.exp(-np.abs(d) * 2/(B+0.1) - 2*(r_1/(20*c + 0.1))**2) + D * np.exp(-np.abs(d)*2/(e * 0.15 + 0.01) - 2*(r_1/(2*f+0.1))**2)
            	#ne = a * np.exp(-np.abs(d) * 2/(B+0.1) - 2*(r_1/(20*c + 0.1))**2) + D
            	ne = (R/(R_0+0.1))**alpha * a * np.exp(-np.abs(Z) * 2/(B+0.1) - 2*(r_1/(20*c + 0.1))**2) 
                #get rid of square 
		return g * ne 

            result.append(quad(fun_inte, 0, 20)[0])
        return np.array(result)

    def curve_fit(self,form='CRs'):
        if form == 'CRs':
            #R_0, alpha, a,B,c,D,g
            guess = [1,1,1,2,3,5]

        func = self.fun_for_curvefit
        xyz = self.produce_xyz()
        print 'xyz.shape',xyz.shape
        params, pcov = optimize.curve_fit(func, xyz[:,:2], xyz[:,2], guess, bounds=(0,1e9), method='trf')
        
        with h5py.File(str(self.v)+'Mhz_fitted_param.hdf5','w') as f:
            f.create_dataset('params',data = params)
            f.create_dataset('v',data = self.v)
        print 'frequency',self.v
        print 'params',params
        print 'pcov',pcov
        return params

    def model_m2(self,l,b,abcz0,form='two_componant'):

        self.l = l * np.pi / 180.0
        self.b = b * np.pi /180.0
        R_0, alpha, a, B, c, g = abcz0

        def fun_inte(r):

            #integrate along the sight direction
            x = r * np.sin(np.pi/2. - self.b) * np.cos(self.l)
            y = r * np.sin(np.pi/2. - self.b) * np.sin(self.l)
            z = r * np.cos(np.pi/2. - self.b)

            x_1 = x - 8.5
            y_1 = y
            z_1 = z

            r_1 = np.sqrt(np.square(x_1) + np.square(y_1) + np.square(z_1))
            b_1 = np.pi/2.0 - np.arccos(z_1/r_1)
            l_1 = np.arctan(y_1/x_1)

            R = r_1
            Z = r_1 * np.sin(b_1)
            ne = (R/(R_0+0.1))**alpha * a * np.exp(-np.abs(Z) * 2/(B+0.1) - 2*(r_1/(20*c + 0.1))**2)
            #get rid of square 
	    return g * ne 

        return quad(fun_inte, 0, 20)[0]
 
    def delta_m(self):

        abcz0 = self.curve_fit()
        
        nside = self.nside
        m = np.zeros(hp.nside2npix(nside))
        for pix_number in range(0,hp.nside2npix(nside)):
            l,b = hp.pixelfunc.pix2ang(nside, pix_number, nest = False, lonlat = True)
            pix_value = self.model_m2(l,b,abcz0)
            m[pix_number] = pix_value
        self.plot_mollview(m,filename= 'integrated_temperature')

       
        delt_m = self.diffuse_raw() - m 
        
	self.plot_mollview(delt_m,filename = 'delt_m')
        
        with h5py.File(str(self.v)+'Mhz' + '_delt_m.hdf5','w') as f:
            f.create_dataset('delt_m',data = delt_m)
          
            f.create_dataset('integrated_temperature_total_m', data = m)
        #back to delt_m and params     
        return delt_m, abcz0
 

#if __name__=='__main__':    
#    #modify
#    for v in [1]:
#        nside = 2**4
#        f = free_free(v = v, nside = nside)
#        f.total_m()
#        print 'end'
#    
#






