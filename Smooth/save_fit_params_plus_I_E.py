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

    def __init__(self, v, nside, index_type, dist, emi_form,test = False):
        self.v = v#20Mhz frequency in hz
        self.nside = nside       
        self.index_type = index_type
        self.dist = dist
        self.emi_form = emi_form
        self.test = test

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
        if self.test == True:
            result,diffuse_x = smooth(self.nside, self.v, self.index_type).galprop_simulate_data()
        else:

            result = smooth(self.nside,self.v, self.index_type).add_5()
        return result

    def diffuse_raw(self):
        diffuse_x = smooth(self.nside, self.v, self.index_type).produce_data()
        return diffuse_x

    def sech2(self,x):
        return np.square(2/(np.exp(x) + np.exp(-x)))
 
    def fun_for_curvefit(self, xyz, A_v, R_0, alpha, R_1, beta, Z_0, gamma, I_E, form = 'n_HII'):
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
                if self.emi_form == 'exp':

                    emissivity = A_v * (R/R_0)**alpha * np.exp(-(R/R_1)**beta) * np.exp(-(np.abs(Z)/Z_0)**gamma)
                if self.emi_form == 'sech2':
                    
                    emissivity = A_v * (R/R_0)**alpha * np.exp(-(R/R_1)**beta) * self.sech2(-(np.abs(Z)/Z_0)**gamma)
                #get rid of square 
		return emissivity

            result.append(quad(fun_inte, 0, self.dist)[0] + I_E)
        return np.array(result)

    def curve_fit(self,form='CRs'):
        if form == 'CRs':
            #R_0, alpha, a,B,c,D,g
            #A_v, R_0, alpha, R_1, beta, Z_0, gamma,I_E
            estimate_left = (24.4-1.7)*(self.v * 1e-3/0.31)**(-2.58-0.03)
            estimate_right = (24.4+1.7)*(self.v * 1e-3/0.31)**(-2.58+0.03)
            guess = [1e7,8.5,2,4,2,3,1,(estimate_left+estimate_right)/2.0]
            #guess = [1e7,8.5,2,4,2,3,1,1]
        print ('guess value',guess)
        #print ('estimate_left',(estimate_left)*0.5,'estimate_right',(estimate_right)*3)
        func = self.fun_for_curvefit
        xyz = self.produce_xyz()
        print ('estimate_left', 1, 'estimate_right',np.min(xyz[:,2]))
        print 'xyz.shape',xyz.shape
        #params, pcov = optimize.curve_fit(func, xyz[:,:2], xyz[:,2], guess, bounds=(np.array([0,1e-5,-3.1,1e-5,-3.1,1e-5,-3.1,(estimate_left)*0.5]),np.array([1e10,100,3.1,100,3.1,20,3.1,(estimate_right+1e7)*3])), method='trf')'
        upper_limit = np.max(xyz[:,2])
        params, pcov = optimize.curve_fit(func, xyz[:,:2], xyz[:,2], guess, bounds=(np.array([0,1e-5,-3.1,1e-5,-3.1,1e-5,-3.1,1]),np.array([1e10,100,3.1,100,3.1,20,3.1,upper_limit])), method='trf')
        
        with h5py.File(str(self.v)+'Mhz_fitted_param.hdf5','w') as f:
            f.create_dataset('params',data = params)
            f.create_dataset('v',data = self.v)
            f.create_dataset('pcov', data = pcov)
        print 'frequency',self.v
        print 'params',params
        print 'pcov',pcov
        return params

    def model_m2(self,l,b,abcz0,form='two_componant'):

        self.l = l * np.pi / 180.0
        self.b = b * np.pi /180.0
        A_v, R_0, alpha, R_1, beta, Z_0, gamma, I_E = abcz0

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
            emissivity = A_v * (R/R_0)**alpha * np.exp(-(R/R_1)**beta) * np.exp(-(np.abs(Z)/Z_0)**gamma)
            #get rid of square
            if self.emi_form == 'exp':

                emissivity = A_v * (R/R_0)**alpha * np.exp(-(R/R_1)**beta) * np.exp(-(np.abs(Z)/Z_0)**gamma)
            if self.emi_form == 'sech2':
                    
                emissivity = A_v * (R/R_0)**alpha * np.exp(-(R/R_1)**beta) * self.sech2(-(np.abs(Z)/Z_0)**gamma)
  
	    return emissivity 

        return quad(fun_inte, 0, self.dist)[0] + I_E
 
    def delta_m(self):
        try:
            with h5py.File(str(self.v)+'Mhz_fitted_param.hdf5','r') as f:
                abcz0 = f['params'][:]
        except:

            abcz0 = self.curve_fit()
        
        nside = self.nside
        m = np.zeros(hp.nside2npix(nside))
        for pix_number in range(0,hp.nside2npix(nside)):
            l,b = hp.pixelfunc.pix2ang(nside, pix_number, nest = False, lonlat = True)
            pix_value = self.model_m2(l,b,abcz0)
            m[pix_number] = pix_value
        self.plot_mollview(m,filename= 'integrated_temperature')
        #diffuse_raw from galprop data directely
        if self.test == True:
            result,diffuse_raw = smooth(self.nside, self.v, self.index_type).galprop_simulate_data()
        else:

            diffuse_raw = self.diffuse_raw() 
        delt_m = diffuse_raw - m 
        
        delt_m_percentage = delt_m / diffuse_raw * 100 
	self.plot_mollview(delt_m,filename = 'delt_m')
	self.plot_mollview(delt_m_percentage,filename = 'delt_m_percentage')
        
        with h5py.File('./output/'+str(self.emi_form)+str(self.v)+'Mhz' + '_delt_m_and_unabsorb_and_delt_m_percentage.hdf5','w') as f:
            f.create_dataset('delt_m',data = delt_m)
            f.create_dataset('delt_m_percentage',data = delt_m_percentage)
            f.create_dataset('integrated_temperature_total_m', data = m)
            f.create_dataset('diffuse_raw', data = diffuse_raw) 
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






