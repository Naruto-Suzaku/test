import h5py
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from SSM import SSM
from pylab import cm
from scipy.optimize import curve_fit
import sys
sys.path.append('../')
#from Constant_Index.improved_leastsq_method import value_of_index
import sys
#sys.path.append("..")
#from Index.produce_index import produce_index
from I_E_term.I_E_equation import I_E

class produce_index(object):
    
    def __init__(self, Nside,freq,index_type,ssmpath, I_E_form, catalog = False, smin = None, input_freq = [10,22,45,85,150,408], beta_1 = 0.7, v_1 = 1.):
        self.nside = Nside
        self.freq = freq
        self.catalog = catalog
        self.smin = smin
        self.index_type = index_type
        self.ssmpath = ssmpath
        self.input_freq = input_freq
        self.I_E_form = I_E_form
        self.beta_1 = beta_1
        self.v_1 = v_1
        #self.index_type = index_type
    def SSM_(self, freq):
        diffuse, source, catalog = SSM(freq = freq, nside = self.nside, catalog = self.catalog, Smin = self.smin, name = "useless.hdf5",inSSMpath = self.ssmpath) 
        return diffuse, source, catalog
    
    def read_file(self):
        dict = {}
        for freq in self.input_freq:
            diffuse, source, catalog = self.SSM_(freq)
            dict.update({np.float(freq):diffuse})
        print 'dict',dict
        return dict


        #f = h5py.File('../init_input/10_22_45_85_150MHz.hdf5','r')
        #g = h5py.File('../init_input/diffuse_408MHz.hdf5','r')
        #diffuse_10 = f['diffuse_10'][:]
        #diffuse_22 = f['diffuse_22'][:]
        #diffuse_45 = f['diffuse_45'][:]
        #diffuse_85 = f['diffuse_85'][:]
        #diffuse_150 = f['diffuse_150'][:]
        #diffuse_408 = g['diffuse_408'][:]
        #
        ##print ('raw_difffuse',diffuse_408)
        ##plt.figure(222)
        ##hp.mollview(np.log10(diffuse_408),cmap = plt.cm.jet)
        ##plt.title('raw_diffuse408' + ' mean=' + str(np.mean(diffuse_408)))
        ##plt.show()
        #
        #f.close()
        #g.close()
        #return {10:diffuse_10,45:diffuse_45,85:diffuse_85,150:diffuse_150,408:diffuse_408}

    def diff_index(self):
        #for item in read_file.item():
        #index : array like [array[].shape = 12*nside**2,...].shape = len(self.input_freq) -1, the 408MHz as the based frequency.
        index = []
        data = self.read_file()
        for (k1,v1) in  data.items():
            for (k_base,v_base) in data.items():
                if (k_base == float(408)) and (k1 != k_base):
                    print ('k1,k_base',k1,k_base)
                    print (np.where(v1==0)[0],np.where(v_base==0)[0],np.where(k1 == 0),np.where(k_base==0))
                    alpha = np.log10(v1/v_base) / np.log10(k1/k_base)
                    index.append(alpha)
        print 'index', index
        print 'len(index)', len(index)
        print 'index[0].shape', index[0].shape
        return index

    def _45_408_induce_index(self):
        data = self.read_file()
        v1 = np.float(45)
        v_base = np.float(408)
        k1 = data[v1]
        k_base = data[v_base]
        alpha = np.log10(k1/k_base)/np.log10(v1/v_base) 
        return alpha

    def plot_index(self):
        #index = self.mean_index()
        index = self._45_408_induce_index()
        plt.figure(2)
        hp.mollview(index, cmap = plt.cm.jet)
        #plt.savefig('fitted_index.eps', format = 'eps')
        plt.savefig('45_408_induce_index.eps', format = 'eps')
        return

    def plot_fixed_pixel_index(self):
        index = self.diff_index()
        import random
        n = random.randint(0,index[0].shape[0]-1)
        Y = []
        for i in index:
            Y.append(i[n])
        plt.figure(3)
        plt.plot(np.arange(len(Y)),Y)
        plt.show()
        return

    def mean_index(self):
        index = self.diff_index()
        index = np.array(index)
        fitted_index = []
        def func_(x, d):
            return d
        for i in range(index.shape[1]):
            Y = index[:,i]
            X = np.arange(Y.size)
            popt, pcov = curve_fit(func_, X, Y)
            fitted_index.append(popt[0])

        print '(X,Y)', (X,Y)
        fitted_index = np.array(fitted_index)
        print 'fitted_index and fitted_index.shape',fitted_index,fitted_index.shape
        #void = np.zeros_like(index[0])
        #for i in index:
        #    void += i
        #void = void / len(index)
        with h5py.File('Nside_'+str(self.nside)+'fitted_index.hdf5', 'w') as f:
            f.create_dataset('fitted_index',data = fitted_index)
        return fitted_index

    #def constant_index(self):
    #    beta = -2.4697003
    #    index_ = np.array(12*self.nside**2 * [beta])
    #    return index_  
    def constant_index_minus_I_E(self):
        #beta = -2.47900839
        #f = value_of_index(I_E_form = self.I_E_form)
        #beta = f.Para_constant_Para_freq_dependence()[0]
        beta = -2.46
        beta = float(beta)
        
        print 'beta',beta
        #('Para_constant', array([2.47900839]), 'Para_0', array([2.73479401, 0.25710172])) 
        index_ = np.array(12*self.nside**2 * [beta])
        return index_

    def freq_dependence_index_minus_I_E(self,freq):
        #beta_0, beta_1 = np.array([2.73479401, 0.25710172])
        #f = value_of_index(I_E_form = self.I_E_form)
        #beta0 = f.Para_constant_Para_freq_dependence()[0]

        #beta0 = -2.48044886
        #beta_1 = self.beta_1
        #v_1 = self.v_1
        #beta = beta0 + beta_1 * np.exp(-freq/v_1)
        beta_1 = self.beta_1;v_1 = self.v_1
        #beta0 = f.Para_constant_Para_freq_dependence()[0]
        beta0 = -2.46
        beta0 = float(beta0)
        beta = beta0 + beta_1 * np.exp(-freq/v_1)
        print 'beta0,beta_1,v_1,freq',beta0,beta_1,v_1,freq
        print 'beta dependence beta:',beta
        #beta_0, beta_1 = f.Para_constant_Para_freq_dependence()[1]
        #beta_0 = float(beta_0)
        #beta_1 = float(beta_1)
        #print 'beta_0',beta_0
        #print 'beta_1',beta_1
        #beta = -(beta_0 + beta_1 * np.log10(self.freq/408.))
        index_ = beta
        return index_

    def diffuse_x(self, freq):
        #if self.index_type == 'constant_index':
        #    index = self.constant_index()

        if self.index_type == 'constant_index_minus_I_E':
            index = self.constant_index_minus_I_E()
        if self.index_type == 'freq_dependence_index_minus_I_E':
            index = self.freq_dependence_index_minus_I_E(freq)

        #if self.index_type == 'pixel_dependence':
         
            #freq is in Mhz
            #try:
            #    with h5py.File('Nside_'+str(freq)+'fitted_index.hdf5', 'r') as g:
            #        index = f['fitted_index'][:]
            #except:
            #    index = self.mean_index()

        
        data_freq = 408.
        #the based data of 408MHz coming from SSM model output
        #data_diffuse, source, catalog = self.SSM_(data_freq)
        #the based data of 408MHz coming from HS14 data set
        data_diffuse = hp.read_map('/public/home/wufq/congyanping/Software/LFSAM_new/Index/input_HS14_408MHz/haslam408_dsds_Remazeilles2014.fits')
        data_diffuse = hp.ud_grade(data_diffuse, self.nside) 
        print 'min(data_diffuse',min(data_diffuse)
        data_diffuse = data_diffuse - I_E(data_freq, self.I_E_form).I_E()
        #for freq  in [1,2,3,4,5,10,16,32,100,408]: 
        diffuse_x = np.multiply(data_diffuse, (freq/data_freq)**index)
        
        with h5py.File('Index_'+'diffuse_'+str(freq)+'MHz.hdf5', 'w') as f:
            f.create_dataset('freq', data = freq)
            f.create_dataset('nside',data = self.nside)
            f.create_dataset('diffuse_x', data = diffuse_x)
            f.create_dataset('diffuse_408',data = data_diffuse)
            f.create_dataset('index',data = index)
        print ('np.isnan(diffuse_x)',np.where(np.isnan(diffuse_x))) 
        return diffuse_x

    def save_data(self):
        
        index = self.mean_index()
        
        diffuse = self.read_file()
        
        data = list(diffuse.items())[-1]
        data_freq = data[0]
        data_diffuse = data[1]
        

        diffuse_1 = np.multiply(data_diffuse ,(1./data_freq)**index)
        
        diffuse_2 = np.multiply(data_diffuse ,(2./data_freq)**index)
        
        diffuse_3 = np.multiply(data_diffuse ,(3./data_freq)**index)
        
        diffuse_4 = np.multiply(data_diffuse ,(4./data_freq)**index)
        
        diffuse_5 = np.multiply(data_diffuse ,(5./data_freq)**index)
        diffuse_6 = np.multiply(data_diffuse ,(6./data_freq)**index)
        diffuse_7 = np.multiply(data_diffuse ,(7./data_freq)**index)
        diffuse_8 = np.multiply(data_diffuse ,(8./data_freq)**index)
        diffuse_9 = np.multiply(data_diffuse ,(9./data_freq)**index)
        
        diffuse_10 = np.multiply(data_diffuse,(10./data_freq)**index)
        
        diffuse_20 = np.multiply(data_diffuse, (20./data_freq)**index)
        
        diffuse_30 = np.multiply(data_diffuse, (30./data_freq)**index)
        
        diffuse_40 = np.multiply(data_diffuse, (40./data_freq)**index)
        diffuse_50 = np.multiply(data_diffuse, (50./data_freq)**index)
        diffuse_60 = np.multiply(data_diffuse, (60./data_freq)**index)
        diffuse_70 = np.multiply(data_diffuse, (70./data_freq)**index)
        diffuse_80 = np.multiply(data_diffuse, (80./data_freq)**index)
        diffuse_90 = np.multiply(data_diffuse, (90./data_freq)**index)
        diffuse_100 = np.multiply(data_diffuse, (100./data_freq)**index)
        
        diffuse_408 = np.multiply(data_diffuse, (408./data_freq)**index)
        
        print ('diffuse_408',diffuse_408)
        plt.figure(1)
        hp.mollview(np.log10(diffuse_1),cmap = plt.cm.jet)
        plt.show()
        
        plt.figure(408)
        hp.mollview(np.log10(diffuse_100),cmap = plt.cm.jet)
        
        plt.title(str(np.mean(diffuse_100)))
        plt.show()
        
        with h5py.File('diffuse_data_with_diff_index.hdf5', 'w') as f:
            f.create_dataset('diffuse_1',data = diffuse_1)
            f.create_dataset('diffuse_2',data = diffuse_2)
            f.create_dataset('diffuse_3',data = diffuse_3)
            f.create_dataset('diffuse_4',data = diffuse_4)
            f.create_dataset('diffuse_5',data = diffuse_5)
            f.create_dataset('diffuse_6',data = diffuse_6)
            f.create_dataset('diffuse_7',data = diffuse_7)
            f.create_dataset('diffuse_8',data = diffuse_8)
            f.create_dataset('diffuse_9',data = diffuse_9)

            f.create_dataset('diffuse_10',data = diffuse_10)
            f.create_dataset('diffuse_20',data = diffuse_20)
            f.create_dataset('diffuse_30',data = diffuse_30)
            f.create_dataset('diffuse_40',data = diffuse_40)
            f.create_dataset('diffuse_50',data = diffuse_50)
            f.create_dataset('diffuse_60',data = diffuse_60)
            f.create_dataset('diffuse_70',data = diffuse_70)
            f.create_dataset('diffuse_80',data = diffuse_80)
            f.create_dataset('diffuse_90',data = diffuse_90)
            f.create_dataset('diffuse_100',data = diffuse_100)
            f.create_dataset('diffuse_408',data = diffuse_408)
            
        return 0

    def plot_mean_index(self):
        if self.index_type == 'constant_index':
            mean_index = self.constant_index()

        if self.index_type == 'constant_index_minus_I_E':
            mean_index = self.constant_index_minus_I_E()

        if self.index_type == 'freq_dependence_index_minus_I_E':
            mean_index = self.freq_dependence_index_minus_I_E
         
        if self.index_type == 'pixel_dependence':
            mean_index = self.mean_index()
        
        cmap = cm.jet
        cmap = cmap.set_under('w')

        plt. figure('mean_index', figsize = (8,6))
        hp.mollview(mean_index, cmap = cmap)
        plt.title('')
        plt.savefig(self.index_type + '_index.eps',format = 'eps')

        plt.figure(5,figsize=(8,6))
        plt.hist(mean_index, 100, edgecolor = 'black')
        plt.tick_params(direction = 'in',top = True)
        plt.tick_params(direction = 'in',right = True)
        #plt.title('Mean:'+ str(np.round(np.mean(mean_index),2)) + '   Std: '+ str(np.round(np.std(mean_index),2)))
        plt.title('')
        plt.savefig('statistic_of_index.eps',format = 'eps')
        #plt.show()
        return 0
if __name__=='__main__':
    f = produce_index(512,1,'constant_index_minus_I_E',ssmpath='/public/home/wufq/congyanping/Software/SSM/inSSM.hdf5', I_E_form='seiffert') 
    f.SSM_(408)
    #diffuse_1 = f.diffuse_x(1)
    #diffuse_2 = f.diffuse_x(2)
    #result = diffuse_1 / diffuse_2
    #plt.figure(1,figsize=(16,9))
    #plt.plot(np.arange(result.size),result,'o')
    #plt.savefig('constant.png')
