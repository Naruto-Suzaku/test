import numpy as np

class I_E(object):

    def __init__(self,v,I_E_form,beta_1=0.7,v_1=1.):
        self.I_E_form = I_E_form
        self.v = v
        self.beta_1 = beta_1
        self.v_1 = v_1

    def I_E(self):

        if self.I_E_form == 'no_seiffert':
            result = 10**(7.66-2.79*np.log10(self.v))

        if self.I_E_form == 'seiffert':
            #fitted the seiffert data excess componant           
            #result = 18.4 *(self.v*1e-3/0.31)**-2.57
            #print result
            #fitted the seiffert data total discerete source componant
            result = 24.4*(self.v*1e-3/0.31)**-2.58
            print result 
        if self.I_E_form == 'Dowell':

            #fitted the Dowell(2018) data larger than 408MHz
            result = 25.38505431*(self.v*1e-3/0.31)**-2.55834165
            print result
            #fitted the Dowell(2018) data less than 408MHz
            result = 15.73203052*(self.v*1e-3/0.408)**-2.55153783
            print result
        if self.I_E_form == 'extra_freq_dependence':
            #result = 15.18657467 * (self.v*1e-3/0.31)**(-2.55142127 + 0.10045386*np.log10(self.v*1e-3/0.31))
            #result = 23.9255305 * (self.v*1e-3/0.31)**(-2.33967063 -  0.22369317*np.log10(self.v*1e-3/0.31))
            #fitted the Dowell(2018)  data less than 408MHz
            result = 12.5137071 * (self.v*1e-3/0.408)**(-3.11491251 - 0.47679418*np.log10(self.v*1e-3/0.408))
        
        if self.I_E_form == 'seiffert_freq_depend':
            #suppose
            beta_1 = self.beta_1
            v_1 = self.v_1
            #beta0 = -2.57
            beta0 = -2.58
            beta = beta0 + beta_1*np.exp(-self.v/v_1)
            #result = 18.4 *(self.v*1e-3/0.31)**beta
            result = 24.4*(self.v*1e-3/0.31)**beta

        return result
if __name__ == '__main__':
    f = I_E(v = 1, I_E_form = 'seiffert')
    f.I_E()
