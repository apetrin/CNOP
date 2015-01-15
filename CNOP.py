from __future__ import division
import statsmodels as st
import statsmodels.api as sm
from scipy import stats
import numpy as np 
from statsmodels.sandbox.distributions.extras import mvstdnormcdf
from scipy.optimize import minimize
from itertools import izip
from scipy.optimize import minimize

FLOAT_EPS = np.finfo(float).eps

class CNOP(st.discrete.discrete_model.DiscreteModel):
    """Correlatesdf BLA BLA BLA
    
    Description:
    TO BE WRITTEN"""

    def __init__(self,endog, exog, **kwargs):
        #endog is a dict of endog.vars
        self.x, self.zplus, self.zminus = map(endog.get, ['x', 'zplus', 'zminus'])
        self.y = exog
        self.interest_step = kwargs.get('interest_step', 0.00125)
        if "J" in kwargs:
            self.J = int(kwargs.get("J"))
        else:        
            self.J = int(max(self.y.abs().max().T["Y"]) / self.interest_step)
        try:
            self.param_len = int(2*self.J +4 + len(self.x.minor_axis) + 
                                len(self.zminus.minor_axis) + len(self.zplus.minor_axis))
            self.alpha_len, self.beta_len = 2, len(self.x.minor_axis)
            self.mum_len, self.gammam_len = self.J, len(self.zminus.minor_axis)
            self.mup_len, self.gammap_len = self.J, len(self.zplus.minor_axis)
            self.rhom, self.rhop = 1, 1
        except AttributeError:
            pass

    informcode = {0: 'normal completion with ERROR < EPS',
                  1: '''completion with ERROR > EPS and MAXPTS function values used;
                        increase MAXPTS to decrease ERROR;''',
                  2: 'N > 500 or N < 1'}

    def cdf(self, X, Y=None, rho=None):
        """ Returns 1D or 2D standard normal CDF"""
        if Y is None: return stats.norm._cdf(X)
        #else: return mvstdnormcdf([-np.inf, -np.inf], 
        #                                [X, Y], rho)
        error,value,inform = stats.mvn.mvndst([-np.inf, -np.inf], [X,Y], [0,0], rho)
        if inform:
            print('something wrong', self.informcode[inform], error)
        return value

    def tester(self):
        x = [14.1,18.2,12.1,23.1,-12,123]
        J=4
        print(self.cdf(0,0,1))
        test = self.cons_generator([(0,4)])
        cons = test
        #print cons[1]
        #print self.J

    #def jac(x, i):
    #    jac = np.zeros(len(x))
    #    jac[i+1], jac[i]  = -1, 1
    #    return jac

    def cons_generator(self, slice, type = "ineq"):
        """
        Function generates a string of constrants,
        required for optimization routine in scipy.optimize.

        INPUT: list of slices, e.g. [(2,4),(7,12),(14,19)]
        constrants are set in between slices
        
        Great thanks to Ilya Shurov for improving this code
        """
        constr = []
        for st, fin in slice:
            for i in range(st,fin-1):
                constr.append({"type":type,"fun":lambda x, i=i:np.array([float(x[i+1]-x[i])])})
        return constr

    def loglike(self, params):
        """
        Log-likelihood of CNOP model.
        params -- [alpha, beta, mu-, gamma-, mu+, gamma+, rho+, rho-]
        len(alpha) = 2
        len(beta) = len(self.x.minor_axis)   #-2 ### USING PANDAS
        len(mu-) = J
        len(gamma-) = len(self.zminus.minor_axis)   #-2
        len(mu+) = J
        len(gamma+) = len(self.zplus.minor_axis)   #-2
        len(rho+) = 1
        len(rho-) = 1
        I use     def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs): algorithm

        np.dot(pan[1].loc[3.1998],params)
        """
        params = list(params)
        alpha, beta = params[:2], params[2:2+len(self.x.minor_axis)]
        del params[:2+len(self.x.minor_axis)]
        mum, gammam =  params[:self.J], params[self.J:self.J+len(self.zminus.minor_axis)]
        del params[:self.J+len(self.zminus.minor_axis)]
        mup, gammap = params[:self.J], params[self.J:self.J+len(self.zplus.minor_axis)]
        del params[:self.J+len(self.zplus.minor_axis)]
        rhop, rhom = params
        del params[:2]
        assert len(params) is 0, "params isn't empty!"

        s = 0
        y, x = self.y, self.x
        zm, zp = self.zminus, self.zplus
        for (yitem, ydf), (xitem, xdf),   (zmitem, zmdf), (zpitem, zpdf) in \
        izip(y.iteritems(), x.iteritems(), zm.iteritems(), zp.iteritems() ): 
            for (ytime, yelement), (xtime, xelement), (zmtime, zmelement), (zptime, zpelement) in \
            izip(ydf.iterrows(), xdf.iterrows(), zmdf.iterrows(), zpdf.iterrows() ): 
                #Two Sums Here,
                #xelement, zmelement, zpelement are Series of interest for item xitem
                # and for time xitem. Items and times are identical throughout theree Panels
                assert xitem == zmitem, "Items doesn't match: xitem != zmitem"
                assert zmitem == zpitem, "Items doesn't match: zpitem != zmitem"
                assert xtime == zmtime, "Times doesn't match: xitem != zmitem"
                assert zmtime == zptime, "Times doesn't match: zpitem != zmitem"

                j = int(round(yelement["Y"],10) / self.interest_step )
                if j == 0:
                    pr = self.cdf(alpha[1] - np.dot(xelement, beta)) - self.cdf(alpha[0] - np.dot(xelement, beta))
                if j >= 0:
                    pr =  self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-1]-np.dot(zpelement, gammap), -rhop)
                    pr -= self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-2]-np.dot(zpelement, gammap), -rhop)
                if j <= 0:
                    pr =  self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-1]-np.dot(zmelement, gammam), rhom)
                    pr -= self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-2]-np.dot(zmelement, gammam), rhom)
                s += np.log(np.clip(pr, FLOAT_EPS, 1))
        return s 

    def fit(self, start_params=None, method='COBYLA', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """method are COBYLA and SLSQP [DEPRECIATED ADD JAC!]. ___Subject to check, COBYLA is better on simple tasks"""
        if start_params is None: start_params = np.zeros(self.param_len)
        constraints = self.cons_generator([(self.alpha_len + self.beta_len, self.alpha_len + self.beta_len + self.J),
                                      (self.alpha_len + self.beta_len + self.J + self.gammam_len,
                                       self.alpha_len + self.beta_len + self.J + self.gammam_len + self.J)
                                      ])
        constraints = []
        return minimize(lambda x:-self.loglike(x), x0=start_params, method=method, constraints=constraints,
                        options = {'maxiter':maxiter, 'disp':disp}, callback=callback
                        )



#TEST = CNOP({'x':[12,13], 'zplus':[14,15], 'zminus': [17,17], 'J':2},[12.15, 12.30])
#TEST.tester()


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings
from scipy.optimize import minimize, check_grad
from pandas.tools.plotting import scatter_matrix

np.set_printoptions(precision = 3, suppress = True)
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier


df = pd.read_csv(u"simul_data_CNOP.csv",sep=';').dropna()
l = dict(zip(df["NO"].unique(),map(lambda x: df[df['NO']==1], df["NO"].unique())))
pan = pd.Panel(l)
pan.major_axis = pan[1]["MONTH"]


###### SUBSAMPLING: 32 is the starting sample, single observation only
y = pan.ix[32:,:,['Y']]
x = pan.ix[32:,:,['X1','X2','X3',u'X4Z1', u'X5Z3', u'X6Z4']]
zplus = pan.ix[32:,:,[u'X4Z1', u'X5Z3', u'X6Z4', u'Z2', u'Z5', u'Z6', u'Z7',]]
zminus = pan.ix[32:,:,[u'X4Z1', u'X5Z3', u'X6Z4', u'Z2', u'Z5', u'Z6', u'Z7',]]
endog = dict( (name,eval(name)) for name in ['x','zplus','zminus'] )

CNOP2 = CNOP(endog, y)
print CNOP2.fit()


