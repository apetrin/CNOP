from __future__ import division
import statsmodels as st
import statsmodels.api as sm
from scipy import stats
import numpy as np 
from statsmodels.sandbox.distributions.extras import mvstdnormcdf
from itertools import izip
from scipy.optimize import minimize, fmin_slsqp
import math

FLOAT_EPS = np.finfo(float).eps

class OrderedProbit(st.discrete.discrete_model.OrderedModel):
    """Ordered Probit
    Bla-Bla-Bla
    """
    def _ordered_recode(self, endog):
        uniques = sorted(set(endog))
        return [uniques.index(i) for i in endog]

    def __init__(self, endog, exog, offset=None, exposure=None, missing='none',
                 **kwargs):
        endog = self._ordered_recode(endog)
        super(OrderedProbit, self).__init__(endog, exog, missing=missing,
                                         offset=offset,
                                         exposure=exposure, **kwargs)
        if exposure is not None:
            self.exposure = np.log(self.exposure)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

    def cdf(self, X):
        """
        Probit (Normal) cumulative distribution function
        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).
        Returns
        --------
        cdf : ndarray
            The cdf evaluated at `X`.
        Notes
        -----
        This function is just an alias for scipy.stats.norm.cdf
        """
        return stats.norm._cdf(X)

    def pdf(self, X):
        """
        Probit (Normal) probability density function
        Parameters
        ----------
        X : array-like
            The linear predictor of the model (XB).
        Returns
        --------
        pdf : ndarray
            The value of the normal density function for each point of X.
        Notes
        -----
        This function is just an alias for scipy.stats.norm.pdf
        """
        X = np.asarray(X)
        return stats.norm._pdf(X)

    def cons_generator(self, slice, dict_out=True):
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
                if dict_out:
                    constr.append({"type":"ineq","fun":lambda x, i=i:np.array([float(x[i+1]-x[i])])})
                    lambda x, i=i: np.zeros(len(x))
                else:
                    constr.append(lambda x, i=i:np.array([x[i+1]-x[i]]))
        return constr

    def cons_fprime(self, x, slice):
        lenf = slice[0][1]-1-slice[0][0]
        fprime=np.zeros(shape=(lenf,len(x)))
        for i in range(lenf):
            fprime[i,-i-1], fprime[i,-i-2] = 1, -1
        return fprime[::-1]

    def loglike(self, params):
        """
        Log-likelihood of ordered probit model
        
        params=(beta, cutoffs)
        """
        #X = self.exog
        #np.sum(np.log(np.clip(self.cdf(np.dot(X,params[0])),
        #    FLOAT_EPS, 1)))
        params = [params[:len(self.exog[0])], params[len(self.exog[0]):]]
        #print params
        #print " ".join([str(len(params[0])),str(len(params[1]))])
        #print params[0]
        #print params[1]
        s=0
        for X, Y in izip(self.exog, self.endog):
            if Y == 0:
                s+= np.log(np.clip(self.cdf(params[1][Y] - np.dot(X,params[0])), FLOAT_EPS, 1))
            elif Y != 0 and Y != max(self.endog):
                s+= np.log(np.clip(self.cdf(params[1][Y] - np.dot(X,params[0])) 
                                   - self.cdf(params[1][Y-1] - np.dot(X,params[0])), FLOAT_EPS, 1))
            elif Y == max(self.endog):
                s+= np.log(np.clip(1-self.cdf(params[1][Y-1] - np.dot(X,params[0])), FLOAT_EPS, 1))
        return s
        #return np.sum(np.log(np.clip(self.cdf(params[1][Y] - np.dot(X,params[0])),
        #    FLOAT_EPS, 1)) for X,Y in izip(self.exog, self.endog) if Y == 0) +\
        #    np.sum(np.log(np.clip(self.cdf(params[1][Y] - np.dot(X,params[0])) - self.cdf(params[1][Y-1] - np.dot(X,params[0])),
        #    FLOAT_EPS, 1)) for X,Y in izip(self.exog, self.endog) if Y != 0 and Y != max(self.endog)) +\
        #    np.sum(np.log(np.clip(1-self.cdf(params[1][Y-1] - np.dot(X,params[0])),
        #    FLOAT_EPS, 1)) for X,Y in izip(self.exog, self.endog) if Y == max(self.endog))

    def fit(self, start_params=None, method='COBYLA', maxiter=35,
            full_output=1, disp=1, callback=None, fun = "minimize", iprint=None, **kwargs):
        if start_params is None: 
            start_params = list(np.zeros(len(self.exog[0]))) + range(max(self.endog))
        constraints = self.cons_generator([(len(self.exog[0]),len(self.exog[0]) + max(self.endog))], dict_out=False)
        if fun == "minimize":
            return minimize(fun = lambda x:-self.loglike(x), x0=start_params, method=method, constraints=constraints,
                        options = {'maxiter':maxiter, 'disp':disp}, callback=callback
                        )
        elif fun == "fmin_slsqp":
            return fmin_slsqp(lambda x:-self.loglike(x), fprime=lambda x:-self.score(x), x0=start_params, 
                              disp=disp, callback=callback, full_output=full_output, iprint=iprint,
                              ieqcons = constraints, iter=maxiter, fprime_ieqcons =self.cons_fprime)
        #discretefit = ProbitResults(self, bnryfit)
        #return BinaryResultsWrapper(discretefit)

    def score(self, params):
        beta, mu = params[:len(self.exog[0])], params[len(self.exog[0]):]
        s=np.zeros(len(params))
        for X, Y in izip(self.exog, self.endog):
            if Y == 0:
                mXb = mu[Y] - np.dot(X, beta)
                s1 = self.pdf(mXb) / self.cdf(mXb) * np.append(np.append(-X,[1]), np.zeros(len(mu)-1))
                s += s1
                #s+= np.log(np.clip(self.cdf(mu[Y] - np.dot(X,beta)), FLOAT_EPS, 1))
            elif Y != 0 and Y != max(self.endog):
                mXb0 = mu[Y]   - np.dot(X, beta)
                mXb1 = mu[Y-1] - np.dot(X, beta)
                s1 = (self.pdf(mXb0) - self.pdf(mXb1)) * (-X)
                mus = np.zeros(len(mu))
                mus[Y], mus[Y-1]  = self.pdf(mXb0), -self.pdf(mXb1)
                s1 = np.append(s1, mus)
                s1 = s1/(self.cdf(mXb0) - self.cdf(mXb1)) 
                s += s1
                #s+= np.log(np.clip(self.cdf(mu[Y] - np.dot(X,beta)) 
                #                   - self.cdf(mu[Y-1] - np.dot(X,beta)), FLOAT_EPS, 1))
            elif Y == max(self.endog):
                mXb = mu[Y-1] - np.dot(X, beta)
                s1 = -self.pdf(mXb) / (1-self.cdf(mXb)) * np.append(np.append(-X,np.zeros(len(mu)-1)), [1])
                s += s1
                #s+= np.log(np.clip(1-self.cdf(mu[Y-1] - np.dot(X,beta)), FLOAT_EPS, 1))
        return s







import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import warnings
from scipy.optimize import minimize, check_grad
from pandas.tools.plotting import scatter_matrix

np.set_printoptions(precision = 3, suppress = True)
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier



df = pd.read_csv(u"simul_data_CNOP.csv",sep=';').dropna()

dat = df[:1000].copy()
del dat['MONTH'], dat['NO'], dat['Y']
exog = dat
endog = df[:1000].copy()["Y"]


#actuall params from EViews for this model
xstart = [0.014145, 0.058282, 0.327108,-0.257436,0.025568,-0.056417,0.339505,0.027607,-0.108454,-0.009823]+\
        [-1.096773,-0.923243,-0.738524,-0.522930,-0.356648,-0.168086,-0.013571,0.131471,0.236204,0.371262,
         1.852668, 1.904029,1.975700,2.078341,2.162041,2.266225,2.390665,2.544107,2.787838,3.026606]
#IF specify the exact parameters of xstart, then converges in ~35 steps
#NOT to the exact xstart, but close 

OP = OrderedProbit(endog, exog)
#print OP.loglike(xstart)
x = OP.fit(fun='fmin_slsqp', maxiter=100, iprint=3, start_params=np.zeros(len(xstart)))
print x
#print OP.score(xstart)