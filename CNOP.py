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
        self.interest_step = endog.get('interest_step', 0.000125)
        if "J" in endog:
            self.J = endog.get("J")
        else:        
            self.J = max(self.x.abs().max().T["Y"]) / self.interest_step
        try:
            self.param_len = int(2*self.J - 2 + len(self.x.minor_axis) + 
                                len(self.zminus.minor_axis) + len(self.zplus.minor_axis))
            self.alpha_len, self.beta_len = 2, len(self.x.minor_axis)-2
            self.mum_len, self.gammam_len = self.J, len(self.zminus.minor_axis)-2
            self.mup_len, self.gammap_len = self.J, len(self.zplus.minor_axis)-2
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
        print self.cdf(0,0,1)
        test = self.cons_generator([(0,4)])
        cons = eval(test)
        #print cons[1]
        #print self.J

    #def jac(x, i):
    #    jac = np.zeros(len(x))
    #    jac[i+1], jac[i]  = -1, 1
    #    return jac

    def cons_generator(self, slice, type = "ineq"):
        """#indian
        Function generates a string of constrants,
        required for optimization routine in scipy.optimize.

        You only need to eval() the output.

        INPUT: list of slices, e.g. [(2,4),(7,12),(14,19)]
        constrants are set in between slices
        """
        constrsstr = "["
        for st, fin in slice:
            for i in range(st,fin-1):
                #constrsstr += '{"type":"' +type + '","fun":lambda x:np.array([float(x[' + \
                #        str(i+1) + ']-x[' + str(i) + '])]),' +\
                #        '"jac":lambda x:jac(x, '+str(i)+')},'
                constrsstr += '{"type":"%s","fun":lambda x:np.array([float(x[%s]-x[%s])])},' \
                                %(type, str(i+1), str(i))

        constrsstr += "]"
        return constrsstr

    def loglike(self, params):
        """
        Log-likelihood of CNOP model.
        params -- [alpha, beta, mu-, gamma-, mu+, gamma+, rho+, rho-]
        len(alpha) = 2
        len(beta) = len(self.x.minor_axis)-2 ### USING PANDAS
        len(mu-) = J
        len(gamma-) = len(self.zminus.minor_axis)-2
        len(mu+) = J
        len(gamma+) = len(self.zplus.minor_axis)-2
        len(rho+) = 1
        len(rho-) = 1
        I use     def fit(self, start_params=None, method='newton', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs): algorithm
        
        np.dot(pan[1].loc[3.1998],params)
        """
        params = list(params)
        alpha, beta = params[:2], params[2:2+len(self.x.minor_axis)-2]
        del params[:2+len(self.x.minor_axis)-2]
        mum, gammam =  params[:self.J], params[self.J:self.J+len(self.zminus.minor_axis)-2]
        del params[:self.J+len(self.zminus.minor_axis)-2]
        mup, gammap = params[:self.J], params[self.J:self.J+len(self.zplus.minor_axis)-2]
        del params[:self.J+len(self.zplus.minor_axis)-2]
        rhop, rhom = params
        del params[:2]
        assert len(params) is 0, "params isn't empty!"

        s = 0
        x = self.x
        zm = self.zm
        zp = self.zp
        j = int(round(xelement["Y"],10) / interest_step )
        for (xitem, xdf),   (zmitem, zmdf), (zpitem, zpdf) in \
        izip(x.iteritems(), zm.iteritems(), zp.iteritems() ): 
            for (xtime, xelement), (zmtime, zmelement), (zptime, zpelement) in \
            izip(xdf.iterrows(),   zmdf.iterrows(),     zpdf.iterrows() ): 
                #Two Sums Here,
                #xelement, zmelement, zpelement are Series of interest for item xitem
                # and for time xitem. Items and times are identical throughout theree Panels
                assert xitem == zmitem, "Items doesn't match: xitem != zmitem"
                assert zmitem == zpitem, "Items doesn't match: zpitem != zmitem"
                assert xtime == zmtime, "Times doesn't match: xitem != zmitem"
                assert zmtime == zptime, "Times doesn't match: zpitem != zmitem"

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
        constraints = eval(self.cons_generator([(self.alpha_len + self.beta_len, self.alpha_len + self.beta_len + self.J),
                                      (self.alpha_len + self.beta_len + self.J + self.gammam_len,
                                       self.alpha_len + self.beta_len + self.J + self.gammam_len + self.J)
                                      ]))
        return minimize(self.loglike, x0=start_params, method=method, constraints=constraints,
                        options = {'maxiter':maxiter, 'disp':disp}, callback=callback
                        )



TEST = CNOP({'x':[12,13], 'zplus':[14,15], 'zminus': [17,17], 'J':2},[12.15, 12.30])
TEST.tester()