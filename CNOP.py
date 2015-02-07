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

class llist(list):
    """Special list for Ordered Models. Element -1 is -Inf, Final element is Inf.

    --------------------------------------------------------
    EXAMPLE:
    x = llist([12.,14,2.2])
    x[1]
    >>>14
    x[-1], x[3]
    >>>(-inf, inf)
    x[-2]
    >>>IndexError: Not supported index -2 was called.
    """

    def __getitem__(self, l):
        if   l == -1:
            return -np.inf
        elif l == len(self):
            return np.inf
        elif l < -1 or l > len(self):
            raise IndexError, "Not supported index %i was called."%l
        else:
            #help(super(list, self))
            return super(llist,self).__getitem__(l)

    def __setitem__(self, key, value):
        if not (key == -1 or key == len(self)):
            return super(llist,self).__setitem__(key, value)

class CNOP(st.discrete.discrete_model.DiscreteModel):
    """Cross-Nested BLA BLA BLA
    
    Description:
    TO BE WRITTEN"""

    @staticmethod
    def _ordered_recode(endog):
        #Recode data to [0,.....,N]
        #NOT YET INTEGRATED
        uniques = sorted(set(endog))
        return [uniques.index(i) for i in endog]

    def __init__(self,endog, exog, **kwargs):
        #endog is a dict of endog.vars
        self.x, self.zplus, self.zminus = map(endog.get, ['x', 'zplus', 'zminus'])
        #self.y = self._ordered_recode(exog) - kwargs.get('infl_y', 0)    #NOT YET INTEGRATED
        self.y = exog
        self.interest_step = kwargs.get('interest_step', 0.00125)
        self.model = kwargs.get('model', 'CNOP')
        if "J" in kwargs:
            self.J = int(kwargs.get("J"))
        else:        
            self.J = int(max(self.y.abs().max().T["Y"]) / self.interest_step)
        try:
            #self.param_len = int(2*self.J +4 + len(self.x.minor_axis) + 
            #                    len(self.zminus.minor_axis) + len(self.zplus.minor_axis))
            self.alpha_len, self.beta_len = 2, len(self.x.minor_axis)
            self.mum_len, self.gammam_len = self.J, len(self.zminus.minor_axis)
            self.mup_len, self.gammap_len = self.J, len(self.zplus.minor_axis)
            if self.model=="CNOPc":
                self.rhom, self.rhop = 1, 1 
            else:
                self.rhom, self.rhop = None, None
        except AttributeError:
            pass
        self.param_len=sum(filter(None, 
            [self.alpha_len,self.beta_len,self.mum_len,self.gammam_len,self.mup_len,self.gammap_len,self.rhom,self.rhop]))

    def __len__(self):
        """Returns number of observations"""
        return len(self.y[0])

    def observations_generator(self):
        """Generator that returns next pooled observation"""
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

                yield yelement, xelement, zmelement, zpelement


    informcode = {0: 'normal completion with ERROR < EPS',
                  1: '''completion with ERROR > EPS and MAXPTS function values used;
                        increase MAXPTS to decrease ERROR;''',
                  2: 'N > 500 or N < 1'}

    @staticmethod
    def cdf(X, Y=None, rho=None):
        """ Returns 1D or 2D standard normal CDF"""
        if Y is None: 
            return stats.norm._cdf(X)
        error,value,inform = stats.mvn.mvndst([-np.inf, -np.inf], [X,Y], [0,0], rho)
        if inform:
            print('something wrong', self.informcode[inform], error)
        return value

    @staticmethod
    def pdf(X):
        """
        This function is just an alias for scipy.stats.norm.pdf
        """
        X = np.asarray(X)
        return stats.norm._pdf(X)

    def tester(self):
        x = [14.1,18.2,12.1,23.1,-12,123]
        J=4
        print(self.cdf(0,0,1))
        test = self.cons_generator([(0,4)])
        cons = test
        #print cons[1]
        #print self.J

    @staticmethod
    def jac(x, i):
        jac = np.zeros(len(x))
        jac[i+1], jac[i]  = 1, -1
        return jac

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
                constr.append({"type":type,
                               "fun":lambda x, i=i:np.array([float(x[i+1]-x[i])]),
                               "jac":lambda x, i=i:self.jac(x=x, i=i)
                               })
        if self.model == "CNOPc": #CNOP MODEL CODE
            constr.append({"type":type,"fun":lambda x:np.array([float(x[-1]+1)])})
            constr.append({"type":type,"fun":lambda x:np.array([float(1-x[-1])])})
            constr.append({"type":type,"fun":lambda x:np.array([float(x[-2]+1)])})
            constr.append({"type":type,"fun":lambda x:np.array([float(1-x[-2])])})
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
        #print(params)
        #print("")
        alpha, beta = params[:2], params[2:2+len(self.x.minor_axis)]
        del params[:2+len(self.x.minor_axis)]
        mum, gammam =  llist(params[:self.J]), params[self.J:self.J+len(self.zminus.minor_axis)]
        del params[:self.J+len(self.zminus.minor_axis)]
        mup, gammap = llist(params[:self.J]), params[self.J:self.J+len(self.zplus.minor_axis)]
        del params[:self.J+len(self.zplus.minor_axis)]
        if self.model == "CNOPc":
            rhop, rhom = params
            del params[:2]
        assert len(params) is 0, "params isn't empty!"

        s = 0.0
        y, x = self.y, self.x
        zm, zp = self.zminus, self.zplus
        
        for yelement, xelement, zmelement, zpelement in self.observations_generator():
            j = int(round(yelement["Y"],10) / self.interest_step )
            pr=0
            if self.model == "CNOPc": #CNOPc MODEL CODE
                #THIS CODE HAS AN ERROR! Border J should be seen independently!
                assert False, "CNOPc!!! We should not be here!"
                if j == 0:
                    pr += self.cdf(alpha[1] - np.dot(xelement, beta)) - self.cdf(alpha[0] - np.dot(xelement, beta))
                if j >= 0:
                    pr +=  self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-1]-np.dot(zpelement, gammap), -rhop)
                    pr -= self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-2]-np.dot(zpelement, gammap), -rhop)
                if j <= 0:
                    pr +=  self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-1]-np.dot(zmelement, gammam), rhom)
                    pr -= self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-2]-np.dot(zmelement, gammam), rhom)
            else: #CNOP MODEL CODE
                if j<0:
                    pr +=  (self.cdf(alpha[0]-np.dot(xelement, beta))) * \
                            (self.cdf(mum[self.J+j]-np.dot(zmelement,gammam))-self.cdf(mum[self.J+j-1]-np.dot(zmelement,gammam)))
                elif j==0:
                    a = self.cdf(alpha[1]-np.dot(xelement, beta)) 
                    b = self.cdf(alpha[0]-np.dot(xelement, beta)) 
                    c = self.cdf(mup[0]-np.dot(zpelement, gammap)) 
                    d = self.cdf(mum[self.J-1]-np.dot(zmelement, gammam)) 
                    pr += a + c - (a*c + b*d)
                elif j>0:
                    pr += (1-self.cdf(alpha[1]-np.dot(xelement, beta))) * \
                            (self.cdf(mup[j]-np.dot(zpelement,gammap))-self.cdf(mup[j-1]-np.dot(zpelement,gammap)))
                else:
                    raise ValueError, "j = %i not incorrectly defined" %j
            s += np.log(np.clip(pr, FLOAT_EPS, 1))
        return s

    def score(self, params):
        """Score function for loglike"""
        params = list(params)
        alpha, beta = params[:2], params[2:2+len(self.x.minor_axis)]
        del params[:2+len(self.x.minor_axis)]
        mum, gammam =  llist(params[:self.J]), params[self.J:self.J+len(self.zminus.minor_axis)]
        del params[:self.J+len(self.zminus.minor_axis)]
        mup, gammap = llist(params[:self.J]), params[self.J:self.J+len(self.zplus.minor_axis)]
        del params[:self.J+len(self.zplus.minor_axis)]
        if self.model == "CNOPc":
            rhop, rhom = params
            del params[:2]
        assert len(params) is 0, "params isn't empty!"
        
        score = np.zeros(self.param_len)
        for yelement, xelement, zmelement, zpelement in self.observations_generator():
            j = int(round(yelement["Y"],10) / self.interest_step )
            pr = 0.
            #print j
            #print score
            if self.model == "CNOP": #CNOP MODEL CODE
                if j<0:
                    pr =  (self.cdf(alpha[0]-np.dot(xelement, beta))) * \
                            (self.cdf(mum[self.J+j]-np.dot(zmelement,gammam))-self.cdf(mum[self.J+j-1]-np.dot(zmelement,gammam)))
                    if pr == 0: 
                        #print "YEIKS < 0"
                        pr=FLOAT_EPS
                    #PARTIAL DERIVATIVES BELOW ARE DENOTED BY P (prime)
                    alphap = [self.pdf(alpha[0]-np.dot(xelement, beta)) *
                                 (self.cdf(mum[self.J+j]-np.dot(zmelement,gammam))-self.cdf(mum[self.J+j-1]-np.dot(zmelement,gammam)))
                              ,0.]
                    betap = -xelement * self.pdf(alpha[0]-np.dot(xelement, beta)) * \
                                 (self.cdf(mum[self.J+j]-np.dot(zmelement,gammam))-self.cdf(mum[self.J+j-1]-np.dot(zmelement,gammam)))
                    mump = np.zeros(self.mum_len)
                    mump[self.J+j]   =  self.cdf(alpha[0]-np.dot(xelement, beta)) * self.pdf(mum[self.J+j]-np.dot(zmelement,gammam))
                    mump[self.J+j-1] = -self.cdf(alpha[0]-np.dot(xelement, beta)) * self.pdf(mum[self.J+j-1]-np.dot(zmelement,gammam))
                    gammamp = -zmelement * self.cdf(alpha[0]-np.dot(xelement, beta)) *\
                                 (self.pdf(mum[self.J+j]-np.dot(zmelement,gammam))-self.pdf(mum[self.J+j-1]-np.dot(zmelement,gammam)))
                    mupp = np.zeros(self.mup_len)
                    gammapp = np.zeros(self.gammap_len)
                    #print len(alphap),len(betap),len(mump),len(gammamp),len(mupp),len(gammapp)
                    score_local = np.concatenate((alphap,betap,mump,gammamp,mupp,gammapp)) / pr
                    score += score_local
                elif j==0:
                    a = self.cdf(alpha[1]-np.dot(xelement, beta)) 
                    b = self.cdf(alpha[0]-np.dot(xelement, beta)) 
                    c = self.cdf(mup[0]-np.dot(zpelement, gammap)) 
                    d = self.cdf(mum[self.J-1]-np.dot(zmelement, gammam)) 
                    pr = a + c - (a*c + b*d)
                    if pr == 0: 
                        #print "YEIKS = 0"
                        pr=FLOAT_EPS
                    # Xpdf denotes PDF for X
                    apdf = self.pdf(alpha[1]-np.dot(xelement, beta)) 
                    bpdf = self.pdf(alpha[0]-np.dot(xelement, beta)) 
                    cpdf = self.pdf(mup[0]-np.dot(zpelement, gammap)) 
                    dpdf = self.pdf(mum[self.J-1]-np.dot(zmelement, gammam)) 
                    #Manually computed
                    alphap = [-bpdf * d , apdf * (1.-c)]
                    betap = -xelement * (apdf*(1.-c)-bpdf*d) #???
                    mump = [0.0]*(self.mum_len-1) + [-b*dpdf]
                    gammamp = zmelement * b * dpdf
                    mupp = [(1.-a)*cpdf] + [0.0]*(self.mup_len-1)
                    gammapp = zpelement * (a-1.) * cpdf
                    score_local = np.concatenate((alphap,betap,mump,gammamp,mupp,gammapp)) / pr
                    #print "ZERO"
                    #print len(alphap),len(betap),len(mump),len(gammamp),len(mupp),len(gammapp)
                    #print score_local
                    score += score_local
                elif j>0:
                    pr = (1-self.cdf(alpha[1]-np.dot(xelement, beta))) * \
                            (self.cdf(mup[j]-np.dot(zpelement,gammap))-self.cdf(mup[j-1]-np.dot(zpelement,gammap)))
                    if pr == 0: 
                        #print "YEIKS < 0"
                        pr=FLOAT_EPS
                    #PARTIAL DERIVATIVES BELOW ARE DENOTED BY P (prime)
                    alphap = [0,
                              -self.pdf(alpha[1]-np.dot(xelement, beta)) *
                                 (self.cdf(mup[j]-np.dot(zpelement,gammap))-self.cdf(mup[j-1]-np.dot(zpelement,gammap)))
                              ]
                    betap = xelement * self.pdf(alpha[1]-np.dot(xelement, beta)) * \
                                 (self.cdf(mup[j]-np.dot(zpelement,gammap))-self.cdf(mup[j-1]-np.dot(zpelement,gammap)))
                    mump = np.zeros(self.mum_len)
                    gammamp = np.zeros(self.gammap_len)
                    mupp = llist(np.zeros(self.mup_len))
                    mupp[j]   =  (1-self.cdf(alpha[1]-np.dot(xelement, beta))) * self.pdf(mup[ j ]-np.dot(zpelement,gammap))
                    mupp[j-1] = -(1-self.cdf(alpha[1]-np.dot(xelement, beta))) * self.pdf(mup[j-1]-np.dot(zpelement,gammap))
                    gammapp = -zpelement * (1-self.cdf(alpha[1]-np.dot(xelement, beta))) *\
                                 (self.pdf(mup[j]-np.dot(zpelement,gammap))-self.pdf(mup[j-1]-np.dot(zpelement,gammap))) #???
                    score_local = np.concatenate((alphap,betap,mump,gammamp,mupp,gammapp)) / pr
                    score += score_local
                else:
                    raise ValueError, "j = %i not incorrectly defined" %j
            #print j
            #print score
        return score        

    def fit(self, start_params=None, method='SLSQP', maxiter=35,
            full_output=1, disp=1, callback=None, **kwargs):
        """method are COBYLA and SLSQP [DEPRECIATED ADD JAC!]. ___Subject to check, COBYLA is better on simple tasks"""
        if start_params is None: start_params = np.zeros(self.param_len)
        constraints = self.cons_generator([(0,2),
                                      (self.alpha_len + self.beta_len, self.alpha_len + self.beta_len + self.J),
                                      (self.alpha_len + self.beta_len + self.J + self.gammam_len,
                                       self.alpha_len + self.beta_len + self.J + self.gammam_len + self.J)
                                      ])
        #constraints = []
        return minimize(fun = lambda x:-self.loglike(x), x0=start_params, method=method, constraints=constraints,
                        options = {'maxiter':maxiter, 'disp':disp}, callback=callback, jac =lambda x:-self.score(x)
                        )

    def se(self,params):
        """
        Return standard errors at optimum point, NOT SE
        """
        hess = self.hessian(params)
        return np.sqrt(np.linalg.inv(-hess).diagonal())

    def hessian (self, x0, epsilon=1.e-5, linear_approx=False,  *args ):
        """
        A numerical approximation to the Hessian matrix of arbitrary function f at
        location x0 (hopefully, the minimum)

        AUTHOR: jgomezdans , https://gist.github.com/jgomezdans
        """
        # ``f`` is the cost function
        f = self.loglike
        # The next line calculates the first derivative
        f1 = self.score(x0)

        # This is a linear approximation. Obviously much more efficient
        # if cost function is linear
        if linear_approx:
            f1 = np.matrix(f1)
            return f1.transpose() * f1    
        # Allocate space for the hessian
        n = x0.shape[0]
        hessian = np.zeros ( ( n, n ) )
        # The next loop fill in the matrix
        xx = x0
        for j in xrange( n ):
            xx0 = xx[j] # Store old value
            xx[j] = xx0 + epsilon # Perturb with finite difference
            # Recalculate the partial derivatives for this new point
            f2 = self.score(x0)
            hessian[:, j] = (f2 - f1)/epsilon # scale...
            xx[j] = xx0 # Restore initial value of x0        
        return hessian



#TEST = CNOP({'x':[12,13], 'zplus':[14,15], 'zminus': [17,17], 'J':2},[12.15, 12.30])
#TEST.tester()

if __name__=="__main__":
 
    import numpy as np
    import pandas as pd
    import matplotlib.pylab as plt
    import warnings
    from scipy.optimize import minimize, check_grad
    from pandas.tools.plotting import scatter_matrix



###############################################################################
###################### OLD CODE BELOW #########################################
###############################################################################
#np.set_printoptions(precision = 3, suppress = True)
#pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier


#df = pd.read_csv(u"simul_data_CNOP.csv",sep=';').dropna()
#l = dict(zip(df["NO"].unique(),map(lambda x: df[df['NO']==1], df["NO"].unique())))
#pan = pd.Panel(l)
#pan.major_axis = pan[1]["MONTH"]


###### SUBSAMPLING: 32 is the starting sample, single observation only
#y = pan.ix[32:,:,['Y']]
#x = pan.ix[32:,:,['X1','X2','X3',u'X4Z1', u'X5Z3', u'X6Z4']]
#zplus = pan.ix[32:,:,[u'X4Z1', u'X5Z3', u'X6Z4', u'Z2', u'Z5', u'Z6', u'Z7',]]
#zminus = pan.ix[32:,:,[u'X4Z1', u'X5Z3', u'X6Z4', u'Z2', u'Z5', u'Z6', u'Z7',]]
#endog = dict( (name,eval(name)) for name in ['x','zplus','zminus'] )

#CNOP2 = CNOP(endog, y)
#print CNOP2.fit()
###############################################################################
###################### OLD CODE ABOVE #########################################
###############################################################################


    df3 = pd.read_csv(u"cnop_MC_dat_short.tsv", sep="\t").dropna()
    exog = df3[["X1", "X2", "X3"]]
    endog = df3[["Y"]]-3

    l = {0:df3}
    pan = pd.Panel(l)
    y = pan.ix[:,:,['Y']]-3
    x = pan.ix[:,:,['X1','X2']]
    zminus = pan.ix[:,:,['X1','X3']]
    zplus = pan.ix[:,:,['X2','X3']]
    endog = dict( (name,eval(name)) for name in ['x','zplus','zminus'] )
    CNOP3 = CNOP(endog, y, model='CNOP',interest_step=1, J=2, disp=False)

    x_real_3 = [0.7681,1.2221, 0.5084,0.3067, # alpha0 alpha0 and beta
            -0.6585,0.4256,0.2621,0.2779, #zminus
            0.1102,1.3007,0.2866,0.9772, #zplus
            ]

    #print("GAUSS RESULTS: mean loglike -1.09000")
    #print "Python RESULTS: mean loglike",CNOP3.loglike(x_real_3)/len(CNOP3)
    #print CNOP3.y
    #print "Score:", CNOP3.score(x_real_3)
    #print CNOP3.x[0].columns

    ########################################################
    ### Starting Values Generation  ########################
    ########################################################

    df3_firststage1 = df3.copy()
    def func(x):
        if x>3: return 1
        if x<3: return -1
        if x==3: return 0
    df3_firststage1['Y'] = df3_firststage1['Y'].apply(func)
    exog_firststage1 = df3_firststage1[["X1", "X2"]]
    endog_firststage1 = df3_firststage1["Y"]-3
    df3_firststage2 = df3[df3["Y"]<=3]
    exog_firststage2 = df3_firststage2[["X1", "X3"]]
    endog_firststage2 = df3_firststage2["Y"]-3
    df3_firststage3 = df3[df3["Y"]>=3]
    exog_firststage3 = df3_firststage3[["X2", "X3"]]
    endog_firststage3 = df3_firststage3["Y"]-3

    from OP import OrderedProbit
    OP_firststage1 = OrderedProbit(endog_firststage1, exog_firststage1)
    OP_firststage2 = OrderedProbit(endog_firststage2, exog_firststage2)
    OP_firststage3 = OrderedProbit(endog_firststage3, exog_firststage3)

    x_firststage1=OP_firststage1.fit(method = "SLSQP", maxiter=200, disp=False)
    x_firststage2=OP_firststage2.fit(method = "SLSQP", maxiter=200, disp=False)
    x_firststage3=OP_firststage3.fit(method = "SLSQP", maxiter=200, disp=False)

    ###### THIS WORKS ONLY FOR THIS SAMPLE
    start_params=np.concatenate((x_firststage1.x[2:],x_firststage1.x[:2],
                                 x_firststage2.x[2:],x_firststage2.x[:2],
                                 x_firststage3.x[2:],x_firststage3.x[:2]))

    ########################################################
    ### OPTIMIZATION
    ########################################################
    res = CNOP3.fit(start_params=start_params, maxiter=600)
    print res