import statsmodels as st
import statsmodels.api as sm
from scipy import stats
import numpy as np 
from statsmodels.sandbox.distributions.extras import mvstdnormcdf


#def jac(x, i):
#    jac = np.zeros(len(x))
#    jac[i+1], jac[i]  = -1, 1
#    return jac

class CNOP(st.discrete.discrete_model.DiscreteModel):
    """Correlatesdf BLA BLA BLA
    
    Description:
    TO BE WRITTEN"""
    def __init__(self,endog, exog, **kwargs):
        self.J, self.x, self.zplus, self.zminus = endog
        self.y = exog
    
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

    def loglike(self, params):
        """
        NOT YET TESTED 20150104!!!!!!!!!!!!!!!!!!!!!!!
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
        I use COBYLA algorithm
        
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
        j = round(xelement["Y"],10)
        for (xitem, xdf),   (zmitem, zmdf), (zpitem, zpdf) in \
        izip(x.iteritems(), zm.iteritems(), zp.iteritems() ): 
            for (xtime, xelement), (zmtime, zmelement), (zptime, zpelement) in \
            izip(xdf.iterrows(),   zmdf.iterrows(),     zpdf.iterrows() ): 
                #xelement, zmelement, zpelement are Series of interest for item xitem
                # and for time xitem. Items and times are identical throughout theree Panels
                assert xitem == zmitem, "Items doesn't match: xitem != zmitem"
                assert zmitem == zpitem, "Items doesn't match: zpitem != zmitem"
                assert xtime == zmtime, "Times doesn't match: xitem != zmitem"
                assert zmtime == zptime, "Times doesn't match: zpitem != zmitem"

                if j == 0:
                    pr = self.cdf(alpha[1] - np.dot(xelement, beta)) - self.cdf(alpha[0] - np.dot(xelement, beta))
                if j <= 0:
                    pr =  self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-1]-np.dot(zpelement, gammap), -rhop)
                    pr -= self.cdf(np.dot(xelement, beta) - alpha[1], mup[abs(j)-2]-np.dot(zpelement, gammap), -rhop)
                if j >= 0:
                    pr =  self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-1]-np.dot(zmelement, gammam), rhom)
                    pr -= self.cdf(alpha[0] - np.dot(xelement, beta), mum[abs(j)-2]-np.dot(zmelement, gammam), rhom)
                pr = np.clip(pr, FLOAT_EPS, 1)
                s += np.log(pr)
        return s


    def tester(self):
        x = [14.1,18.2,12.1,23.1,-12,123]
        J=4
        print self.cdf(1,0,1)
        test = self.cons_generator(J)
        cons = eval(test)
        print cons[1]
        #print self.J

    def jac(x, i):
        jac = np.zeros(len(x))
        jac[i+1], jac[i]  = -1, 1
        return jac
    
    def cons_generator(self, J, type = "ineq"):
        """#indian
        Function generates a string of constrants,
        required for optimization routine in scipy.optimize.
        
        You only need to eval() the output.
        """
        constrsstr = "["
        for i in range(J-1):
            constrsstr += '{"type":"' +type + '","fun":lambda x:np.array([float(x[' + \
                    str(i+1) + ']-x[' + str(i) + '])]),' +\
                    '"jac":lambda x:jac(x, '+str(i)+')},'
        constrsstr += "]"
        return constrsstr
        
    
    
TEST = CNOP([1,[12,13],[14,15],[17,17]],[12.15, 12.30])
TEST.tester()

x = [14.1,18.2,12.1,23.1,-12,123]
J=4

#test = TEST.cons_generator(J)
#cons = eval(test)
#print cons

