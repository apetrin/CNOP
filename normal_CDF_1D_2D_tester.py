import statsmodels as st
import statsmodels.api as sm
from scipy import stats
import numpy as np
from statsmodels.sandbox.distributions.extras import mvstdnormcdf

######### FUNCTION ###############


def cdf(self, X, Y=None, rho=None):
    """ Returns 1D or 2D standard normal CDF"""
    if Y is None:
        return stats.norm._cdf(X)
    else:
        return mvstdnormcdf([-np.inf, -np.inf],
                            [X, Y], rho)
    # else:
    #    error,value,inform = stats.mvn.mvndst([-np.inf, -np.inf], [X,Y], [0,0], rho)
    #    return value

##################################


def test1D(n):
    "1-D CDF tester"
    for i in xrange(n):
        cdf(None, X=np.random.normal())
        # computes two random vars for compatability with test2D
        np.random.normal(), np.random.rand()
# VERY effective


def test2D(n):
    "2-D CDF tester"
    for i in xrange(n):
        cdf(None, X=np.random.normal(),
            Y=np.random.normal(),
            rho=2 * (np.random.rand() - 0.5))
# VERY SLOW! :-(((((


#### TESTERS ####
import time
from datetime import datetime

start = time.time()
test1D(10000)
end = time.time()
print "Time for 1D:\t%.12f" % (end - start)

start = time.time()
test2D(10000)
end = time.time()
print "Time for 2D:\t%.12f" % (end - start)
print
print "Executed on " + str(datetime.now())

print cdf(None, -0.7826149711806311, 0.06742110669908244, 2 * (0.02578049908396074 - 0.5))
# three random vars. First two: st.norm. Third - uniform [0,1)
rands = [[-
          0.7826149711806311, 0.06742110669908244, 0.02578049908396074], [0.08850443176122, -
                                                                          1.304831143436072, 0.2154293586639222], [1.1931481097444596, 0.46887850502038925, 0.30152401888235014], [-
                                                                                                                                                                                   0.9196539646829609, -
                                                                                                                                                                                   0.4993375972062696, 0.9624409282016226], [-
                                                                                                                                                                                                                             2.0043430926624173, 0.07166831840915536, 0.03035067401709335], [-
                                                                                                                                                                                                                                                                                             1.860594616451685, 0.8288322592621417, 0.19041890820303942], [-
                                                                                                                                                                                                                                                                                                                                                           1.961649888990427, -
                                                                                                                                                                                                                                                                                                                                                           1.313062095030843, 0.6944063000155531], [-
                                                                                                                                                                                                                                                                                                                                                                                                    0.9160631070868033, 2.5020864976085386, 0.8325029262155285], [1.0986444993865494, -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                  0.8234126401637187, 0.5470620559504965], [-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1.6090922324340031, -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            1.7632682699479842, 0.2189980900085765]]
# CORRECT result for each of rands
res = [
    0.4159360655877158,
    0.7537960335313908,
    0.5948797687154994,
    0.247079364899756,
    0.6900485454203418,
    0.05947832226149406,
    0.3571700286844244,
    0.004230098349728095,
    0.11023843445690854,
    0.5623864874944672]
# WRONG result obrained from 16.25 file edition
res2_wrong = [
    0.0005339181835823684,
    0.01344508407145032,
    0.5769874986749153,
    0.17031759567884822,
    2.0041212282383103e-10,
    0.007682719361730178,
    0.008269529051997095,
    0.17981610777966664,
    0.1828331759542227,
    1.6022909479176083e-05]
