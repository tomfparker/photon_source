from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize

import MultiplexAnalytical as ma
import New_Time_Multiplex as ntm
import TargetPerformance as tp



##############################################################################################


# plot functions

def min_eta_g2(prob, g2max, g2min, g2step, etaH, Nmax, etaSmin, etaSmax, etaStep ):


    g2array = np.arange(g2min, g2max, g2step)


    const_array = [prob for g2 in g2array]
    bin_array = [tp.getMinEta_binary(prob, etaH, g2, Nmax, etaSmin, etaSmax, etaStep) for g2 in np.arange(g2min, g2max, g2step)]
    chained_array = [ntm.getMaxetaS(prob, etaH, 0.99,  etaSmin, etaSmax, 2, g2, 1,etaStep, Nmax)for g2 in np.arange(g2min, g2max, g2step)]

    print(const_array)
    print(bin_array)
    print(chained_array)

    fig, ax = plt.subplots()

    plt.plot(g2array, const_array, color='black', alpha=0.3)
    plt.plot(g2array, bin_array, color='black', alpha=0.3)
    #plt.plot(g2array, chained_array, color='black', alpha=0.3)



    #plt.ylim([0, Nplotmax])
    #plt.xlim([0, 1])

    plt.xlabel("$\eta_s$")
    plt.ylabel("$N$")

    return ax



###############################################################################################

# Target performance plots


#plot_target_contours(prob, g2, etaH, step, Nplotmax, NMax, etaSmin, etaSmax, Nstart)


# This is figure plot for eta_s vs N
tp.plot_target_contours(0.3, 0.1, 0.9, 0.01, 60, 200, 0.3, 0.99, 10)

#tp.plot_target_contours_hard_coded_g2(0.3, 0.01, 0.9, 0.01, 700, 1000, 0.3, 0.99, 1)


#getMaxetaS(pSuccess, etaH, etaS,  etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax)

#print(ntm.getMaxetaS(0.3, 0.9, 0.99,  0.9, 0.98, 3, 0.1, 1, 0.01, 1000))

#print(ma.getMaxNP(10000,  0.9, 0.99, 0.9, 0.01))

#print(tp.getMinEta_binary(0.3, 0.9, 0.01, 2000, 0.3, 0.96, 0.01))
#tp.plot_target_contours(0.3, 0.01, 0.9, 0.01, 1000, 1000, 0.3, 0.96, 5)

#print(ntm.get_performance_search(0.3, 0.01, 0.9, 0.99, 0.9, 0.98, 3, 1, 0.02, 1000, 100))




#plt.show()

#print([tp.getMinEta_binary(0.3, 0.9, g2, 1000, 0.4, 0.99, 0.001) for g2 in np.arange(0.01, 0.1, 0.01)])

#print([tp.getMinEta_binary(0.3, 0.9, g2, 1000, 0.4, 0.99, 0.001) for g2 in np.arange(0.01, 0.1, 0.01)])


#min_eta_g2(0.3, 0.1, 0.01, 0.01, 0.9, 500, 0.7, 0.95, 0.001 )
plt.show()

#print(ntm.getMaxetaS(0.3, 0.9, 0.99,  0.9, 0.995, 2, 0.02, 1, 0.001, 500))