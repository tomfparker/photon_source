import TimeBinMultiplex as tMux
import SpatialMultiplex as sMux
import MultiplexAnalytical as aMux

from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize



#Plot the maximum delivery proabbility P(S,H) ( (0.x*max) against etas for const g2


def plot_Compare_NMax(etaH, g2, max, etaSmin, etaSmax, step):

    fig, ax = plt.subplots()


# get_etaS_N(etaSmin, etaSmax, g2, etaH, max, step)
# analytic = aMux.get_etaS_N(etaSmin, etaSmax, g2, etaH, max, step)

# plt.plot(analytic[0], analytic[1])

    etaLoopArray = np.arange(etaSmin, etaSmax, 0.1);

    #getMaxRange(etaLoopvals, g2, NMax, etaH, etaS, trunc)
   # time  = tMux.getMaxRange(etaLoopArray, g2, 40, etaH, 0.99, 4)


    #plt.scatter(etaLoopArray, time[1])


    space  = sMux.getMaxRange(etaLoopArray, 0.3, 40, 0.95, 0.99, 4)


    plt.scatter(etaLoopArray, space[1])


    plt.xlabel('$\eta_S$')
    plt.ylabel('N$_{max}$')
    plt.title(' $g^{(2)}_H(0) = %s$' % g2)


   # fi2, ax2 = plt.subplots()

   # plt.plot( np.arange(etaSmin, etaSmax, step), max*np.arange(etaSmin, etaSmax, step))

   # plt.plot(etaLoopArray, time[0])

  #  plt.plot(etaLoopArray, space[0])


plot_Compare_NMax(0.9, 0.1, 0.9, 0.7, 0.999, 0.01)
plt.show()