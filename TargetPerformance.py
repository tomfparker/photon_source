from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize
from MultiplexAnalytical import *
import New_Time_Multiplex as ntm

#########################################################################################################

#Functions to plot N and ets for source target performances.

#########################################################################################################


def getN_successProb(probSuccess, etaH, etaS, g2):

     squeezing =  get_squeezing(g2, etaH)

     probTempFunc = lambda NTemp: JointPronAnalytic(squeezing, etaH, etaS, NTemp) - probSuccess

     return int(fsolve( probTempFunc, 1))



def get_source_performance(prob, g2, etaH, step):

    etaSrange = np.arange(prob+0.01,0.97, step )

    sourceNum = [getN_successProb(prob, etaH, etaS, g2) for etaS in etaSrange]

    return [etaSrange, sourceNum]




def getMinEta_binary(probSuccess, etaH, g2, Nmax, etaSmin, etaSmax, step):


    etaLoop= 0.999
    maxVals = [getMaxNP(Nmax,  etaH, etaLoop,etaSTemp, g2)[1] for etaSTemp in np.arange(etaSmin, etaSmax, step )]
    idx = (np.abs(np.subtract(maxVals,probSuccess))).argmin()


    return(np.arange(etaSmin, etaSmax, step )[idx])



def getN_successProb_binary(probSuccess, etaH, etaS, g2):

    squeezing = get_squeezing(g2, etaH)

    probTempFunc = lambda NTemp: JointPronAnalytic(squeezing, etaH, etaS**( np.log2(NTemp)), NTemp) - probSuccess

    return int(fsolve(probTempFunc, 1))





def getN_successProb_binary_search(probSuccess, etaH, etaS, g2, Nmax):

    squeezing = get_squeezing(g2, etaH)

    probArray =[JointPronAnalytic(squeezing, etaH, etaS ** (np.log2(NTemp)), NTemp) for NTemp in range(1, Nmax, 1)]

    idx = (np.abs(np.subtract(probArray, probSuccess))).argmin()

    return np.arange(1, Nmax, 1)[idx]
    #return  [probArray, np.arange(1, Nmax, 1)[idx]]



def getN_successProb_const_search(probSuccess, etaH, etaS, g2, Nmax):

    squeezing = get_squeezing(g2, etaH)

    probArray =[JointPronAnalytic(squeezing, etaH, etaS , NTemp) for NTemp in range(1, Nmax, 1)]

    idx = (np.abs(np.subtract(probArray, probSuccess))).argmin()

    #return np.arange(1, Nmax, 1)[idx]
    return [ np.arange(1, Nmax, 1)[idx]]




def get_source_performance_binary(prob, g2, etaH, step, NMax_search, etaSmin, etaSmax):


    etaMin = getMinEta_binary(prob, etaH, g2, NMax_search, etaSmin, etaSmax, step)

    etaSrange = np.arange(etaMin+0.05,0.95, step, float )

    sourceNum = [getN_successProb_binary_search(prob, etaH, etaS, g2, NMax_search) for etaS in etaSrange]

    return [etaSrange, sourceNum]


def get_source_performance_binary_solver(prob, g2, etaH, step, NMax_search, etaSmin, etaSmax):

    """
    Solve for the number of sources: for a binary tree switch network, return the switch loss range and corresponding number of sources
    """

    etaMin = getMinEta_binary(prob, etaH, g2, NMax_search, etaSmin, etaSmax, step)

    etaSrange = np.arange(etaMin+0.01,0.99, step, float )

    sourceNum = [getN_successProb_binary(prob, etaH, etaS, g2)for etaS in etaSrange]


    return [etaSrange, sourceNum]


#########################################################################################################

#fuctions for chained switch networks

#########################################################################################################




#########################################################################################################

# plotting functions

#########################################################################################################

def plot_target_contours(prob, g2, etaH, step, Nplotmax, NMax, etaSmin, etaSmax, Nstart):
    """
    Function to produce a plot of the target performance for each scheme


     """

    const_data =  get_source_performance(prob, g2, etaH, step)
    bin_data  =  get_source_performance_binary_solver(prob, g2, etaH, step, NMax, etaSmin, etaSmax)

    print(const_data)
    print(bin_data)

    chained_data = ntm.get_performance_search(prob, g2, etaH, 0.99, etaSmin, etaSmax, 2, 1, step, NMax, Nstart)

    fig, ax = plt.subplots()

    plt.scatter(const_data[0], const_data[1], color='black', alpha =0.3)
    plt.plot([prob, prob],[0,Nplotmax], 'g--', color='black', alpha=0.2)

    plt.scatter(bin_data[0], bin_data[1], color='blue', alpha = 0.3)
    plt.scatter(bin_data[0][0], bin_data[1][0], marker='*',color='blue', s=200, alpha =0.3)
    plt.plot([bin_data[0][0], bin_data[0][0]], [0, Nplotmax], 'g--', color='black', alpha=0.2)

    plt.scatter(chained_data[0], chained_data[1], color='green', alpha = 0.3)
    plt.scatter(chained_data[0][len(chained_data[0])-1], chained_data[1][len(chained_data[1])-1], marker='*', color='green', s=200, alpha=0.3)
    plt.plot([chained_data[0][len(chained_data[0])-1], chained_data[0][len(chained_data[0])-1]], [0, Nplotmax], 'g--', color='black', alpha=0.2)

    max =getN_successProb_const_search(prob, etaH, 0.999, g2, NMax)

    plt.scatter(1,max[0],marker='o',color='black', s=150, alpha =0.4)


    plt.ylim([0, Nplotmax])
    plt.xlim([0, 1])

    plt.xlabel("$\eta_s$")
    plt.ylabel("$N$")



    return ax


    #plt.ylim((0, 1))

    #ax.set_aspect((NMax - 1))

def plot_target_contours_hard_coded_g2(prob, g2, etaH, step, Nplotmax, NMax, etaSmin, etaSmax, Nstart):
    """
    Function to produce a plot of the target performance for each scheme: for p =0.3, g2 = 0.01: N = 136, eta_s = 0.992
    this is hardcoded in.


     """

    const_data =  get_source_performance(prob, g2, etaH, step)
    bin_data  =  get_source_performance_binary_solver(prob, g2, etaH, step, NMax, etaSmin, etaSmax)

    #print(const_data)
    print(bin_data)

    #chained_data = ntm.get_performance_search(prob, g2, etaH, 0.99, etaSmin, etaSmax, 2, 1, step, NMax, Nstart)

    fig, ax = plt.subplots()

    plt.scatter(const_data[0], const_data[1], color='black', alpha =0.3)
    plt.plot([prob, prob],[0,Nplotmax], 'g--',color='black',alpha = 0.2)

    plt.scatter(bin_data[0], bin_data[1], color='blue', alpha = 0.3)
    plt.scatter(bin_data[0][0], bin_data[1][0], marker='*',color='blue', s=200, alpha =0.3)
    plt.plot([bin_data[0][0], bin_data[0][0]], [0, Nplotmax], 'g--', color='black', alpha = 0.2)

    #plt.scatter(chained_data[0], chained_data[1], color='green', alpha = 0.3)
    plt.scatter(0.992,136 , marker='*', color='green', s=200, alpha=0.3)
    plt.plot([0.992, 0.992], [0, Nplotmax], 'g--', color='black', alpha = 0.2)

    max =getN_successProb_const_search(prob, etaH, 0.999, g2, NMax)

    plt.scatter(1,max[0],marker='o',color='black', s=150, alpha =0.4)


    plt.ylim([0, Nplotmax])
    plt.xlim([0, 1])

    plt.xlabel("$\eta_s$")
    plt.ylabel("$N$")



    return ax


#def plot_etsmin_vs_g2(prob, g2, etaH, step, Nplotmax, NMax, etaSmin, etaSmax, Nstart)


#print(getMinEta_binary(0.3, 0.9, 0.01, 1000, 0.01, 0.95, 0.01))


    #print(getMaxNP(200,  0.9, 0.25, 0.99, 0.1))

    #print(getN_successProb_binary_search(0.2, 0.9, 0.25, 0.01, 300))
    #print(getN_successProb_binary_search(0.2, 0.9, 0.9, 0.01, 200))
    #print(getN_successProb_const_search(0.2, 0.9, 0.9, 0.01, 200))

    #print(getN_successProb(0.2, 0.9, 0.9, 0.01))




    #def get_source_performance_Binary(prob, g2, etaH, step, etaTestmin, etaTestmax, Nmax, testStep):


    #    etaSmin = getMinEta_binary(prob, etaH, g2, Nmax, etaTestmin,  etaTestmax, testStep)
    #    etaSrange = np.arange(etaSmin + 0.01, 0.99, step)

     #   sourceNum = [getN_successProb_binary(prob, etaH, etaS, g2) for etaS in etaSrange]

    #    return [etaSrange, sourceNum]



    #print( getMinEta_binary(0.5, 0.9, 0.1, 300, 0.1,  0.99, 0.01))

    #print(JointPronAnalytic( get_squeezing(0.1, 0.9), 0.9, 0.8**( np.log2(60)), 60))


    #print(getN_successProb_binary(0.5, 0.9, 0.6, 0.1))

    #print(get_source_performance_Binary(0.05, 0.1, 0.9, 0.1, 0.1, 0.9, 300, 0.01))

    #print(getN_successProb_binary(0.05, 0.9, 0.9, 0.1))




    #print(getMinEta_binary(0.05, 0.9, 0.1, 100))



    #print(getN_successProb(0.2, 0.9, 0.9, 0.01))
    #Nt = getN_successProb_binary_search(0.2, 0.9, 0.9, 0.01, 100)
    #sq =  get_squeezing(0.01, 0.9)
    #ex = get_source_performance(0.2, 0.01, 0.9, 0.001)
    #print(JointPronAnalytic(sq, 0.9, 0.9, Nt[0]))
    #print(JointPronAnalytic(sq, 0.9, 0.9, 61))

    #exampleProb = [JointPronAnalytic(sq, 0.9, 0.9, N) for N in np.arange(1,50,1)]
    #print(range(1,60,1))

    #plt.plot( range(1,50,1), exampleProb)
    #print(ex)

    #print(get_source_performance_binary(0.2, 0.1, 0.9, 0.1, 600, 0.1, 0.9))

    #
    #plt.plot(ex[0], ex[1])
    #const_data = get_source_performance(0.1,0.01,0.9,0.01)

    #print(const_data[0])
    #print(const_data[1])

    #plt.plot(const_data[0],const_data[1])



plot_target_contours(0.3, 0.01, 0.9, 0.005, 60, 200, 0.3, 0.999, 5)
#plot_target_contours(0.3, 0.01, 0.9, 0.1, 500)

#get_source_performance_binary(prob, g2, etaH, step, NMax_search, etaSmin, etaSmax)

#bin_example = get_source_performance_binary(0.3, 0.1, 0.9, 0.03, 500, 0.5, 0.99)
#print(bin_example[0])
#print([bin_example[1][i][1] for i in np.arange(0,len(bin_example[1]),1)])
#plt.plot(bin_example[0],bin_example[1])

#print(getN_successProb_binary_search(0.1, 0.9, etaS, g2, Nmax))
#print(getMinEta_binary(0.3, 0.9, 0.1, 500, 0.1, 0.99, 0.01))
#print(getN_successProb_binary(0.3, 0.9, 0.35, 0.1))

#example_bin = get_source_performance_binary(0.3, 0.1, 0.9, 0.01, 700, 0.5, 0.99)
#plt.plot(example_bin[0], example_bin[1])

#example_2 =  get_source_performance_binary_solver(0.3, 0.1, 0.9, 0.01, 200, 0.2, 0.999)
#plt.plot(example_2[0], example_2[1])
#print(getMinEta_binary(0.3, 0.9, 0.1, 200, 0.1, 0.99, 0.01))

plt.show()

#print(getMinEta_binary(0.3, 0.9, 0.01, 1000, 0.4, 0.99, 0.01))