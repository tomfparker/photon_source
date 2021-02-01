from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize


#plt.style.use('seaborn-muted')



# Multiplexing with a storage loop and a single switch


def binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

######################################################################################################################

#Functions for photon statistics



def prob(n,r):
    """
       Photon statistics for a thermal state. photon number n, squeezing r
       """
    return ((1/(np.cosh(r)))**2)*(np.tanh(r)**(2*n))


def condition_prob_herlad_k(k, etaH, D):
    """
         The conditional probability of heralding given k photons with D multiplexed detectors
         """

    return sum([(etaH**n)*((1- etaH)**(k-n))*binomial(n,k)*((1/D)**(n-1)) for n in range(1,k+1,1)])


def conditionalProb_nk(n , etaLoop, etaS, k, j, N):
    """
          The conditional probability of the preperation of n photons given k were generated in the jth time bin
          """
    return (((etaLoop**(N-j))*etaS)**n)*((1-(etaLoop**(N-j))*etaS)**(k-n))*binomial(k,n)

def jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r):
    """
              The joint probability of delivering m photons and a herald
              """

    return sum([((1- sum([prob(k1, r)*condition_prob_herlad_k(k1, etaH, D) for k1 in range(1, trunc+1, 1)]))**(N-j+1)) *  sum([prob(k2, r)*condition_prob_herlad_k(k2, etaH, D)*conditionalProb_nk(m , etaLoop, etaS, k2, j, N) for k2 in range(1, trunc+1, 1)]) for j in range(1,N+1,1)])


def jointProb_SH(N, etaH, etaS, etaLoop, r, trunc, D):
    """
                  The joint probability of signal and herald field detection
                  """

    return sum([jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r) for m in range(1,trunc+1)])

def prob_Herald(N, etaH, D, trunc,r):
    """
                     The joint probability of heralding a photon from N time bins, assuming that we only use the
                     most recent photon from teh pulse train
                     """

    return sum([sum([condition_prob_herlad_k(k, etaH, D)*prob(k,r) for k in range(1,trunc+1,1)]) * (1 -  sum([condition_prob_herlad_k(k2, etaH, D)*prob(k2,r) for k2 in range(1,trunc+1,1)]))**(N-j) for j in range(1,N+1,1)])


def conditional_nH(m, N, etaLoop, etaS, etaH, D, trunc, r):
    """
                               The
                                  """

    return jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r)/prob_Herald(N, etaH ,D,trunc, r)


def g2heralded(N, etaH, etaS, etaLoop, r ,trunc, D):

    """
                     Return the heralded g2
                      """
    return sum([n*(n-1)*conditional_nH(n , N, etaLoop, etaS, etaH, D, trunc, r) for n in range(1,trunc+1,1)])/(sum([n *conditional_nH(n, N, etaLoop, etaS, etaH, D, trunc, r) for n in range(1, trunc+1, 1)]))**2





"""

The old way of calculating the g2: Think this is incorrect


def conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc,D):
    """
                     #The probability of delivering n photons conditioned on a herald event
"""

    probArray = []

    for j in range(1,N+1,1):

        numeratorTemp = sum([prob(k, r) * condition_prob_herlad_k(k, etaH, D) * conditionalProb_nk(n, etaLoop, etaS, k, j, N) for k in
         range(1, trunc + 1, 1)])

        denominatorTemp =  sum([prob(k, r) * condition_prob_herlad_k(k, etaH, D)  for k in
         range(1, trunc + 1, 1)])

        probArray.append(numeratorTemp/denominatorTemp)



    return sum(probArray)


def g2heralded(N, etaH, etaS, etaLoop, r ,trunc, D):

    """
                    # Return the heralded g2
"""

    return sum([n*(n-1)*conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc,D) for n in range(1,trunc+1,1)])/(sum([n * conditional_nH(n, N, etaH, etaS, etaLoop, r, trunc,D) for n in range(1, trunc+1, 1)]))**2
  """


################################################################################################################

#Solve for g2 function for squeezing param
def get_squeezing(g2, N, etaH, etaS, etaLoop,  trunc,D):

    """
                   Solve for the squeezing parameter r for a given g2
                      """
    g2TempFunc= lambda rtemp :   g2heralded(N, etaH, etaS, etaLoop, rtemp, trunc,D) - g2

    return  fsolve(g2TempFunc, 0.1)


#get squeezing values over a range of source numbers
def get_getSqueezing_array(Nmax,  etaH, etaS, etaLoop,  trunc, g2, step, D ):
    """
                       return an array of squeezing different source nums up to Nmax
                          """

    return [get_squeezing(g2, int(N), etaH, etaS, etaLoop, trunc, D)[0] for N in np.arange(1,Nmax+1,step)]


###############################################################################################################


def get_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc):
    """
                           Get the joint probability of detection up to NMax
                              """
    Narray = np.arange(2, N_max, 1)

    psh = [jointProb_SH(N, etaH, etaS, etaLoop, r, trunc, D) for N in Narray]

    return [Narray, psh]

def plot_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc):

    [Narray, psh] = get_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc)

    f, ax = plt.subplots()


    plt.plot(Narray, psh)

    #plt.xlim((0, 1))
    #plt.ylim((0, 0.5))

    return ax




def plot_SH_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D):
    """
                               Plot the joint probability against source number for a constant g2
                                  """

    sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)
    num =  np.arange(1, Nmax + 1,step )
    joint = [jointProb_SH(int(N), etaH, etaS, etaLoop, sq[indx], trunc, D) for  indx, N in enumerate(num)]


    f, ax = plt.subplots()

    plt.plot(num, joint)



    plt.xlabel("Source Number")
    plt.ylabel("Probability of Success")

    return [ax, num, joint]


def compare_SH_N_constg(Nmax, etaH, etaS, etaLoopArray, trunc, g2, step, D):




    num =  np.arange(1, Nmax + 1,step )
    joint = [plot_SH_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)[2] for etaLoop in etaLoopArray]

    f1, ax1 = plt.subplots()
    [ax1.plot(num, j) for j in joint]

    plt.xlabel("Time Bins")
    plt.ylabel("Probability of Success P(S,H)")

    plt.xlim((1, Nmax - 1))
    plt.ylim((0, 1))

    ax1.set_aspect((Nmax-1))

    return [ax1, num, joint]


def plot_one_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D):
    """
                               Plot the p(1|H) against source number for a constant g2
                                  """

    sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)
    num =  np.arange(1, Nmax + 1,step )
    joint = [jointprob_mH(1,int(N), etaLoop, etaS, etaH, D, trunc, sq[indx]) for indx, N in enumerate(num)]



    f, ax = plt.subplots()

    plt.plot(num, joint)

    plt.xlim((0, Nmax))
    plt.ylim((0, 1))

    plt.xlabel("Source Number")
    plt.ylabel("Single Photon Probability p(1,H)")

    return [ax, num, joint]


#plot_SH_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.009, 1, 4)

def getMaxNP(Nmax,  etaH, etaS, etaLoop,  trunc, g2, D):
    """
                                   Get the maximum success probability and corresponding number of sources required
                                      """
    sq1 = get_squeezing(g2, 1, etaH, etaS, etaLoop, trunc, D)
    sq2 = get_squeezing(g2, 2, etaH, etaS, etaLoop, trunc, D)
    psArraytemp = [jointProb_SH(int(1), etaH, etaS, etaLoop, sq1, trunc, D),jointProb_SH(int(2), etaH, etaS, etaLoop, sq2, trunc, D)]



    for N in range(2,Nmax+1,1):



        if psArraytemp[N-1] > psArraytemp[N-2]:

            sq = get_squeezing(g2, N+1, etaH, etaS, etaLoop, trunc, D)
            psArraytemp.append(jointProb_SH(int(N+1), etaH, etaS, etaLoop, sq, trunc, D))

        else:

            Nmax = N -1
            psMax = psArraytemp[N-2]

            break

    return [Nmax, psMax]



def plot_P_and_N_max(Nmax,  etaH, etaS, etaLoopMin, etaLoopMax,  trunc, g2, D, etaLoopStep):



    data = [getMaxNP(int(Nmax), etaH, etaS, etaLoop, trunc, g2, D) for etaLoop in np.arange(etaLoopMin, etaLoopMax, etaLoopStep)]

    Parray = [i[1] for i in data]

    NMaxarrray = [i[0] for i in data]

    etaArray  = np.arange(etaLoopMin, etaLoopMax, etaLoopStep)

    f, ax = plt.subplots()

    plt.plot(etaArray, Parray)

    plt.xlim((etaLoopMin,etaLoopMax))
    plt.ylim((0, 1))

    return([etaArray, Parray, NMaxarrray])

def plotData(logtreeFile, timeFile, nby1File):

    logTree = np.loadtxt(logtreeFile, delimiter=",")

    etaSlogtree = logTree[0]
    NMaxLogTree = logTree[2]
    ProbLogTree  = logTree[1]

    timeData = np.loadtxt(timeFile, delimiter=",")

    etaS_Time = timeData[0]
    NMaxtime = timeData[2]
    Probtime = timeData[1]

    f, ax = plt.subplots()

    plt.plot(np.append(etaSlogtree,1), np.append(ProbLogTree,1))
    plt.plot(etaSlogtree, etaSlogtree)
    plt.plot(np.append(etaS_Time,1), np.append(Probtime,1))

    nby1 = np.loadtxt(nby1File, delimiter=",")

    eta_s_nby1 = nby1[0]
    N_nby1 = nby1[1]


    print([etaSlogtree, ProbLogTree])

    plt.xlim(0.5, 1)
    plt.ylim((0, 1))

    plt.xlabel("$\eta_S$")
    plt.ylabel("Maximum Success Probability")

    f2, ax2 = plt.subplots()


    plt.plot(etaSlogtree, NMaxLogTree)
    plt.plot(eta_s_nby1, N_nby1)
    plt.plot(etaS_Time, NMaxtime)


    plt.xlabel("Source Number N")
    plt.ylabel("Success Probability")

    plt.xlim(0.5, 1)
    plt.ylim((0, 120))

    ax.set_aspect(1/2)
    ax2.set_aspect(0.5 / 120)
from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize


#plt.style.use('seaborn-muted')



# Multiplexing with a storage loop and a single switch


def binomial(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke.
    See http://stackoverflow.com/questions/3025162/statistics-combinations-in-python
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

######################################################################################################################

#Functions for photon statistics



def prob(n,r):
    """
       Photon statistics for a thermal state. photon number n, squeezing r
       """
    return ((1/(np.cosh(r)))**2)*(np.tanh(r)**(2*n))


def condition_prob_herlad_k(k, etaH, D):
    """
         The conditional probability of heralding given k photons with D multiplexed detectors
         """

    return sum([(etaH**n)*((1- etaH)**(k-n))*binomial(n,k)*((1/D)**(n-1)) for n in range(1,k+1,1)])


def conditionalProb_nk(n , etaLoop, etaS, k, j, N):
    """
          The conditional probability of the preperation of n photons given k were generated in the jth time bin
          """
    return (((etaLoop**(N-j))*etaS)**n)*((1-(etaLoop**(N-j))*etaS)**(k-n))*binomial(k,n)

def jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r):
    """
              The joint probability of delivering m photons and a herald
              """

    return sum([((1- sum([prob(k1, r)*condition_prob_herlad_k(k1, etaH, D) for k1 in range(1, trunc+1, 1)]))**(N-j+1)) *  sum([prob(k2, r)*condition_prob_herlad_k(k2, etaH, D)*conditionalProb_nk(m , etaLoop, etaS, k2, j, N) for k2 in range(1, trunc+1, 1)]) for j in range(1,N+1,1)])


def jointProb_SH(N, etaH, etaS, etaLoop, r, trunc, D):
    """
                  The joint probability of signal and herald field detection
                  """

    return sum([jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r) for m in range(1,trunc+1)])

def prob_Herald(N, etaH, D, trunc,r):
    """
                     The joint probability of heralding a photon from N time bins, assuming that we only use the
                     most recent photon from teh pulse train
                     """

    return sum([sum([condition_prob_herlad_k(k, etaH, D)*prob(k,r) for k in range(1,trunc+1,1)]) * (1 -  sum([condition_prob_herlad_k(k2, etaH, D)*prob(k2,r) for k2 in range(1,trunc+1,1)]))**(N-j) for j in range(1,N+1,1)])


def conditional_nH(m, N, etaLoop, etaS, etaH, D, trunc, r):
    """
                               The
                                  """

    return jointprob_mH(m, N, etaLoop, etaS, etaH, D, trunc, r)/prob_Herald(N, etaH ,D,trunc, r)


def g2heralded(N, etaH, etaS, etaLoop, r ,trunc, D):

    """
                     Return the heralded g2
                      """
    return sum([n*(n-1)*conditional_nH(n , N, etaLoop, etaS, etaH, D, trunc, r) for n in range(1,trunc+1,1)])/(sum([n *conditional_nH(n, N, etaLoop, etaS, etaH, D, trunc, r) for n in range(1, trunc+1, 1)]))**2





"""

The old way of calculating the g2: Think this is incorrect


def conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc,D):
    """
                     #The probability of delivering n photons conditioned on a herald event
"""

    probArray = []

    for j in range(1,N+1,1):

        numeratorTemp = sum([prob(k, r) * condition_prob_herlad_k(k, etaH, D) * conditionalProb_nk(n, etaLoop, etaS, k, j, N) for k in
         range(1, trunc + 1, 1)])

        denominatorTemp =  sum([prob(k, r) * condition_prob_herlad_k(k, etaH, D)  for k in
         range(1, trunc + 1, 1)])

        probArray.append(numeratorTemp/denominatorTemp)



    return sum(probArray)


def g2heralded(N, etaH, etaS, etaLoop, r ,trunc, D):

    """
                    # Return the heralded g2
"""

    return sum([n*(n-1)*conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc,D) for n in range(1,trunc+1,1)])/(sum([n * conditional_nH(n, N, etaH, etaS, etaLoop, r, trunc,D) for n in range(1, trunc+1, 1)]))**2
  """


################################################################################################################

#Solve for g2 function for squeezing param
def get_squeezing(g2, N, etaH, etaS, etaLoop,  trunc,D):

    """
                   Solve for the squeezing parameter r for a given g2
                      """
    g2TempFunc= lambda rtemp :   g2heralded(N, etaH, etaS, etaLoop, rtemp, trunc,D) - g2

    return  fsolve(g2TempFunc, 0.1)


#get squeezing values over a range of source numbers
def get_getSqueezing_array(Nmax,  etaH, etaS, etaLoop,  trunc, g2, step, D ):
    """
                       return an array of squeezing different source nums up to Nmax
                          """

    return [get_squeezing(g2, int(N), etaH, etaS, etaLoop, trunc, D)[0] for N in np.arange(1,Nmax+1,step)]


###############################################################################################################


def get_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc):
    """
                           Get the joint probability of detection up to NMax
                              """
    Narray = np.arange(2, N_max, 1)

    psh = [jointProb_SH(N, etaH, etaS, etaLoop, r, trunc, D) for N in Narray]

    return [Narray, psh]

def plot_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc):

    [Narray, psh] = get_jointSH_N(r, N_max, etaLoop, etaS, etaH, D, trunc)

    f, ax = plt.subplots()


    plt.plot(Narray, psh)

    #plt.xlim((0, 1))
    #plt.ylim((0, 0.5))

    return ax




def plot_SH_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D):
    """
                               Plot the joint probability against source number for a constant g2
                                  """

    sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)
    num =  np.arange(1, Nmax + 1,step )
    joint = [jointProb_SH(int(N), etaH, etaS, etaLoop, sq[indx], trunc, D) for  indx, N in enumerate(num)]


    f, ax = plt.subplots()

    plt.plot(num, joint)



    plt.xlabel("Source Number")
    plt.ylabel("Probability of Success")

    return [ax, num, joint]

def plot_SH_N_constg2_array(Nmax, etaH, etaS, etaswitch_range, trunc, g2, step):
    """
                               Plot the joint probability against source number for a constant g2
                                  """

    sqarray = [get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step, 1) for etaLoop in etaswitch_range]
    num =  np.arange(1, Nmax + 1,step )
    joint = [[jointProb_SH(int(N), etaH, etaS, etaLoop, sq[indx], trunc, 1) for indx, N in enumerate(num)] for sq in sqarray]


    f, ax = plt.subplots()

    [plt.plot(num, jointval) for jointval in joint]
    #plt.plot(num, joint)

    plt.xlabel("$N$")
    plt.ylabel("$p_{success}$")

    plt.xlim((1, Nmax - 1))
    plt.ylim((0, 1))

    ax.set_aspect((Nmax - 1))

    return [ax, num, joint]

def thesis_plot_function(Nmax, etaswitch_range, trunc, g2, step):

    data = [plot_SH_N_constg2_array(Nmax, 0.9, 0.99, etaswitch, trunc, g2, step) for etaswitch in etaswitch_range]

    f, ax = plt.subplots()

    [plt.plot(plotdat[1], plotdat[2][0]) for plotdat in data]
    # plt.plot(num, joint)

    plt.xlabel("$N$")
    plt.ylabel("$p_{success}$")

    plt.xlim((1, Nmax - 1))
    plt.ylim((0, 1))

    ax.set_aspect((Nmax - 1))



    return data


def plot_Example_time(g2, etas,etaswitch_range, etaH, NMax, fileName):

    Narray = np.arange(1,NMax, 0.5)
    squeez = get_squeezing(g2, etaH)

    jProb = [[JointPronAnalytic(squeez, etaH, etas*p, i) for i in Narray] for p in etaswitch_range]

    #jprob2 = [JointPronAnalytic(squeez, etaH, etas*etaswitch**np.log2(i), i) for i in Narray]

    fig, ax = plt.subplots()
    [plt.plot(Narray, jProb_i) for jProb_i in jProb]
    #plt.plot(Narray, jprob2)
    plt.ylim([0,1])

    plt.xlabel("$N$")
    plt.ylabel("$p_{success}$")

    plt.xlim((1, NMax-1))
    plt.ylim((0, 1))

    ax.set_aspect((NMax - 1))

    np.savetxt(fileName,[Narray,jProb[0]], delimiter=",")


def compare_SH_N_constg(Nmax, etaH, etaS, etaLoopArray, trunc, g2, step, D):




    num =  np.arange(1, Nmax + 1,step )
    joint = [plot_SH_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)[2] for etaLoop in etaLoopArray]

    f1, ax1 = plt.subplots()
    [ax1.plot(num, j) for j in joint]

    plt.xlabel("Time Bins")
    plt.ylabel("Probability of Success P(S,H)")

    plt.xlim((1, Nmax - 1))
    plt.ylim((0, 1))

    ax1.set_aspect((Nmax-1))

    return [ax1, num, joint]





def plot_one_N_constg2(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D):
    """
                               Plot the p(1|H) against source number for a constant g2
                                  """

    sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step, D)
    num =  np.arange(1, Nmax + 1,step )
    joint = [jointprob_mH(1,int(N), etaLoop, etaS, etaH, D, trunc, sq[indx]) for indx, N in enumerate(num)]



    f, ax = plt.subplots()

    plt.plot(num, joint)

    plt.xlim((0, Nmax))
    plt.ylim((0, 1))

    plt.xlabel("Source Number")
    plt.ylabel("Single Photon Probability p(1,H)")

    return [ax, num, joint]


#plot_SH_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.009, 1, 4)

def getMaxNP(Nmax,  etaH, etaS, etaLoop,  trunc, g2, D):
    """
                                   Get the maximum success probability and corresponding number of sources required
                                      """
    sq1 = get_squeezing(g2, 1, etaH, etaS, etaLoop, trunc, D)
    sq2 = get_squeezing(g2, 2, etaH, etaS, etaLoop, trunc, D)
    psArraytemp = [jointProb_SH(int(1), etaH, etaS, etaLoop, sq1, trunc, D),jointProb_SH(int(2), etaH, etaS, etaLoop, sq2, trunc, D)]

    psMax=0

    for N in range(2,Nmax+1,1):



        if psArraytemp[N-1] >= psArraytemp[N-2]:

            sq = get_squeezing(g2, N+1, etaH, etaS, etaLoop, trunc, D)
            psArraytemp.append(jointProb_SH(int(N+1), etaH, etaS, etaLoop, sq, trunc, D))

        else:

            Nmaxval = N -1
            psMax = psArraytemp[N-2]

            break
    print(psArraytemp)
    return [Nmaxval, psMax]



def plot_P_and_N_max(Nmax,  etaH, etaS, etaLoopMin, etaLoopMax,  trunc, g2, D, etaLoopStep):



    data = [getMaxNP(int(Nmax), etaH, etaS, etaLoop, trunc, g2, D) for etaLoop in np.arange(etaLoopMin, etaLoopMax, etaLoopStep)]

    Parray = [i[1] for i in data]

    NMaxarrray = [i[0] for i in data]

    etaArray  = np.arange(etaLoopMin, etaLoopMax, etaLoopStep)

    f, ax = plt.subplots()

    plt.plot(etaArray, Parray)

    plt.xlim((etaLoopMin,etaLoopMax))
    plt.ylim((0, 1))

    return([etaArray, Parray, NMaxarrray])

def plotData(logtreeFile, timeFile, nby1File):

    logTree = np.loadtxt(logtreeFile, delimiter=",")

    etaSlogtree = logTree[0]
    NMaxLogTree = logTree[2]
    ProbLogTree  = logTree[1]

    timeData = np.loadtxt(timeFile, delimiter=",")

    etaS_Time = timeData[0]
    NMaxtime = timeData[2]
    Probtime = timeData[1]

    f, ax = plt.subplots()

    plt.plot(np.append(etaSlogtree,1), np.append(ProbLogTree,1))
    plt.plot(etaSlogtree, etaSlogtree)
    plt.plot(np.append(etaS_Time,1), np.append(Probtime,1))

    nby1 = np.loadtxt(nby1File, delimiter=",")

    eta_s_nby1 = nby1[0]
    N_nby1 = nby1[1]


    print([etaSlogtree, ProbLogTree])

    plt.xlim(0.5, 1)
    plt.ylim((0, 1))

    plt.xlabel("$\eta_S$")
    plt.ylabel("Maximum Success Probability")

    f2, ax2 = plt.subplots()


    plt.plot(etaSlogtree, NMaxLogTree)
    plt.plot(eta_s_nby1, N_nby1)
    plt.plot(etaS_Time, NMaxtime)


    plt.xlabel("Source Number N")
    plt.ylabel("Success Probability")

    plt.xlim(0.5, 1)
    plt.ylim((0, 120))

    ax.set_aspect(1/2)
    ax2.set_aspect(0.5 / 120)

def  jointProb_func(N, etaH, etaS, etaLoop, g2, trunc, D):

    sq = get_squeezing(g2, N, etaH, etaS, etaLoop, trunc, D)

    return jointProb_SH(N, etaH, etaS, etaLoop, sq, trunc, D)

#############################################################################
# Functions for source performance plots.

#Find N needed to reach pSuccess and gh2 @ etaH, D
def getN_successProb(probSuccess, etaH, etaS, etaLoop, g2, trunc, D):

    probTempFunc = lambda NTemp: jointProb_func(int(NTemp), etaH, etaS, etaLoop, g2, trunc, D) - probSuccess

    return int(fsolve( probTempFunc, 1))


#Find the minimum etaS at which pSuccess can be reached

def getMaxetaS(pSuccess, etaH, etaS,  etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax):

    init_prob = getMaxNP(Nmax, etaH, etaS, etaLoopMin, trunc, g2, D)[1]
    etaLoopVal = etaLoopMin

    for i in np.arange( etaLoopMin,etaLoopMax,etaStep):

        if init_prob <= pSuccess:

            etaLoopVal = i
            init_prob = getMaxNP(Nmax, etaH, etaS, i, trunc, g2, D)[1]


    return etaLoopVal




def getPerformanceParams(Prob, g2, etaH, etaS, etaLoopMin, etaLoopMax, trunc, D, etaStep, Nmax):

    #Get the etaS and N needed for p and g2

    etaSbound = getMaxetaS(Prob, etaH, etaS,  etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax)

    Narray = [getN_successProb(Prob, etaH, etaS, etaLoop, g2, trunc, D) for etaLoop in np.arange(etaSbound, etaLoopMax, etaStep)]

    return [np.arange(etaSbound, etaLoopMax, etaStep), Narray]

    #return etaSbound





def  jointProb_func(N, etaH, etaS, etaLoop, g2, trunc, D):

    sq = get_squeezing(g2, N, etaH, etaS, etaLoop, trunc, D)

    return jointProb_SH(N, etaH, etaS, etaLoop, sq, trunc, D)

#############################################################################
# Functions for source performance plots.

#Find N needed to reach pSuccess and gh2 @ etaH, D
def getN_successProb(probSuccess, etaH, etaS, etaLoop, g2, trunc, D):
    # This function does not seem to work, probably cos it can only take integer values of N

    probTempFunc = lambda NTemp: jointProb_func(int(NTemp), etaH, etaS, etaLoop, g2, trunc, D) - probSuccess

    return fsolve(probTempFunc, 10)

    #return(probTempFunc)

"""""
def getN_successProb_search(probSuccess, etaH,etaS, etaLoop, g2, trunc, D):



    etaSrange = np.arange(0.99, etaLoop, -etaSstep)
    Narray = [1]

    for index, etaSval in enumerate(etaSrange):

        joint_temp = 0

        while joint_temp < probSuccess:

            joint_temp = jointProb_func(int(Narray(index)), etaH,  etaS, etaSval, g2, trunc, D)
            Narray(index) = Narray(index) + 1

        else if index < len(etaSrange):

            Narray.append(Narray(index))

    return(Narray)
"""

def getN_successProb_search(probSuccess, etaH,etaS, etaLoop, g2, trunc, D, Nstart, Nmax):

    """
    search for min N needed to satisfy ProbSuccess and g2 for given source params, starting search at Nstart
    """

    joint_temp = 0
    N = Nstart

    while joint_temp < probSuccess:

        print([N, joint_temp])
        if N < Nmax:

            joint_temp = jointProb_func(int(N), etaH,  etaS, etaLoop, g2, trunc, D)
            N = N + 1

        else:

            N = Nmax
            break

    return(N)


def get_performance_search(Prob, g2, etaH, etaS, etaLoopMin, etaLoopMax, trunc, D, etaStep, Nmax, Nstart):

    """
    get target performance based on search through N array
    """

    etaSbound = getMaxetaS(Prob, etaH, etaS, etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax)
    Narray = []
    print(etaSbound)
    Nstart_temp = Nstart

    for  etaSval in np.arange(etaLoopMax,etaSbound, -etaStep):

        Nval = getN_successProb_search(Prob, etaH, etaS, etaSval, g2, trunc, D, Nstart_temp, Nmax)
        Narray.append(Nval)

        Nstart_temp = Nval-5

        print([etaSval, Nval])

    return(np.arange(etaLoopMax,etaSbound, -etaStep), Narray)





#Find the minimum etaS at which pSuccess can be reached

def getMaxetaS(pSuccess, etaH, etaS,  etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax):

    init_prob = getMaxNP(Nmax, etaH, etaS, etaLoopMin, trunc, g2, D)[1]
    etaLoopVal = etaLoopMin

    for i in np.arange( etaLoopMin,etaLoopMax,etaStep):
        print([i, init_prob])
        if init_prob < pSuccess:



            etaLoopVal = i
            init_prob = getMaxNP(Nmax, etaH, etaS, i, trunc, g2, D)[1]


    return etaLoopVal




def getPerformanceParams(Prob, g2, etaH, etaS, etaLoopMin, etaLoopMax, trunc, D, etaStep, Nmax):

    #Get the etaS and N needed for p and g2

    etaSbound = getMaxetaS(Prob, etaH, etaS,  etaLoopMin, etaLoopMax, trunc, g2, D, etaStep, Nmax)

    Narray = [getN_successProb(Prob, etaH, etaS, etaLoop, g2, trunc, D) for etaLoop in np.arange(etaSbound, etaLoopMax, etaStep)]

    return [np.arange(etaSbound, etaLoopMax, etaStep), Narray]

    #return etaSbound



#print(getMaxetaS(0.3,0.9,0.99,0.5, 0.99, 3, 0.1, 1, 0.01, 500))
""""
example =getPerformanceParams(0.3,0.1,0.9,0.99, 0.6, 0.99, 3, 1, 0.01, 1000)
print(example)
#plt.plot(example[0], example[1])
"""

""""
                   Kwiat source statistics 
                      """

#plot_one_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.009, 1, 4)
#plot_one_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.09, 1, 4)

#plot_SH_N_constg2(40, 0.9, 0.99, 0.9, 4, 0.01, 1, 1)
#plot_SH_N_constg2_array(40, 0.9, 0.99, [0.99,0.9], 3, 0.1, 2)
#output_test = plot_SH_N_constg2_array(20, 0.9, 0.99, [0.99,0.99,0.8], 3, 0.1, 2)
#output_test_2 = plot_SH_N_constg2_array(20, 0.9, 0.99, [0.9], 3, 0.1, 2)
#output_test_3 = plot_SH_N_constg2_array(20, 0.9, 0.99, [0.99], 3, 0.1, 2)
#plot_SH_N_constg2_array(20, 0.9, 0.99, [0.8], 3, 0.1, 2)

#print(output_test)
#print(output_test_3)


#thesis_plot_function(50, [[0.95]], 3, 0.1, 5)
#thesis_plot_function(50, [[0.95]], 3, 0.1, 5)

#plt.show()

#print(getN_successProb(0.3, 0.9, 0.99, 0.95, 0.1, 3, 1))
#test= [jointProb_func(int(NTemp), 0.9, 0.99, 0.95, 0.1, 3, 1)  for NTemp in np.arange(1,30,1)]
#print(test)
#print(getN_successProb(0.3, 0.9, 0.99, 0.95, 0.1, 3, 1))

#print(getN_successProb_search(0.3, 0.9,0.99, 0.98, 0.1, 3, 1, 1, 50))
#data = get_performance_search(0.5, 0.1, 0.9, 0.99, 0.8, 0.99, 3, 1, 0.01, 100, 5)
#print(data)

##plt.plot(data[0], data[1])


#plt.show()
#print(getMaxetaS(0.3, 0.9, 0.99, 0.9, 0.99, 2, 0.03, 1, 0.01, 300))
#print(getMaxNP(300, 0.9, 0.99, 0.995, 2, 0.01, 1))
#print(getN_successProb_search(0.3, 0.9, 0.99, 0.98, 0.03, 2, 1, 8,300))


"""
                   Kwiat source statistics 
                      """

#plot_one_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.009, 1, 4)
#plot_one_N_constg2(40, 0.62*0.88, 0.86*0.62, 0.988, 4, 0.09, 1, 4)


#print([getN_successProb_search(0.3, 0.9, 0.999, etaSval, 0.01, 2, 1, 100,1000 ) for etaSval in [0.992]])

###################################################################