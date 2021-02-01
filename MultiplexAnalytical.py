from numpy import *
import matplotlib.pyplot as plt
from pylab import *
import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.optimize import minimize


#########################################################################################################

# Analytic statistics of multiplexing with a N x 1 switch

#########################################################################################################

# plt.style.use('seaborn-muted')



########## Functions Photon Statistics ##################################################################


# P(S,H)

def JointPronAnalytic(r, etaH, etaS, N):

    num= etaS * (-1 +2**N * (1/ (2 - etaH + etaH*np.cosh(2*r)))**N)* (4*(2 + etaH*(-1 + etaS) - etaS)*np.cosh(2*r) - (etaH*(-1 + etaS) - etaS)*(3+np.cosh(4*r)))
    denom = 2 *(-2 + etaH + etaS - etaH*etaS + (etaH* (-1 + etaS) - etaS) *np.cosh(2* r))* (2 - etaS + etaS*np.cosh(2*r))

    val = num/denom

    return val

# Heralded g^(2)

def g2Analytic(r, etaH):

    num = 4*np.sinh(r) ** 2 *(1 + (-1 + etaH)* np.tanh(r) ** 2) *(2 - etaH + (-1 + etaH)* tanh(r)** 2*(3 + (-1 + etaH) * tanh(r) ** 4))

    denom = (2-  etaH + etaH* cosh(2*r))*(1 + (-1 + etaH)*tanh(r)**4)**2

    return num/denom


######################### #############################################################################################


# Solve for the squeezing parameter r for a fixed g2 and etaH
def get_squeezing(g2, etaH):

    g2TempFunc= lambda rtemp :   g2Analytic(rtemp, etaH) - g2

    return  fsolve(g2TempFunc, 0.1)


# The max probability for infinite sources is etaS. This function solves for the number of sources to reach some fraction
# x of that maximum, for a given g2 and losses.
def get_N_x(g2, etaH, etaS, x):

    squeez = get_squeezing(g2, etaH)
    prob_TempFunc = lambda N90: JointPronAnalytic(squeez, etaH, etaS, N90) - x*etaS

    return fsolve(prob_TempFunc, 1)




# Get N needed for max prob for a range of values of etaS:
# Return a array of etaS and N_max values.
def get_etaS_N(etaSMin, etaSMax, g2, etaH, x, etaSstep):


    etaSArray = np.arange(etaSMin, etaSMax, etaSstep)

    N_x_Array = [get_N_x(g2, etaH, s, x) for s in etaSArray]

    return [etaSArray, N_x_Array ]




########### fixed N, g2



# For fixed g2 and N, find P(S,H) as a function of the herald and signal loss
def generate_prob_grid(N, g2):

    etaSRange = np.arange(0,1,0.02)
    etaHRange =  np.arange(0,1,0.02)

    X, Y = meshgrid(etaSRange, etaHRange)
    X2, Y2 = meshgrid( etaSRange,[get_squeezing(g2, i) for i in etaHRange])


    jointprobGrid = JointPronAnalytic(Y2, Y, X, N)

    print(jointprobGrid)

    fig = plt.figure()
    CS = plt.contour(jointprobGrid, extent=[0, 1, 0, 1], colors = 'k')
    plt.clabel(CS, inline=True, fontsize=10)

    plt.xlabel('$\eta_H$')
    plt.ylabel('$\eta_S$')
    plt.title('P(S,H): N = %s,  $g^{(2)}_H(0) = %s$'%(N, g2))


    plt.imshow(jointprobGrid, extent=[0, 1, 0, 1],origin='lower', alpha=0.5)




# Plot P(S,H) as a function of N for fixed g2, eta_s and eta_h
def plot_N_Prob(g2, etas,etaswitch, etaH, NMax):

    Narray = np.arange(1,NMax, 0.5)
    squeez = get_squeezing(g2, etaH)
    jProb = [JointPronAnalytic(squeez, etaH, etas*etaswitch, i) for i in Narray]

    #jprob2 = [JointPronAnalytic(squeez, etaH, etas*etaswitch**np.log2(i), i) for i in Narray]

    fig = plt.figure()
    plt.plot(Narray, jProb)
    #plt.plot(Narray, jprob2)
    plt.ylim([0,1])

    plt.xlabel("Source Number")
    plt.ylabel("Probability of Success")


def plot_Example(g2, etas,etaswitch_range, etaH, NMax, fileName):

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



def plot_Example_logTree(g2, etas,etaswitch_range, etaH, NMax, fileName):

    Narray = np.arange(1, NMax, 0.5)
    squeez = get_squeezing(g2, etaH)

    jProb = [[JointPronAnalytic(squeez, etaH, etas * p**np.log2(i), i) for i in Narray] for p in etaswitch_range]

    fig, ax = plt.subplots()
    [plt.plot(Narray, jProb_i) for jProb_i in jProb]
    # plt.plot(Narray, jprob2)
    plt.ylim([0, 1])

    plt.xlabel("$N$")
    plt.ylabel("$p_{success}$")

    plt.xlim((1, NMax-1))
    plt.ylim((0, 1))

    ax.set_aspect((NMax - 1))

    np.savetxt(fileName,[Narray,jProb[0]], delimiter=",")


def getMaxNP(Nmax,  etaH, etaS, etaLoop, g2):
    """
                                   Get the maximum success probability and corresponding number of sources required
                                      """
    sq = get_squeezing(g2, etaH)

    psArraytemp = [JointPronAnalytic(sq, etaH, etaS * etaLoop ** np.log2(1), 1),JointPronAnalytic(sq, etaH, etaS * etaLoop ** np.log2(2), 2)]

    Nmaxval = 0
    psMax = 0


    for N in range(2,Nmax+1,1):



        if psArraytemp[N-1] > psArraytemp[N-2]:


            psArraytemp.append(JointPronAnalytic(sq, etaH, etaS * etaLoop ** np.log2(N+1), N+1))

        else:

            Nmaxval = N -1
            psMax = psArraytemp[N-2]

            break

    return [Nmaxval, psMax]

def find_N_at_P():
    """
                                       For a given Probability (and switch efficiency and g2) find the number of sources needed
                                          """

    JointPronAnalytic(r, etaH, etaS, N) - P


def plot_P_and_N_max(Nmax,  etaH, etaS, etaLoopMin, etaLoopMax, g2, etaLoopStep):



    data = [getMaxNP(int(Nmax), etaH, etaS, etaLoop, g2) for etaLoop in np.arange(etaLoopMin, etaLoopMax+etaLoopStep, etaLoopStep)]

    Parray = [i[1] for i in data]

    NMaxarrray = [i[0] for i in data]

    etaArray  = np.arange(etaLoopMin, etaLoopMax+etaLoopStep, etaLoopStep)

    f, ax = plt.subplots()

    plt.plot(etaArray, Parray)
    plt.plot(etaArray, etaArray)

    plt.xlim((etaLoopMin,1))
    plt.ylim((0, 1))




    return([etaArray, Parray, NMaxarrray])


def get__N_at_P(g2,eta_H, etaLoopMin, etaLoopMax, x ):

    etaLoopArray = np.arange(etaLoopMin, etaLoopMax, 0.001)

    Narray = [get_N_x(g2, eta_H, etaLoop, x) for etaLoop in etaLoopArray]

    return [etaLoopArray, Narray]




#example_data = get__N_at_P(0.1,0.9, 0.5, 0.999, 0.99 )
#np.savetxt("nby1maxN_g0_1.csv", example_data, delimiter=",")
#plt.plot(example_data[0], example_data[1])

#print(get_N_x(0.2, 0.99, 0.9, 0.99))


#logtreeData = plot_P_and_N_max(500,  0.9, 0.999, 0.5, 0.995, 0.1, 0.005)
#np.savetxt("logtreeDatag20_1.csv", logtreeData, delimiter=",")
plot_Example(0.01,0.999,[0.99, 0.9,0.8],0.9,100, "nbyOne_example_g2_1")
plot_Example(0.1,0.999,[0.99, 0.9,0.8],0.9,100, "nbyOne_example_g2_1")


plot_Example_logTree(0.01,0.999,[ 0.99,0.9,0.8],0.9,100,"logTree_example_g2_1")
plot_Example_logTree(0.1,0.999,[ 0.99,0.9,0.8],0.9,100,"logTree_example_g2_1")

#generate_prob_grid(60, 0.1)

#fig, ax = plt.subplots()


#from TimeBinMultiplex import plot_P_N
#plot_P_N(10, 1, 0.99999, 0.9, 5, 0.1, 1)

#get_etaS_N(etaSMin, etaSMax, g2, etaH, x, etaSstep)


#analytic = get_etaS_N(0.5, 0.99, 0.1, 0.99, 0.9, 0.1)

#plt.plot(analytic[0], analytic[1])


#example = get_etaS_N(0.01, 1, 0.1, 0.90,0.90, 0.1)

#plt.plot(example[0], example[1])
#plot_Example_logTree(0.1,0.999,[0.8],0.9,100,"logTree_example_g2_1")
#plot_Example_logTree(0.1,0.999,[0.85],0.9,100,"logTree_example_g2_1")
plt.show()
