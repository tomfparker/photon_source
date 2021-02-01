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
    return ((1/(np.cosh(r)))**2)*(np.tanh(r)**(2*n))



def probHerald(r, etaH, N, trunc):

    return sum([ (1 - (sum([((1 - etaH)**n)*prob(n, r) for n in range(0, trunc+1,1)]))) *  (sum([((1 - etaH)**n)*prob(n, r) for n in range(0, trunc+1,1)]))**(N-j) for j in range(1,N+1,1)])



def conditionProb_Hk(etaH, k):

      return sum([(etaH**n)*((1 - etaH)**(k-n))*binomial(k,n) for n in range(1,k+1,1)])



def conditionalProb_nk(n , etaLoop, etaS, k, j, N):

    return (((etaLoop**(N-j))*etaS)**n)*((1-(etaLoop**(N-j))*etaS)**(k-n))*binomial(k,n)


def jointProb_nH_j(n,j,N, etaH, etaS, etaLoop, r, trunc):

    return sum([prob(k,r)*conditionProb_Hk(etaH, N)*conditionalProb_nk(n, etaLoop, etaS, k, j, N) for k in range(1,trunc+1,1)])


#def jointProb_nH(n,N, etaH, etaS, etaLoop,r, trunc):

   # return sum([jointProb_nH_j(n,j,N, etaH, etaS, etaLoop, r, trunc)* sum([((1 - etaH) ** n1) * prob(n1, r) for n1 in range(0, trunc+1, 1)]) ** (N-j) for j in range(1,N+1,1)])

def jointProb_nH(n,N, etaH, etaS, etaLoop,r, trunc):

    return sum([jointProb_nH_j(n,j,N, etaH, etaS, etaLoop, r, trunc)* (1-sum([conditionProb_Hk(etaH, n1) * prob(n1, r) for n1 in range(1, trunc+1, 1)])) ** (N-j) for j in range(1,N+1,1)])


def jointSprob_SH(N, etaH, etaS, etaLoop, r, trunc):

   return sum([ jointProb_nH(n, N, etaH, etaS, etaLoop, r, trunc) for n in range(1, trunc+1,1)])


def conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc):

    return jointProb_nH(n,N, etaH, etaS, etaLoop,r, trunc)/probHerald(r, etaH, N, trunc)


def g2heralded(N, etaH, etaS, etaLoop, r ,trunc):

    return sum([n*(n-1)*conditional_nH(n , N, etaH, etaS, etaLoop, r, trunc) for n in range(1,trunc+1,1)])/(sum([n * conditional_nH(n, N, etaH, etaS, etaLoop, r, trunc) for n in range(1, trunc+1, 1)]))**2


###################################################################################################################

# Functions to get and vis data



def getJointG(N, etaH, etaS, etaLoop, rvals, trunc):

    joint  = [jointSprob_SH(N, etaH, etaS, etaLoop, r, trunc) for r in rvals]
    g2 = [g2heralded(N, etaH, etaS, etaLoop, r, trunc) for r in rvals]

    return [joint, g2]



def plot_joint_g2(N, etaHVals, etaS, etaLoopVals, rvals, trunc):

    f,ax = plt.subplots()

    for indx, eta in enumerate(etaLoopVals):

        tempdata = getJointG(N, etaHVals[indx], etaS, eta, rvals,trunc)

        plt.plot(tempdata[0], tempdata[1])

    plt.xlim((0, 1))
    plt.ylim((0, 0.5))

    return ax


#plot_joint_g2(1, [0.5, 0.75,0.999], 0.999,[0.5, 0.75,0.999], np.arange(0.0001,1,0.02), 4)



#Solve for g2 function for squeezing param
def get_squeezing(g2, N, etaH, etaS, etaLoop,  trunc):

    g2TempFunc= lambda rtemp :   g2heralded(N, etaH, etaS, etaLoop, rtemp, trunc) - g2

    return  fsolve(g2TempFunc, 0.1)


#get squeezing values over a range of source numbers
def get_getSqueezing_array(Nmax,  etaH, etaS, etaLoop,  trunc, g2, step ):

    return [get_squeezing(g2, int(N), etaH, etaS, etaLoop, trunc)[0] for N in np.arange(1,Nmax+1,step)]

def plot_P_N(Nmax, etaH, etaS, etaLoop, trunc, g2, step):

    sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step)
    num =  np.arange(1, Nmax + 1,step )
    joint = [jointSprob_SH(int(N), etaH, etaS, etaLoop, sq[indx], trunc) for  indx, N in enumerate(num)]


    f, ax = plt.subplots()

    plt.plot(num, joint)



    plt.xlabel("Source Number")
    plt.ylabel("Probability of Success")

    return [ax, num, joint]


def plot_Example_time(Nmax, etaH, etaS, etaLoop_Range, trunc, g2, step):

    sq = [get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step) for etaLoop in etaLoop_Range]
    num =  np.arange(1, Nmax + 1,step )

    joint = [[jointSprob_SH(int(N), etaH, etaS, etaLoopval, sq[index_2][indx], trunc) for indx, N in enumerate(num)] for index_2,  etaLoopval in enumerate(etaLoop_Range)]


    f, ax = plt.subplots()

    [plt.plot(num, j) for j in  joint]



    plt.xlabel("Time Bins")
    plt.ylabel("Probability of Success P(S,H)")

    plt.xlim((1, Nmax-1))
    plt.ylim((0, 1))

    return [ax, num, joint]


def getPSHconstg2(N,  etaH, etaS, etaLoop,  trunc, g2, step):

    sq = get_squeezing(g2, N, etaH, etaS, etaLoop,  trunc)

    joint = jointSprob_SH(int(N), etaH, etaS, etaLoop, sq, trunc)
    return joint

def getMaxNP(Nmax,  etaH, etaS, etaLoop,  trunc, g2):


    sq1 = get_squeezing(g2, 1, etaH, etaS, etaLoop, trunc)
    sq2 = get_squeezing(g2, 2, etaH, etaS, etaLoop, trunc)
    psArraytemp = [jointSprob_SH(int(1), etaH, etaS, etaLoop, sq1, trunc),jointSprob_SH(int(2), etaH, etaS, etaLoop, sq2, trunc)]


    for N in range(2,Nmax+1,1):



        if psArraytemp[N-1] > psArraytemp[N-2]:

            sq = get_squeezing(g2, N+1, etaH, etaS, etaLoop, trunc)
            psArraytemp.append(jointSprob_SH(int(N+1), etaH, etaS, etaLoop, sq, trunc))

        else:

            Nmax = N -1
            psMax = psArraytemp[N-2]

            break

    return [Nmax, psMax]





def getMaxRange(etaLoopvals, g2, NMax, etaH, etaS, trunc):


    data= [getMaxNP(NMax,  etaH, etaS, etaLoop,  trunc, g2) for etaLoop in etaLoopvals]

    pMaxArray = [dataval[1] for dataval in data]

    NmaxArray = [dataval[0] for dataval in data]

    return[pMaxArray, NmaxArray]


def g2_modeNum(Nmax, eta_s, eta_h,etaLoop, trunc, initialg2 ):
    """
      Get heralded g2 as a function of the number of time bins for a constant squeezing
       """

    squeezing = get_squeezing(initialg2, 1, eta_h, eta_s, etaLoop,  trunc)

    Narray = np.arange(2,Nmax,1)

    g2Vals = [g2heralded(N, eta_h, eta_s, etaLoop, squeezing, trunc) for N in Narray]

    return([Narray, g2Vals])

def plot_g2_modeNum(Nmax, eta_s, eta_h, etaLoop , trunc, initialg2):
    """
          Get heralded g2 as a function of the number of time bins for a constant squeezing
           """

    [narray, g2array]=g2_modeNum(Nmax, eta_s, eta_h, etaLoop, trunc, initialg2)

    f, ax = plt.subplots()

    plt.plot(narray, g2array)

    plt.xlabel("Source Number")
    plt.ylabel("g$^{(2)}$")


#plot_g2_modeNum(40, 0.9, 0.9, 0.988 , 4, 0.2)











#etaLoopArray = [0.5,0.55,0.6,0.65, 0.7, 0.75, 0.8, 0.85,0.9, .95]
#example  = getMaxRange(etaLoopArray, 0.3, 40, 0.95, 0.99, 4)


#print(example)

#fig, ax = plt.subplots()

#plt.plot(etaLoopArray, example[0])
#fig2, ax2 = plt.subplots()

#plt.scatter(etaLoopArray, example[1])



   # psMax = []

   # N = 1

   # if N ==1:

    #    psArraytemp.append(jointSprob_SH(int(1), etaH, etaS, etaLoop, sq, trunc))



    #sq = get_squeezing(g2, N, etaH, etaS, etaLoop, trunc)



   # sq = get_getSqueezing_array(Nmax, etaH, etaS, etaLoop, trunc, g2, step)
   # num = np.arange(1, Nmax + 1, step)
   # joint = [jointSprob_SH(int(N), etaH, etaS, etaLoop, sq[indx], trunc) for indx, N in enumerate(num)]

   # f, ax = plt.subplots()

    #plt.plot(num, joint)


#testFunc = lambda N: -1*getPSHconstg2(int(N),  0.999, 0.999, 0.7,  4, 0.1, 1)




#test =  minimize(testFunc, 3).fun

#testFunc2 = lambda N: getPSHconstg2(int(N),  0.999, 0.999, 0.7,  4, 0.1, 1) - 0.108
#test2 = fsolve(testFunc2, 4)

#print(test)
#print(test2)

 #plot_P_N(Nmax, etaH, etaS, etaLoop, trunc, g2, step)

######!!!!!!!!!!!!!!!!!###############
plot_Example_time(20, 0.9, 0.9, [0.988], 4, 0.2, 1)



#plot_P_N(20, 0.95, 0.5, 0.9, 4, 0.05, 1)
#plot_P_N(10, 0.95, 0.99999, 0.5, 5, 0.1, 1)
#plot_P_N(50, 0.95, 0.99999, 0.99, 5, 0.1, 3)

#print([g2heralded(int(num), 0.999,0.999,0.9, 0.1 ,3)for num in [1,2,5,20]])


#sq=get_squeezing(0.1, 1, 0.99, 0.999, 0.999, 3)
#print(sq)
#print(sq)

#print(g2heralded(1, 0.999,0.999,0.999, sq ,3))

#exampleArray = get_getSqueezing_array(10,  0.999, 0.999, 0.999, 4, 0.1 )

#print(exampleArray)




#plot_joint_g2(1, [0.5, 0.75,0.999], 0.999,[0.5, 0.75,0.999], np.arange(0.0001,1,0.02), 4)
#plot_joint_g2(10, [0.5, 0.75,0.999], 0.999,[0.5, 0.75,0.999], np.arange(0.0001,1,0.02), 4)

#example = getJointG(1, 0.5, 0.5, 0.5, np.arange(0.0001,1,0.02),4)
#example2 = getJointG(5, 0.5, 0.5, 0.5, np.arange(0.0001,1,0.02),4)
#example3 = getJointG(10, 0.5, 0.5, 0.5, np.arange(0.0001,1,0.02),4)


#fig, ax = plt.subplots()
#plt.plot(example[0], example[1])
#plt.plot(example2[0], example2[1])
#plt.plot(example3[0], example3[1])

#plt.xlim((0, 1))
#plt.ylim((0, 0.5))


#plt.show()


#squeez = np.arange(0.0001,0.7,0.01)


#joint  = [jointSprob_SH(1, 0.999, 0.999, 0.9999, r, 10) for r in squeez]
#g2 = [g2heralded(1, 0.999, 0.999, 0.9999, r, 10) for r in squeez]

#joint10  = [jointSprob_SH(10, 0.999, 0.999, 0.9999, r, 10) for r in squeez]
#g210 = [g2heralded(10, 0.999, 0.999, 0.9999, r, 10) for r in squeez]

#print(joint)
#print(g2)


#fig, ax = plt.subplots()
#plt.plot(joint, g2)
#plt.plot(joint10, g210)

#plt.show()


#def g2example(r):
#    return g2heralded(10, 0.999, 0.999, 0.999, r, 3)-0.1

#rex = fsolve(g2example, 0.1)

#print(rex)

#print(g2heralded(10, 0.999, 0.999, 0.999, rex, 3))

#g2heralded(1., 0.999, 0.999, 0.999, 0.1, 3)

#def jointp(N):
  #  return jointSprob_SH(int(N),0.2, 0.2, 0.2, 0.1, 4)

#def dertest(N):

  #  return derivative(jointp, N)

#num = fsolve(dertest, 100)

#print(num)

#Num = np.arange(1,40,1)

#joint  = [jointSprob_SH(int(r), 0.1, 0.1, 0.1, 0.1, 4) for r in Num]

#fig, ax = plt.subplots()
#plt.plot(Num, joint)

#print(jointSprob_SH(100,0.5, 0.5, 0.5, 0.1, 4))

#def plot_P_N(Nmax, etaH, etaS, etaLoop, trunc, g2, step):
#plot_P_N(40, 1, 0.99999, 0.9, 5, 0.1, 1)
plt.show()