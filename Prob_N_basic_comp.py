# This plots how the joint probability of detection scales with the number of sources (or time bins) that are multiplexed.
# This should be for a constant g2 and herald detection effciency

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

