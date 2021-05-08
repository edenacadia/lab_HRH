#Eden McEwen
# help for pulling data files

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd
import scipy.optimize as opt

# check how bottles changed over measurements
d_water = 997e3 # g/m³
# 0 - bottle weight
# 1 - neck volume
# 2 - total volume

b12 = [[458, 454, 454, 456],
       [464, 472, 474, 472],
       [514, 518, 518, 518]]

b5 = [[452, 452, 450],
      [472, 476, 474],
      [518, 518, 518]]

b2 = [[450, 448, 450],
      [472, 476, 472],
      [520, 520, 520]]

mean_neck = np.average(b12[1])
mean_top = np.average(b12[2])

calib_f = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])
data_m = np.array([0, 56, 112, 152, 200, 250, 300, 356, 406, 470])
cold = np.array([0, 54, 118, 150, 206, 254, 306, 356, 402, 452, 470])

def csv_fq(index):
    #given the index of a file
    # max freq is 8000
    return (8000/1025)*(1025 - index)

def csv_fq_scale(index):
    return (8000/1025)*index

def csv_indx(fq):
    #given the index of a file
    # max freq is 8000
    return (1025/8000)*(8000 - fq)

def f_resonance(csv_f):
    spec_np = np.genfromtxt(csv_f, delimiter=',')
    spec_np = spec_np[:,1:]
    #time average
    time_svg = np.average(spec_np, axis = 1)
    idx = np.argmax(time_svg)
    mean_s = np.average(spec_np[idx]) # mean value for this cut
    std_s = np.std(spec_np[idx]) # standard deviation
    #take indexes of relevant times
    idxs_filter = np.argwhere(spec_np > mean_s + std_s) # indexes above this cut
    spec_flted = spec_np[:, np.unique(idxs_filter[:,1])] # filtered on time idexes
    #find final freq
    time_fvg = np.average(spec_flted, axis = 1)
    max_ffeq = np.argmax(time_fvg) # index of max peak
    max_feq = csv_fq(max_ffeq) # freq of max peak
    peak_qual = signal.peak_widths(time_fvg, [max_ffeq])
    return [max_feq, peak_qual[0][0]]

def f_max(csv_f):
    spec_np = np.genfromtxt(csv_f, delimiter=',')
    spec_np = spec_np[:,1:-10]
    #time average
    time_svg = np.average(spec_np, axis = 1)
    idx = np.argmax(time_svg)
    mean_s = np.average(spec_np[idx]) # mean value for this cut
    std_s = np.std(spec_np[idx]) # standard deviation
    #take indexes of relevant times
    idxs_filter = np.argwhere(spec_np > mean_s) # indexes above this cut
    spec_flted = spec_np[:, np.unique(idxs_filter[:,1])] # filtered on time idexes
    #find final freq
    time_fvg = np.average(spec_flted, axis = 1)
    max_ffeq = np.argmax(time_fvg) # index of max peak
    return max_ffeq


# Chi squared goodness of fit
def hand_chisquared(ms, exp, unc):
    ms = np.array(ms)
    exp = np.array(exp)
    return np.sum(((ms - exp)/unc)**2)

def hand_chisquared_r(ms, exp, unc, df):
    ms = np.array(ms)
    exp = np.array(exp)
    return np.sum(((ms - exp)/unc)**2)/df

### FITS

#### Ply fit:

def Poly(x, a, b):
    return b*x**a

def Poly_est(x, a, b):
    #returns y estimates
    y_est = [Poly(xi, a, b) for xi in x]
    return y_est

def Poly_fit(x, y):
    # returns [slope, intercept]
    # the optimized curve returns the optimized abc values for Gaussian
    # [height, position of peak center (mean), standard deviation]
    return opt.curve_fit(Poly, x, y)

# Linear Fits

def Linear_tot(x, y):
    a, b = Linear_Fit(x, y)
    a_err, b_err = Linear_err(x, y, a, b)
    return np.array([a, a_err, b, b_err])

def Linear(x, a, b):
    return a * x + b

def Linear_Fit(x, y):
    # returns [slope, intercept]
    # the optimized curve returns the optimized abc values for Gaussian
    # [height, position of peak center (mean), standard deviation]
    popt, pcov = opt.curve_fit(Linear, x, y)
    return popt[0], popt[1]
    
def Linear_fit(x, y):
    # returns fit
    return opt.curve_fit(Linear, x, y)
    
def Linear_est(x, a, b):
    #returns y estimates
    y_est = [Linear(xi, a, b) for xi in x]
    return y_est

#### Uncertainties of fit

def Linear_err(x, y, a, b):
    cu = com_uncert(x, y, a, b)
    b_err = err_intercept(x, y, cu)
    a_err = err_gradient(x, y, cu)
    return a_err, b_err

# Main TWO

def err_intercept(x, y, cu):
    # calculates error in liner fit intercepts
    x = np.array(x)
    dlt = delta(x)
    return cu * np.sqrt(np.sum(x**2)/dlt)

def err_gradient(x, y, cu):
    # calculates error in gradient
    x = np.array(x)
    dlt = delta(x)
    return cu * np.sqrt(x.shape[0]/dlt)

# Helper Functions

def com_uncert(x, y, a, b):
    # calculates the common uncertainty
    x = np.array(x)
    y = np.array(y)
    diff_array = (y - a * x - b)**2
    diff = np.sum(diff_array)
    return np.sqrt((1/(x.shape[0]-2))*diff)

def delta(x):
    # helper function
    x = np.array(x)
    return x.shape[0]*np.sum(x**2) - np.sum(x)**2