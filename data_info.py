#Eden McEwen
# help for pulling data files

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import pandas as pd

calib_f = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])
data_m = np.array([0, 56, 112, 152, 200, 250, 300, 356, 406, 470])
cold = []

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