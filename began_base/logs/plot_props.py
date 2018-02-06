import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12) 
from matplotlib.pyplot import cm
import numpy as np
import os
import pdb
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default=None) 
args = parser.parse_args()

xp0s = np.loadtxt(os.path.join(args.log_dir, 'xp0s.txt'))
gp0s = np.loadtxt(os.path.join(args.log_dir, 'gp0s.txt'))

log_step = 100
x_vals = [log_step * i for i in range(len(xp0s))]

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

plt.figure(figsize=(8, 4))
plt.title('Proportion class 0, estimated')
plt.plot(x_vals, xp0s, color='blue', alpha=0.3, linestyle=':', linewidth=1.0)
plt.plot(x_vals, movingaverage(xp0s, 10), color='blue', alpha=1, linestyle=':', linewidth=2.0, label='x')
plt.plot(x_vals, gp0s, color='green', alpha=0.3, linestyle='-', linewidth=2.0)
plt.plot(x_vals, movingaverage(gp0s, 10), color='green', alpha=1, linestyle='-', linewidth=2.0, label='g')
plt.axhline(y=0.5, color='gray', linestyle='-', alpha=0.2)
plt.xlabel('Iteration')
plt.ylabel('Prop 0, est.')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'prop0.png'))
os.system((
    'echo $PWD -- {} | mutt momod@utexas.edu -s '
    '"prop0_{}" -a "{}/prop0.png"').format(
        args.log_dir, args.log_dir, args.log_dir))
print 'Emailed prop0.png'
