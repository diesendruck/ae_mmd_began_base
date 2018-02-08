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
import json

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='logs/test') 
parser.add_argument('--results_file', type=str, default='results.json') 
args = parser.parse_args()

# xp0s = np.loadtxt(os.path.join(args.log_dir, 'xp0s.txt'))
# gp0s = np.loadtxt(os.path.join(args.log_dir, 'gp0s.txt'))

results = np.array(json.load(open('{}/{}'.format(args.log_dir, args.results_file), 'r')))
kept = results.shape[0] / 2

fig = plt.figure(figsize=(4, 4))

plt.subplot(1,2,1)
data = results[-kept:, 11:14]
means = data.mean(0)
sds = data.std(0)
plt.bar(range(3), means.tolist(), yerr=sds.tolist(), align='center', tick_label=['0', '1', '5'], color='.75', capsize=12.)
plt.ylim(0, 1)
plt.tight_layout()

plt.subplot(1,2,2)
data = results[-kept:, 5:8]
means = data.mean(0)
sds = data.std(0)
plt.bar(range(3), means.tolist(), yerr=sds.tolist(), align='center', tick_label=['0', '1', '5'], color='.75', capsize=12.)
plt.ylim(0, 1)
plt.tight_layout()

plt.savefig(os.path.join(args.log_dir, 'source_and_target_barplot.png'))

# os.system((
#     'echo $PWD -- {} | mutt guywcole@utexas.edu -s '
#     '"prop0_{}" -a "{}/source_and_target_barplot.png"').format(
#         args.log_dir, args.log_dir, args.log_dir))
# print 'Emailed prop0.png'


