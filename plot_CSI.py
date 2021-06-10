import time
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from numpy.random import seed
from matplotlib import pyplot as plt
import matplotlib

from utils import *

def exp1():
	generated_samples =np.asarray(pickle.load(open('dataset/reconstructed/SVAE-X-16.pickle','rb'))).reshape(-1,120)

	dataset = np.asarray(pickle.load(open('dataset/EXP1.pickle','rb')))
	X_train, y_train, X_tst, y_tst = dataset

	for target in range(16):
		plt.figure()
		for x_g in generated_samples[y_train==target]:
			#font = {'family' : 'Verdana','weight' : 'normal','size'   : 12}	
			#matplotlib.rc('font', **font)
			matplotlib.rcParams['pdf.fonttype'] = 42
			matplotlib.rcParams['ps.fonttype'] = 42	
			plt.title('Location $p_{%d}$ (reconstructed)'%(target+1) , fontsize=20)
			plt.xlabel('CSI Index', fontsize=18)
			plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
			plt.axis([0, 120, 0, 1])
			plt.grid(True)

			plt.plot(x_g)
		plt.savefig('visualizations/EXP1/VAE-exp1-p%d.png'%(target+1), dpi=400)
		plt.savefig('visualizations/EXP1/VAE-exp1-p%d.eps'%(target+1), dpi=1)  
		plt.close() 

def exp2():
	generated_samples =np.asarray(pickle.load(open('dataset/reconstructed/SVAE-X-14.pickle','rb'))).reshape(-1,120)

	dataset = np.asarray(pickle.load(open('dataset/EXP2.pickle','rb')))
	X_train, y_train, X_tst, y_tst = dataset

	for target in range(14):
		plt.figure()
		for x_g in generated_samples[y_train==target]:
			#font = {'family' : 'Verdana','weight' : 'normal','size'   : 12}	
			#matplotlib.rc('font', **font)
			matplotlib.rcParams['pdf.fonttype'] = 42
			matplotlib.rcParams['ps.fonttype'] = 42	
			plt.title('Location $p_{%d}$ (reconstructed)'%(target+1) , fontsize=20)
			plt.xlabel('CSI Index', fontsize=18)
			plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
			plt.axis([0, 120, 0, 1])
			plt.grid(True)

			plt.plot(x_g)
		plt.savefig('visualizations/EXP2/VAE-exp2-p%d.png'%(target+1), dpi=400)
		plt.savefig('visualizations/EXP2/VAE-exp2-p%d.eps'%(target+1), dpi=1)  
		plt.close() 



if __name__ == '__main__':
	exp1()
	exp2()

