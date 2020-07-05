#!/usr/bin/python3
# -*- coding: utf-8 -*-


import scipy as sp
import time
import scipy.io
import scipy.stats
from sklearn.metrics import cohen_kappa_score, accuracy_score
from matplotlib import pyplot as plt
import modelOptimJointClassifUnmix


## parameters
sp.random.seed(0)
SNR          = 30
corrupRatio  = 0.

classif_loss = 'cross-entropy' #'quadratic' or 'cross-entropy'
K            = 10 # nb of clusters
lambdas      = [100.,1.,1.,0.1,0.1]
lambda_q     = 0.1
epsilon      = 0.01
adaptTV      = True
sigma        = 0.01
initClassif  = False

tol         = 1e-4
maxInitIter = 1000
maxIter     = 1000

## load data
print('Load data...')
data   = sp.load('./semisynthAisa.npz')
labels = data['labels']
im     = data['im']
M      = data['M']
R      = M.shape[1]
A_true = data['A']
del data
print('done')

## reindex labels
C = len(sp.unique(labels))
for i,c in enumerate(sp.unique(labels)):
  labels[labels==c] = i+1
fullLabels = sp.copy(labels)

## prepare mask
# mask = sp.ones((labels.shape[0],labels.shape[1]),dtype=bool)
maskTrain = sp.zeros((labels.shape[0],labels.shape[1]),dtype=bool)
maskTrain[:25,:25]      = True
maskTrain[:25,140:165]  = True
maskTrain[:25,-25:]     = True
maskTrain[-25:,:25]     = True
maskTrain[-25:,140:165] = True
maskTrain[-25:,-25:]    = True
maskTest = ~maskTrain

labels[maskTest] = 0
labelsTest = fullLabels.copy()
labelsTest[maskTrain] = 0


## add noise
noiseStd =sp.sqrt(sp.mean(im)**2 * 10**(-SNR/float(10)))
noise = noiseStd*sp.random.randn(im.shape[0],im.shape[1],im.shape[2])
print("SNR:",10*sp.log10(sp.mean(im)**2/noiseStd**2))
im += noise


## prepare labels for training set
labels = sp.copy(fullLabels)
labels[maskTest] = 0

if corrupRatio !=  0.:
  prob = sp.empty((C))
  for c in range(C):
    prob = corrupRatio/float(C-1) * sp.ones((C))
    prob[c] = 1. - corrupRatio
    labels[labels==c+1] = sp.random.choice(C,sp.sum(labels==c+1),p=prob)+1


## Cofact model
optimScheme = modelOptimJointClassifUnmix.JointClassifUnmixModel(lambdas=lambdas,lambda_q=lambda_q,epsilon=epsilon,sigma=sigma,tol=tol)

print('Start cofactorization problem...')
start_time = time.time()

[A_est, M_est, B_est, Z_est, Q_est, classif_est, objectiveFct, timeInit] = optimScheme.optimize(im+noise,R,K,C,labels,maxIter,M=M,maxInitIter=maxInitIter,adaptTV=adaptTV,initClassif=initClassif,printEnable=True,classif_loss=classif_loss,srand=0)
processingTime = time.time() - start_time
print('Processing time: {} seconds'.format(processingTime))


## evaluate classif
classif_est_2d = sp.argmax(classif_est.reshape((C,im.shape[0], im.shape[1])), axis=0) + 1
print( 'OA:{}'.format(accuracy_score(fullLabels[maskTest],classif_est_2d[maskTest])) )
print( 'Kappa:{}'.format(cohen_kappa_score(fullLabels[maskTest],classif_est_2d[maskTest])) )


## plot results
Z_2d  = sp.argmax(Z_est.reshape((K,im.shape[0], im.shape[1])), axis=0)

f, ax = plt.subplots(2,2)
ax[0,0].imshow(fullLabels,interpolation='none')
ax[0,0].set_title('True class labels')
ax[1,0].imshow(Z_2d,interpolation='none')
ax[1,0].set_title('Cluster labels')
labPrint = labels.copy().astype(float)
labPrint[labPrint==0] = sp.nan
ax[0,1].imshow(labPrint,interpolation='none')
ax[0,1].set_title('Training set (corrupted at {}%)'.format(100*corrupRatio))
ax[1,1].imshow(classif_est_2d,interpolation='none')
ax[1,1].set_title('Estimated labels')


ll = 5
f, ax = plt.subplots(3,ll,sharex=True,sharey=True)
for n in range(A_true.shape[0]):
  ax[n//ll,n%ll].imshow(A_true[n,:,:],interpolation='none',cmap=plt.get_cmap('viridis'), vmin=0., vmax=1.)
  ax[n//ll,n%ll].set_title("Abund. "+str(n)+" (A_true)")

f, ax = plt.subplots(3,ll,sharex=True,sharey=True)
for n in range(A_est.shape[0]):
  ax[n//ll,n%ll].imshow(A_est[n,:].reshape((im.shape[0], im.shape[1])),interpolation='none',cmap=plt.get_cmap('viridis'), vmin=0., vmax=1.)
  ax[n//ll,n%ll].set_title("Abund. "+str(n)+" (A)")

plt.show()

