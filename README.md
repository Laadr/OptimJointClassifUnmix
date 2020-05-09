# Matrix Cofactorization for Joint Representation Learning and Supervised Classification -- Application to Hyperspectral Image Analysis

This code implements the method described in

Lagrange, A., Fauvel, M., May, S., Bioucas-Dias, J., & Dobigeon, N. (2020). Matrix cofactorization for joint representation learning and supervised classification–Application to hyperspectral image analysis. Neurocomputing, 385, 132-147.

# Disclaimer

Feel free to contact me if you encounter any problem with the code.

# Usage
An exemple of how to use the code is provided in the file runExpl.py.

## Class initialization inputs

`JointClassifUnmixModel(lambdas=[1.,1.,1.,0.1,0.],lambda_q=0.1,sigma=0.01,epsilon=0.01,tol=1e-4)`

- lambdas:  hyperparamaters weighting the terms of the objective function: 0)Y-MA 1)A-BZ 2)CD-QZD 3)||A||\_1 4)||Z||\_TV (cf paper)
- lambda_q: hyperparameter weighting the penalization of matrix Q, used only when classif_loss is 'cross-entropy'
- sigma:    parameter in weight of adaptative TV (cf paper)
- epsilon:  parameter to make the TV-norm gradient-Lipschitz (cf paper)
- tol:      tolerance used to check convergence


## I/O of optimization method
NB: be careful with labels: label 0 in input is equivalent to a unknown label

`JointClassifUnmixModel.optimize(Y,R,K,C,labels,maxIter,classif_loss='cross-entropy',M=None,maxInitIter=1000,adaptTV=False,printEnable=False,initClassif=False,initFilePath=None,srand=0)`

Inputs:
  - Y:            image (spatial dimension 1 x spatial dimension 2 x spectral dimension=d).
  - R:            number of endmembers.
  - K:            number of clusters.
  - C:            number of classes.
  - labels:       label matrix where label 0 corresponds to unknown class to be estimated.
  - maxIter:      maximum number of iterations of PALM algorithm.
  - classif_loss: loss function to use for classification term ('quadratic'|'cross-entropy')
  - M:            endmembers matrix with R spectra (dxR).
  - maxInitIter:  maximum number of iterations for algorithm initializing A.
  - adaptTV:      (bool) use TV weighted by the gradient of the penchromatic image.
  - printEnable:  (bool) print information when True. Default False.
  - initClassif:  (bool) use random forest to initialize classes.
  - initFilePath: (bool) path to store results of initialization, (if maxInitIter=0, path to load initialization)
  - srand:        random seed.

Ouput:
  - A_est:            estimated abundance matrix.
  - M:                endmember matrix.
  - B_est:            estimated cluster centroid matrix.
  - Z_est:            estimated cluster matrix.
  - Q_est:            estimated interaction matrix.
  - C_est:            estimated classification matrix.
  - objectiveFct_tab: historic of the value of the terms of the objective function.
  - timeInit:         processing time due to initialization.

# Authors
Author: Adrien Lagrange (ad.lagrange@gmail.com)

Under Apache 2.0 license
