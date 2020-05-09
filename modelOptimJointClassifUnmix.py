# -*- coding: utf-8 -*-

# import sys 
# import os

import scipy as sp
# from scipy import signal
import scipy.linalg as splin
# from matplotlib import pyplot as plt
# import multiprocessing as mp
import time
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


class JointClassifUnmixModel(object):
  """
    Optimization scheme to perform joint unmixing and classification
    NB: be careful with labels: label 0 in input is equivalent to a unknown label

    lambdas:  hyperparamaters weighting the terms of the objective function: 0)Y-MA 1)A-BZ 2)CD-QZD 3)||A||_1 4)||Z||_TV (cf paper)
    lambda_q: hyperparameter weighting the penalization of matrix Q, used only when classif_loss is 'cross-entropy'
    sigma:    parameter in weight of adaptative TV (cf paper)
    epsilon:  parameter to make the TV-norm gradient-Lipschitz (cf paper)
    tol:      tolerance used to check convergence
  """
  def __init__(self,lambdas=[1.,1.,1.,0.1,0.],lambda_q=0.1,sigma=0.01,epsilon=0.01,tol=1e-4):
    super(JointClassifUnmixModel, self).__init__()
    self.lambdas  = sp.asarray(lambdas,dtype=float) # 0)Y-MA 1)A-BZ 2)CD-QZD 3)||A||_1 4)||Z||_TV
    self.lambda_q = float(lambda_q) #||Q||_2
    self.sigma    = sigma
    self.epsilon  = epsilon
    self.tol      = tol

  def optimize(self,Y,R,K,C,labels,maxIter,classif_loss='cross-entropy',M=None,maxInitIter=1000,adaptTV=False,printEnable=False,initClassif=False,initFilePath=None,srand=0):
    """
      Run PALM algorithm to optimize the objective function.
      Input:
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
    """

    # init random number generator seed
    sp.random.seed(srand)

    # Init parameters
    (heightImg, widthImg, d) = Y.shape
    P                        = heightImg*widthImg

    # Compute weight for spatial regul
    if self.lambdas[4]!=0.:
      print('Spatial regularization enabled.')

      if adaptTV:
        # create panchromatic image
        panchro = sp.sum(Y/sp.mean(sp.mean(Y.astype(float),axis=0,keepdims=True),axis=1,keepdims=True),axis=2)
        panchro = (panchro - panchro.min())/(panchro.max() - panchro.min())

        # compute horizontal and vertical gradient of panchro
        panGradU = sp.zeros((heightImg, widthImg),dtype=float)
        panGradR = sp.zeros((heightImg, widthImg),dtype=float)
        panGradU[1:,:] = -sp.diff(panchro,axis=0) #Z_i-1,j - Z_i,j
        panGradR[:,:-1] = sp.diff(panchro,axis=1) #Z_i,j+1 - Z_i,j

        W = 1. / (self.sigma + sp.sqrt(panGradU**2 + panGradR**2))
        W *= (P/W.sum())
        Wmax = W.max()
        del panGradU,panGradR

      else:
        W = sp.ones((heightImg, widthImg),dtype=float)
        Wmax = W.max()

    # Reshape data
    Y      = Y.reshape((P,d)).T
    labels = labels.reshape((P))
    P      = Y.shape[1]      

    # Init variables
    timeInit = 0.
    if maxInitIter != 0.:
      # Initialize with sparse dictionary learning
      print('Initialization...')
      start_time = time.time()
      if M is not None:
        A_est, B_est, Z_est = self.initVar(Y,R,K,maxInitIter,M,printEnable=printEnable,srand=0)
      else:
        candByClass = int(R/C)
        A_est, M, B_est, Z_est,ind = self.initVar2(Y,labels.ravel(),candByClass,C,K,maxInitIter,printEnable=printEnable,srand=srand)
        R = M.shape[1]
        if initFilePath is not None:
          sp.savez(initFilePath,R=R,M=M,A_est=A_est, B_est=B_est, Z_est=Z_est,ind=ind)
      timeInit = time.time() - start_time
      print("...done ({} seconds)".format(timeInit))

    else:
      if initFilePath is not None:
        print('Load initialization from {}...'.format(initFilePath))
        M     = sp.load(initFilePath)['M']
        R     = sp.load(initFilePath)['R']
        A_est = sp.load(initFilePath)['A_est']
        B_est = sp.load(initFilePath)['B_est']
        Z_est = sp.load(initFilePath)['Z_est']
        print("...done")

      else:
        print('Basic initialization...')
        R = M.shape[1]

        A_est = sp.maximum(0.,sp.dot(splin.pinv(M),Y))
        clustLabel = KMeans(n_clusters=K,n_init=10,n_jobs=-1).fit_predict(A_est.T)
        B_est = sp.empty((R,K))
        for i in range(K):
            B_est[:,i] = sp.mean(A_est[:,clustLabel==i],axis=1)

        Z_est = sp.random.rand(K,P)
        for k in range(K):
          Z_est[k,clustLabel==k]=1.
          Z_est[k,clustLabel!=k]=0.
        print("...done")

    # Rescale weights according to problem size
    lambdas        = sp.asarray([self.lambdas[0]/(d*sp.amax(Y)**2),self.lambdas[1],self.lambdas[2],self.lambdas[3],self.lambdas[4]],dtype=float)
    lambda_q   = self.lambda_q*P/C

    # Separate labeled from non labeled samples
    indUSpl = sp.where(labels == 0)[0]
    indLSpl = sp.where(labels != 0)[0]

    # Dumb initialization
    Q_est = sp.random.rand(C,K)
    C_est = sp.random.rand(C,P)
    C_est = C_est/sp.sum(C_est,axis=0)
    for c in range(C):
      C_est[c,labels==(c+1)]=1.
      C_est[c,labels!=(c+1)]=0.
    objectiveFct_tab = sp.empty((5,maxIter+2))
    objectiveFct_tab[:,:2] = [1.,2.]

    # Initialize classif
    if initClassif:
      print("classif init !")
      C_est[:,indUSpl] = self.initClassif(Y,A_est,M,B_est,Z_est,labels,C, n_neighbors=10, printEnable=printEnable,srand=0)
      Q_est = sp.dot(C_est,splin.pinv(Z_est))

    # Precomputation to save computational burden
    MTM = sp.dot(M.T,M)
    MTY = sp.dot(M.T,Y)
    D = P/float(len(indUSpl))*sp.ones((P),dtype=float)
    for c in range(C):
      D[labels==(c+1)] = P/float(C_est[c,indLSpl].sum())
    DUmax = sp.amax(D[indUSpl])
    D_all = sp.unique(D)
    C_2d = C_est.reshape((C,heightImg, widthImg))
    if lambdas[4]!=0.:
      gradU_C = sp.zeros((C,heightImg, widthImg),dtype=float)
      gradR_C = sp.zeros((C,heightImg, widthImg),dtype=float)
      gradU_C[:,1:,:] = -sp.diff(C_2d,axis=1) #C_i-1,j - C_i,j
      gradR_C[:,:,:-1] = sp.diff(C_2d,axis=2) #C_i,j+1 - C_i,j


    ## Optimization
    print( "Log: start optimization..." )
    iterNb = 2
    if classif_loss == 'quadratic':
      while (iterNb <= maxIter) and ( iterNb <=5 or (objectiveFct_tab[4,iterNb-2]-objectiveFct_tab[4,iterNb-1])/objectiveFct_tab[4,iterNb-2]>self.tol):

        # Optim with respect to A
        coeffA = lambdas[0]*MTM + lambdas[1]*sp.eye(R)
        Aupdate = A_est - 1/(1.01*splin.norm(coeffA)) * ( sp.dot(coeffA,A_est) - lambdas[0]*MTY - lambdas[1]*sp.dot(B_est,Z_est) )
        A_est = sp.maximum(0, Aupdate - self.lambdas[3]/(1.01*splin.norm(coeffA))) # max elementwise 

        # Optim with respect to B
        coeffB = lambdas[1]*sp.dot(Z_est,Z_est.T)
        Bupdate = B_est - 1/(1.01*splin.norm(coeffB)) * ( sp.dot(B_est,coeffB) - lambdas[1]*sp.dot(A_est,Z_est.T) )
        B_est = sp.maximum(0, Bupdate) # max elementwise

        # Optim with respect to Z
        QTQ = sp.dot(Q_est.T,Q_est)
        coeffZ = lambdas[1]*sp.dot(B_est.T,B_est)
        LZ = sp.asarray([splin.norm(coeffZ + lambdas[2]*d*QTQ) for d in D_all]).max()
        Zupdate = Z_est - 1/(1.01*LZ) * ( sp.dot(coeffZ,Z_est) + lambdas[2]*sp.dot(QTQ,D[sp.newaxis,:]*Z_est) - lambdas[1]*sp.dot(B_est.T,A_est) - lambdas[2]*sp.dot(Q_est.T,D[sp.newaxis,:]*C_est) )
        thresh_Z = sp.amax( (sp.cumsum( sp.sort(Zupdate,axis=0)[::-1,:],axis=0 )-1) / sp.arange(1,K+1)[:,sp.newaxis] , axis=0)
        Z_est = sp.maximum(0, Zupdate - thresh_Z) # max elementwise

        # Optim with respect to Q
        coeffQ = lambdas[2]*sp.dot(D[sp.newaxis,:]*Z_est,Z_est.T)
        Q_est = Q_est - 1/(1.01*splin.norm(coeffQ)) * ( sp.dot(Q_est,coeffQ) - lambdas[2]*sp.dot(C_est,D[:,sp.newaxis]*(Z_est.T)) )

        # Optim with respect to C
        if lambdas[4]!=0.:
          LC = lambdas[2]*DUmax + lambdas[4]*sp.sqrt(8.)*Wmax/self.epsilon
          Wgrad = W / sp.sqrt(sp.sum(gradR_C**2,axis=0)+sp.sum(gradU_C**2,axis=0)+self.epsilon**2)

          gradTV = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradTV[:,-1,:]  = Wgrad[-1,:]*gradU_C[:,-1,:]
          gradTV[:,:,0]   = Wgrad[:,0]*gradR_C[:,:,0]
          gradTV[:,:-1,:] += sp.diff(Wgrad*gradU_C,axis=1)
          gradTV[:,:,1:]  += -sp.diff(Wgrad*gradR_C,axis=2)
          Cupdate = C_est[:,indUSpl] - 1/(1.01*LC) * ( lambdas[2]*C_est[:,indUSpl]*D[sp.newaxis,indUSpl] - lambdas[2]*sp.dot(Q_est,D[sp.newaxis,indUSpl]*Z_est[:,indUSpl]) + lambdas[4]*gradTV.reshape((C,heightImg*widthImg))[:,indUSpl] )
        else:
          Cupdate = C_est[:,indUSpl] - 1/(1.01*DUmax) * ( C_est[:,indUSpl]*D[sp.newaxis,indUSpl] - sp.dot(Q_est,D[sp.newaxis,indUSpl]*Z_est[:,indUSpl]) )

        thresh_C = sp.amax( (sp.cumsum( sp.sort(Cupdate,axis=0)[::-1,:],axis=0 )-1) / sp.arange(1,C+1)[:,sp.newaxis] , axis=0)
        C_est[:,indUSpl] = sp.maximum(0, Cupdate - thresh_C) # max elementwise

        if lambdas[4]!=0.:
          C_2d = C_est.reshape((C,heightImg, widthImg))
          gradU_C = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradR_C = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradU_C[:,1:,:] = -sp.diff(C_2d,axis=1) #C_i-1,j - C_i,j
          gradR_C[:,:,:-1] = sp.diff(C_2d,axis=2) #C_i,j+1 - C_i,j

        # # Optim with respect to M
        # AAT = sp.dot(A_est,A_est.T)
        # Mupdate  = M - 1/(1.01*splin.norm(AAT)) * ( sp.dot(M,AAT) - sp.dot(Y,A_est.T) )
        # M    = sp.maximum(0, Mupdate) # max elementwise
        # M /= splin.norm(M,axis=0,keepdims=True)
        # MTM = sp.dot(M.T,M)
        # MTY = sp.dot(M.T,Y)

        # Compute and store objective function terms
        objectiveFct_tab[0,iterNb] = lambdas[0]/2*splin.norm(Y-sp.dot(M,A_est))**2 + self.lambdas[3]*A_est.sum()
        objectiveFct_tab[1,iterNb] = lambdas[1]/2*splin.norm(A_est-sp.dot(B_est,Z_est))**2
        objectiveFct_tab[2,iterNb] = lambdas[2]/2*splin.norm(sp.sqrt(D[sp.newaxis,:])*C_est-sp.dot(Q_est,sp.sqrt(D[sp.newaxis,:])*Z_est))**2
        if lambdas[4] != 0.:
          objectiveFct_tab[3,iterNb] = lambdas[4]*sp.sqrt(sp.sum( gradU_C**2 + gradR_C**2, axis=0) + self.epsilon**2).sum()
        objectiveFct_tab[4,iterNb] = objectiveFct_tab[:-1,iterNb].sum()

        # Print objective function terms every 10 iterations
        if printEnable and (iterNb%10)==1:
          print("\nIter "+str(iterNb)+" :")
          # print("Soft thresold: {}".format(self.lambdas[3]/(1.01*splin.norm(coeffA))))
          print("0.5*lambd_0*||Y-MA||^2 + lambd_sp*||A||_1: {}".format(objectiveFct_tab[0,iterNb],'e'))
          print("0.5*lambd_1*||A-BZ||^2: {}".format(objectiveFct_tab[1,iterNb],'e'))
          print("0.5*lambd_2*||CD-QZD||^2: {}".format(objectiveFct_tab[2,iterNb],'e'))
          print("lambd_4*||C||_TV: {}".format(objectiveFct_tab[3,iterNb],'e'))

        iterNb += 1

    elif classif_loss == 'cross-entropy':
      while (iterNb <= maxIter) and (iterNb <=5 or (objectiveFct_tab[4,iterNb-2]-objectiveFct_tab[4,iterNb-1])/objectiveFct_tab[4,iterNb-2]>self.tol):

        # Optim with respect to A
        coeffA = lambdas[0]*MTM + lambdas[1]*sp.eye(R)
        Aupdate = A_est - 1/(1.01*splin.norm(coeffA)) * ( sp.dot(coeffA,A_est) - lambdas[0]*MTY - lambdas[1]*sp.dot(B_est,Z_est) )
        A_est = sp.maximum(0, Aupdate - self.lambdas[3]/(1.01*splin.norm(coeffA))) # max elementwise 

        # Optim with respect to B
        coeffB = lambdas[1]*sp.dot(Z_est,Z_est.T)
        Bupdate = B_est - 1/(1.01*splin.norm(coeffB)) * ( sp.dot(B_est,coeffB) - lambdas[1]*sp.dot(A_est,Z_est.T) )
        B_est = sp.maximum(0, Bupdate) # max elementwise

        # Optim with respect to Z
        coeffZ = lambdas[1]*sp.dot(B_est.T,B_est)
        LZ = splin.norm(coeffZ) + lambdas[2]*sp.sum(sp.sum(D[sp.newaxis,:]*C_est,axis=1)*sp.sum(Q_est**2,axis=1))
        Zupdate = Z_est - 1/(1.01*LZ) * ( sp.dot(coeffZ,Z_est) - lambdas[1]*sp.dot(B_est.T,A_est) - lambdas[2]*sp.dot( Q_est.T , (D[sp.newaxis,:]*C_est)/(1.+sp.exp(sp.dot(Q_est,Z_est))) ) )
        thresh_Z = sp.amax( (sp.cumsum( sp.sort(Zupdate,axis=0)[::-1,:],axis=0 )-1) / sp.arange(1,K+1)[:,sp.newaxis] , axis=0)
        Z_est = sp.maximum(0, Zupdate - thresh_Z) # max elementwise

        # Optim with respect to Q
        LQ = lambdas[2]*sp.sum(D[sp.newaxis,:]*C_est,axis=1) + lambda_q
        for c in range(C):
          Q_est[c,:] = Q_est[c,:] - 1/(1.01*LQ[c]) * (-lambdas[2]*sp.sum( (D[sp.newaxis,:]*C_est[c,:]/(1.+sp.exp(sp.dot(Q_est[sp.newaxis,c,:],Z_est)))) *Z_est, axis=1 ) + lambda_q*Q_est[c,:])

        # Optim with respect to C
        if lambdas[4]!=0.:
          LC = lambdas[4]*sp.sqrt(8.)*Wmax/self.epsilon
          Wgrad = W / sp.sqrt(sp.sum(gradR_C**2,axis=0)+sp.sum(gradU_C**2,axis=0)+self.epsilon**2)

          gradTV = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradTV[:,-1,:]  = Wgrad[-1,:]*gradU_C[:,-1,:]
          gradTV[:,:,0]   = Wgrad[:,0]*gradR_C[:,:,0]
          gradTV[:,:-1,:] += sp.diff(Wgrad*gradU_C,axis=1)
          gradTV[:,:,1:]  += -sp.diff(Wgrad*gradR_C,axis=2)
          Cupdate = C_est[:,indUSpl] - 1/(1.01*LC) * ( lambdas[2]*D[sp.newaxis,indUSpl]*sp.log(1.+sp.exp(-sp.dot(Q_est,Z_est[:,indUSpl]))) + lambdas[4]*gradTV.reshape((C,heightImg*widthImg))[:,indUSpl] )
        else:
          Cupdate = C_est[:,indUSpl] - 1/(1.01) * ( D[sp.newaxis,indUSpl]*sp.log(1.+sp.exp(-sp.dot(Q_est,Z_est[:,indUSpl]))) )

        thresh_C = sp.amax( (sp.cumsum( sp.sort(Cupdate,axis=0)[::-1,:],axis=0 )-1) / sp.arange(1,C+1)[:,sp.newaxis] , axis=0)
        C_est[:,indUSpl] = sp.maximum(0, Cupdate - thresh_C) # max elementwise

        if lambdas[4]!=0.:
          C_2d = C_est.reshape((C,heightImg, widthImg))
          gradU_C = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradR_C = sp.zeros((C,heightImg, widthImg),dtype=float)
          gradU_C[:,1:,:] = -sp.diff(C_2d,axis=1) #C_i-1,j - C_i,j
          gradR_C[:,:,:-1] = sp.diff(C_2d,axis=2) #C_i,j+1 - C_i,j


        # Compute and store objective function terms
        objectiveFct_tab[0,iterNb] = lambdas[0]/2*splin.norm(Y-sp.dot(M,A_est))**2 + self.lambdas[3]*A_est.sum()
        objectiveFct_tab[1,iterNb] = lambdas[1]/2*splin.norm(A_est-sp.dot(B_est,Z_est))**2
        objectiveFct_tab[2,iterNb] = lambdas[2]*sp.sum(D[sp.newaxis,:]*sp.log(1.+sp.exp(-sp.dot(Q_est,Z_est)))*C_est) + lambda_q/2*sp.sum(Q_est**2)
        if lambdas[4] != 0.:
          objectiveFct_tab[3,iterNb] = lambdas[4]*sp.sqrt(sp.sum( gradU_C**2 + gradR_C**2, axis=0) + self.epsilon**2).sum()
        objectiveFct_tab[4,iterNb] = objectiveFct_tab[:-1,iterNb].sum()


        # Print objective function terms every 10 iterations
        if printEnable and (iterNb%10)==1:
          print("\nIter "+str(iterNb)+" :")
          # print("Soft thresold: {}".format(self.lambdas[3]/(1.01*splin.norm(coeffA))))
          print("0.5*lambd_0*||Y-MA||^2 + lambd_sp*||A||_1: {}".format(objectiveFct_tab[0,iterNb],'e'))
          print("0.5*lambd_1*||A-BZ||^2: {}".format(objectiveFct_tab[1,iterNb],'e'))
          print("0.5*lambd_2*cross-entropy + 0.5*lambd_q*||Q||^2: {}".format(objectiveFct_tab[2,iterNb],'e'))
          print("lambd_4*||C||_TV: {}".format(objectiveFct_tab[3,iterNb],'e'))

        iterNb += 1

    return A_est, M, B_est, Z_est, Q_est, C_est, objectiveFct_tab[:,2:iterNb], timeInit


  def initVar(self,Y,R,K,maxIter,M,printEnable=False,srand=0):

    # init random number generator seed
    sp.random.seed(srand)

    # Reshape data
    P = Y.shape[1]

    # Init variables
    # A_est = sp.random.rand(R,P)
    A_est = sp.maximum(0.,sp.dot(splin.pinv(M),Y))

    # Compute constants
    MTM  = sp.dot(M.T,M)
    MTY = sp.dot(M.T,Y)
    LM   = splin.norm(MTM)

    ## Optimization
    objectiveFct_tab = sp.empty((maxIter+2))
    objectiveFct_tab[:2] = [1.,2.]
    iterNb       = 0
    A_previous   = A_est.copy()
    y_fista      = A_est.copy()
    t_k          = 1.
    t_k_previous = 1.
    while (iterNb < maxIter) and (iterNb <=5 or abs(objectiveFct_tab[iterNb-2]-objectiveFct_tab[iterNb-1])/objectiveFct_tab[iterNb-2]>self.tol):

      # Optim with respect to A
      Aupdate  = y_fista - 1/(1.01*LM) * ( sp.dot(MTM,y_fista) - MTY )
      A_est    = sp.maximum(0, Aupdate)
      t_k_previous = t_k
      t_k = 0.5*(1.+sp.sqrt(1.+4.*(t_k_previous**2)))
      y_fista = (1.+(t_k_previous-1)/t_k) * A_est - (t_k_previous-1)/t_k * A_previous
      A_previous = A_est.copy()

      if printEnable:
        if (iterNb%10)==1:
          print("Iter "+str(iterNb)+" :")
          print("Rec. error: {}".format(0.5*splin.norm(Y-sp.dot(M,A_est))**2,'e'))

      objectiveFct_tab[iterNb] = 0.5*((Y-sp.dot(M,A_est))**2).sum()
      iterNb += 1

    del y_fista,A_previous,MTM,MTY

    B_est, Z_est = None, None    
    if K != 0:
      ## K-means
      clustLabel = KMeans(n_clusters=K,n_init=10,n_jobs=-1).fit_predict(A_est.T)

      B_est = sp.empty((M.shape[1],K))
      for i in range(K):
          B_est[:,i] = sp.mean(A_est[:,clustLabel==i],axis=1)

      Z_est = sp.random.rand(K,P)
      for k in range(K):
        Z_est[k,clustLabel==k]=1.
        Z_est[k,clustLabel!=k]=0.

    return A_est, B_est, Z_est

  def initVar2(self,Y,labels,candByClass,C,K,maxIter,printEnable=False,srand=0):

    # init random number generator seed
    sp.random.seed(srand)

    # Reshape data
    P = Y.shape[1]
    d = Y.shape[0]

    # Init candidate
    indCand = sp.zeros((P),dtype=int)
    nc = sp.zeros((C+1),dtype=int)
    Rc = sp.zeros((C+1),dtype=int)
    nc[0] = (labels==0).sum()
    for c in range(C):
      # K-means
      clustModel = KMeans(n_clusters=candByClass,n_init=5,n_jobs=-1)
      clustLabel = clustModel.fit_predict(Y[:,labels==c+1].T)

      nc[c+1] = (labels==c+1).sum()
      turnTrue = sp.zeros(nc[c+1], dtype=bool)
      for k in range(candByClass):
        indK = sp.where(clustLabel==k)[0]
        distK = sp.zeros((len(indK)))
        for k2 in range(candByClass):
          if k2 != k:
            distK += sp.arccos( sp.dot(clustModel.cluster_centers_[sp.newaxis,k2,:],Y[:,labels==c+1][:,indK]) / (splin.norm(clustModel.cluster_centers_[k2,:])*splin.norm(Y[:,labels==c+1][:,indK],axis=0)) ).ravel()
        turnTrue[indK[sp.argmax(distK)]] = True
        
      indCand[labels==c+1] = (c+1)*turnTrue
      Rc[c+1] = (indCand==c+1).sum()

    Rc[0] = (indCand!=0).sum()
    M_est = Y[:,indCand!=0]
    print("Dict. size: {}".format(M_est.shape[1]))

    # Init variables
    A_est = sp.maximum(0.,sp.dot(splin.pinv(M_est),Y))

    # Normalize term weighting
    lambdas_0 = self.lambdas[0]/(d*sp.amax(Y)**2)

    # Compute constants
    D = P/float(nc[0])*sp.ones((1,P),dtype=float)
    for c in range(C):
      D[0,labels==(c+1)] = P/float(nc[c+1])
    Dmax = P/float(nc.min())
    MTM  = sp.dot(M_est.T,M_est)
    MTYD = sp.dot(M_est.T,Y)*D
    LM   = splin.norm(MTM)

    ## Optimization
    iterNb       = 0
    objectiveFct_tab = sp.empty((3,maxIter+2))
    objectiveFct_tab[:,:2] = [1.,2.]
    A_previous   = A_est.copy()
    y_fista      = A_est.copy()
    t_k          = 1.
    t_k_previous = 1.
    while (iterNb < maxIter) and (iterNb <=5 or abs(objectiveFct_tab[2,iterNb-2]-objectiveFct_tab[2,iterNb-1])/objectiveFct_tab[2,iterNb-2]>0.1*self.tol):

      # Optim with respect to A
      Aupdate  = y_fista - 1/(1.01*Dmax*LM) * ( sp.dot(MTM,y_fista)*D - MTYD )
      normALine = splin.norm(Aupdate,axis=1,keepdims=True)
      A_est    = sp.maximum(0, (normALine - self.lambdas[3]/(1.01*lambdas_0*LM))/normALine ) * Aupdate
      A_est    = sp.maximum(0, A_est)
      t_k_previous = t_k
      t_k = 0.5*(1.+sp.sqrt(1.+4.*(t_k_previous**2)))
      y_fista = (1.+(t_k_previous-1)/t_k) * A_est - (t_k_previous-1)/t_k * A_previous
      A_previous = A_est.copy()

      objectiveFct_tab[0,iterNb] = 0.5*lambdas_0*((Y*sp.sqrt(D)-sp.dot(M_est,A_est)*sp.sqrt(D))**2).sum()
      objectiveFct_tab[1,iterNb] = self.lambdas[3]*splin.norm(A_est,axis=1).sum()
      objectiveFct_tab[2,iterNb] = objectiveFct_tab[:-1,iterNb].sum()

      if printEnable:
        if (iterNb%10)==1:
          energy = splin.norm(A_est,axis=1)
          print("Iter "+str(iterNb)+" :")
          print("Dict. elements: {}".format((energy!=0).sum()))
          print("Dict. percent: {}".format(100.*(energy!=0).sum()/float(Rc[0])))
          print("Soft thresold: {}".format(self.lambdas[3]/(1.01*lambdas_0*LM)))
          print("Rec. error: {}".format(objectiveFct_tab[0,iterNb],'e'))
          print("Sparse pen.: {}".format(objectiveFct_tab[1,iterNb],'e'))

      iterNb += 1

    del y_fista,A_previous,MTM,MTYD

    normALine = splin.norm(A_est,axis=1)
    M_est = M_est[:,normALine!=0.]
    A_est = A_est[normALine!=0.,:]

    B_est, Z_est = None, None    
    if K != 0:
      ## K-means
      clustLabel = KMeans(n_clusters=K,n_init=10,n_jobs=-1).fit_predict(A_est.T)

      B_est = sp.empty((M_est.shape[1],K))
      for i in range(K):
          B_est[:,i] = sp.mean(A_est[:,clustLabel==i],axis=1)

      Z_est = sp.random.rand(K,P)
      for k in range(K):
        Z_est[k,clustLabel==k]=1.
        Z_est[k,clustLabel!=k]=0.

    ind = sp.zeros((P),dtype=bool)
    ind[indCand!=0] = normALine!=0.
    return A_est, M_est, B_est, Z_est, ind

  def initClassif(self,Y,A_est,M,B_est,Z_est,labels,C, n_neighbors=10, printEnable=False,srand=0):

    # init random number generator seed
    sp.random.seed(srand)

    # Separate labeled from non labeled samples
    indUSpl = sp.where(labels == 0)[0]
    indLSpl = sp.where(labels != 0)[0]

    # random forest
    clfgrid = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-2)
    clfgrid.fit(Y[:,indLSpl].T, labels[indLSpl])

    psiMean = sp.empty((Y.shape[0],Z_est.shape[0]))
    Z_lab = sp.argmax(Z_est,axis=0)
    for k in range(Z_est.shape[0]):
      psiMean[:,k] = sp.mean(Y[:,Z_lab==k],axis=1)

    return sp.dot( clfgrid.predict_proba(psiMean.T).T , Z_est[:,indUSpl] )
