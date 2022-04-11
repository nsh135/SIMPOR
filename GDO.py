from collections import Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np



debug = False
class GDO:
    def __init__(self,X,y,k_neighbor ):
        self.X = X
        self.y = y
        self.k_neighbor = k_neighbor
        self.labelCount = Counter(self.y)
        if debug: print( "GDO label count {}".format(self.labelCount) )
        self.minorityClass = 0 if self.labelCount[0] <= self.labelCount[1] else 1
        self.majorityClass = 0 if self.minorityClass ==1 else 1

    def anchorImportatnce(self,anchor):
        """
        GDO
        Implementation of Gaussian Distribution Based Oversampling for Imbalanced Data Classification (Yuxi Xie et al.)
        return anchor importatnce and nearestDist
        """
        neighbor = NearestNeighbors(n_neighbors=self.k_neighbor, radius=1)
        neighbor.fit(self.X)
        nbrs_dist, nbrs_idx = neighbor.kneighbors([anchor], self.k_neighbor, return_distance=True)
        nbrs_dist= nbrs_dist[0]
        nbrs_idx = nbrs_idx[0]
        if debug: print("neighbors index, distance  ", nbrs_idx, nbrs_dist)
        nbrs_label = self.y[nbrs_idx]
        neighborLabelCount = Counter(nbrs_label)
        if debug: print("neighbors labels ", nbrs_label, neighborLabelCount)
        if debug: print(" nbrs_dist, nbrs_label==self.majorityClass ",nbrs_dist,nbrs_label==self.majorityClass)
        if neighborLabelCount[self.majorityClass]:
            C_anchor = neighborLabelCount[self.majorityClass]/self.k_neighbor
        else: 
            C_anchor = 0

        if debug: print(nbrs_dist[nbrs_label==self.majorityClass])
        if  not np.isnan(nbrs_dist.mean()) and not np.isnan(nbrs_dist[nbrs_label==self.majorityClass].mean()):
            D_anchor = nbrs_dist[nbrs_label==self.majorityClass].mean() / nbrs_dist.mean()
        else:
            D_anchor= 0

        if debug: print("*****D_anchor  nbrs_dist.mean() ", D_anchor,nbrs_dist.mean(), nbrs_dist[nbrs_label==self.majorityClass].mean() )
        I_anchor = C_anchor + D_anchor 
        nearestDist= 0 
        if debug: print("nbrs_idx ", nbrs_idx)
        for idx,_ in  enumerate(nbrs_idx):
            if debug: print("***", nbrs_label[idx])
            if nbrs_label[idx] == self.minorityClass and idx != 0:
                nearestDist = nbrs_dist[idx]
                break
        if debug: print("***Nearest distance : {}".format(nearestDist) )
        return I_anchor, nearestDist

    def sample_spherical(self,centroid_x ,radius, npoints):
        """
        sample a point on n-sphere centered by centroid_x
        Unit circle
        """
        #generate points in n-sphere centered by the origin
        ndim = len(centroid_x)
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        vec *= radius
        vec = vec.T
        
        #sanity check if the length of points are 1
        for v in vec:# loop all points
            s = 0
            for i in range(ndim):# loop all dims
                s += v[i]**2
            if s - radius**2 > 0.0000001 : raise ValueError('Initial value is not on sphere')
                
        # shift to be centered by centroid_x
        vec = vec + centroid_x
        return vec

    def balancing(self):
        num_generate = max(0,self.labelCount[self.majorityClass] - self.labelCount[self.minorityClass] )
        Minority_X = self.X[self.y==self.minorityClass]
        Minority_Dist = [] # distance between this minority sample and its nearest neigbor
        Minority_Prob = [] ## probability a sample is selected
        for i,x in enumerate(Minority_X):
            a,b = self.anchorImportatnce(x)
            Minority_Prob.append(a)
            Minority_Dist.append(b)
        
        if debug: print(Minority_Prob)
        #normalize probability 
        if np.sum(Minority_Prob) != 0:
            Minority_Prob = np.array(Minority_Prob)/np.sum(Minority_Prob)
        else:
            Minority_Prob = np.array(Minority_Prob)/len(Minority_Prob)
        
        
        X = []
        y = []
        ##random seletect minority to generate synthetic data
        while num_generate >= 0:
            randIdx = np.random.choice(range(len( Minority_X )), p=Minority_Prob)
            y.append(self.minorityClass)
            d = np.random.normal(0, Minority_Dist[randIdx], 1)  
            X.append(  self.sample_spherical(Minority_X[randIdx] ,d, 1)  )
            num_generate -=1

        X = np.array(X)
        X = X.reshape( (X.shape[0],X.shape[-1])  )
        y = np.array(y)
        if debug: print("---Generated X.shape, y shape",X.shape,y.shape)
        X, y = np.concatenate([self.X,X ]), np.concatenate([self.y,y ]) 
        return   X, y

if __name__ == '__main__':
    X = np.array( [[0,1], [2,3], [4, 5] , [4,2], [2,4] , [0,9], [1,2] , [1,1] , [2,3] , [4,4] ] )
    y = np.array([0,1,0,0,1,1,1,1,1,1] )
    print(X.shape,y.shape)
    gdo= GDO(X,y,5)
    X_,y_ = gdo.balancing()
    print(X_.shape, y_.shape)
    



    


