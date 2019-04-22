class CCM:
    def __init__(self, data):
        self.data = data
        self.N = len(data[:,0])
        
    
    def predict(self, tau, predictee_dim, manifold_dim, num_neighbors=None, plotTime = 200):
        #time series to consider
        Y = self.data[:,predictee_dim]
        
        #create delay coordinate manifold
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points
        N2 = self.N - manifold_dim*tau
        
        #size of simplex
        if num_neighbors is None:
            k = manifold_dim+1
        else:
            k = num_neighbors
            
        half = int(N2/2)
        Y_pred = np.zeros(self.N)
        Y_pred[:half]= Y[:half]
        
        point = np.zeros(manifold_dim)
        #given first half of data, try to continue series
        for i in range(half, self.N):
            #grab manifold point to predict next
            #oldest index used
            timeIndex = i-1-(manifold_dim-1)*tau
            for j in range(manifold_dim):
                #left entry is most delayed, manifold_dim*tau, 
                point[j] = Y_pred[timeIndex]
                timeIndex += tau
                
            #new predicted value using first-half manifold
            new = self.k_simplex_predict(point, M_Y, k, half)
            
            #record new value
            Y_pred[i]=new
            
        #might not be fair to compare same distance out for different stencil lengths, 
        #since longer stencils use true data for longer
        plt.plot(range(half,half+plotTime),Y[half:half+plotTime])
        plt.plot(range(half,half+plotTime),Y_pred[half:half+plotTime])
        #plt.plot(range(half,half+30*tau),Y[half:half+30*tau])
        #plt.plot(range(half,half+30*tau),Y_pred[half:half+30*tau])
        plt.xlabel('$t$')
        plt.ylabel('$Y$')
        plt.show()
        
    '''
    perform CCM to 'predict' the time series of the predictee variable 
    using the manifold of the predictor
    '''
    def cross_map(self, tau, predictee_dim, predictor_dim, manifold_dim, num_neighbors=None):
        '''
        will see how well X 'predicts' Y
        '''
        
        X = self.data[:,predictor_dim]
        Y = self.data[:,predictee_dim]
        
        
        '''
        create manifold 
        each row is a data point in delay-coordinate space
        (N-dim*tau) x (dim)
        '''
        M_X = self.make_manifold(X, manifold_dim, tau)
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points
        N2 = self.N - manifold_dim*tau
        
        #each manifold point has a corresponding guess
        cross_guess = np.zeros(N2)
        
        #size of simplex
        if num_neighbors is None:
            k = manifold_dim+1
        else:
            k = num_neighbors
        
        #for each manifold point
        for i in range(0, N2):
            #change last input to max(i,k+1) if want to use all up to self
            #int(N2/2)to use first half of data
            cross_guess[i] = self.k_simplex_cross_map(M_X[i,:], M_X, M_Y, k , max(i,k+1))
            
        
        plt.plot(range(k,N2),M_Y[k:N2 , manifold_dim - 1])
        plt.plot(range(k,N2),cross_guess[k:N2])
        plt.xlabel('$t$')
        plt.ylabel('$Y$')
        plt.show()
    
    '''
    predict corresponding Y manifold point for the given X manifold point 
    based on X manifold up to time t 
    (can stop recording manifold before the point in question)
    with exponential interpolation weighting
    '''
    def k_simplex_cross_map(self, point, M_X, M_Y, k, t):
        #get nearest X-manifold points and their manifold index
        knn = self.k_nearest_neighbors(point , M_X , k , t)
        
        
        #exponentially weight by distance
        knn[:,0]= np.exp(-knn[:,0])
        
        #normalize 
        knn[:,0]= knn[:,0]/np.sum(knn[:,0])
        
        #grab manifold dimension
        m = len(point)
        
        #weighted sum of their corresponding rows' rightmost value
        #should this be the whole row, predicting a full manifold point?
        cross_guess = np.sum(knn[:,0]*M_Y[knn[:,1].astype(int), m-1])
        return cross_guess
    
    '''
    predict next entry after the given manifold point 
    based on manifold up to time t 
    (can stop recording manifold before the point in question)
    with exponential interpolation weighting
    '''
    def k_simplex_predict(self, point, manifold, k, t):
        knn = self.k_nearest_neighbors(point, manifold,k,t)
        
        #exponentially weight by distance
        knn[:,0]= np.exp(-knn[:,0])
        
        #normalize 
        knn[:,0]= knn[:,0]/np.sum(knn[:,0])
        
        #weighted sum of next-points (third element of next row on manifold)
        nextrows = knn[:,1]+1
        nextrows = nextrows.astype(int)
        prediction = np.sum(knn[:,0]*manifold[nextrows,len(point)-1])
        return prediction
        
    '''
    returns [distance, index] for the k closest points to the passed point
    on the passed manifold up to time t
    '''
    def k_nearest_neighbors(self, point, manifold, k, t):
        #subtract our last point from all previous
        diff = manifold[:t,:]-point
        
        #take norm of those differences
        dists = np.linalg.norm(diff,axis=1)
        
        #attach indices (for latest time in trio) to each distance
        d = np.zeros([t, 2]) 
        d[:,0]= dists
        d[:,1]= range(0,t)
        
        #sort based on first entry (distance), bring indices along for the ride
        d2 = d[d[:,0].argsort()]
        
        #if included self, return the next k points
        if d2[0,0]==0:
            return d2[1:k+1,:]
        else:
            return d2[0:k,:]
            
            
        
    def make_manifold(self, X, dim, tau):
        N = self.N
        manifold = np.zeros([N-dim*tau , dim])
        
        for i in range(0, N - dim*tau):
            for j in range(0, dim):
                #upper left is x[0], beside it is x[tau], then x[2tau] 
                manifold[i,j] = X[i+tau*j]
                
        return manifold
        
    
    '''
run prep.py, dynamical_system.py, Lorenz.py, then:
    
lorrie = Lorenz(28,10,8/3)
sol = lorrie.integrate(0,50,0.01, np.array([1,1,1]))
lorrie.trajectory_plot()
manny = CCM(sol.y.T)

manny.predict(20,1,4, plotTime = 300)

manny.cross_map(1,1,0,3)

compare to

manny = CCM(np.random.rand(5000,2))
manny.cross_map(1,1,0,3)


    '''
    