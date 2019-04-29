class manifolding:
    def __init__(self, data):
        self.data = data
      
        
    def corr_plot(self, tau,  manifold_dim, data_dim = None, num_neighbors=None):
        if data_dim is not None:
            #time series to consider
            Y = self.data[:,data_dim]
        else:
            Y = self.data
            
        N = len(Y)
        N2 = N - manifold_dim*tau
        half = int(N2/2)
        
        corrs = np.ones(11)
        Y_preds = np.zeros([N,11])
        Y_preds[:,0]= Y
        
        for P in range(1,11):
            Y_preds[:,P]= self.predict_1_p_step(tau, manifold_dim, p=P)
            
            corrs[P] = np.corrcoef(Y[half:],Y_preds[half:,P])[1,0]
            
        
        plt.plot(range(1,11),corrs[1:], marker='o')
        plt.xlabel('$p$')
        plt.ylabel('$correlation coeff$')
        plt.ylim([0,1])
        plt.show()
        
        
        '''
        for P in range(1,11):
            Y_preds[:,P]= self.predict_p_steps(tau, manifold_dim, p=P)
            
            corrs[P] = np.corrcoef(Y[half:],Y_preds[half:,P])[1,0]
            
        
        plt.plot(range(1,11),corrs[1:], marker='o')
        plt.xlabel('$p$')
        plt.ylabel('$correlation coeff$')
        plt.show()
        '''
        
        

    
    
    def predict_1_p_step(self, tau,  manifold_dim, data_dim = None, num_neighbors=None, p=1):
       
        if data_dim is not None:
            #time series to consider
            Y = self.data[:,data_dim]
        else:
            Y = self.data
            
        N = len(Y)
        
        #create delay coordinate manifold
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points
        N2 = N - manifold_dim*tau
        
        #size of simplex
        if num_neighbors is None:
            k = manifold_dim+1
        else:
            k = num_neighbors
            
        half = int(N2/2)
        Y_pred = np.zeros(N)
        Y_pred[:half]= Y[:half]
        
        point = np.zeros(manifold_dim)
        #given first half of data as library, try to continue series forward
        #using step of p indices forward from real-data stencil
        #to predict index i, stencil = indices i-p, i-p-tau, i-p-2tau, etc
        for i in range(half, N):
            #grab manifold point to predict next
            #oldest index used
            timeIndex = i-p-(manifold_dim-1)*tau
            for j in range(manifold_dim):
                #left entry is most delayed, i-p-(manifold_dim-1)*tau, 
                point[j] = Y[timeIndex]
                timeIndex += tau
                
            #new predicted value using first-half manifold
            new = self.k_simplex_predict(point, M_Y, k, half, forwardStep=p)
            
            
            #record new value
            Y_pred[i]=new
        
        if p ==1 or p==2 or p==5:
            #figure 1b and c, plotting real data against predictions 2 and 5 steps in the future 
            plt.scatter(Y[half:],Y_pred[half:])
            plt.xlabel('$\Delta Y_{Observed}$')
            plt.ylabel('$\Delta Y_{Predicted}$')
            plt.show()
        
        return Y_pred
    
    
    '''
    my idea of what's being plotted in the paper:
    for the second half of data, take real history to make manifold point, 
    then predict next points individually, based off predicted data, out to p steps
    record that predicted p-future point, then start over for the next real-data point
    '''
    def predict_p_steps(self, tau,  manifold_dim, data_dim = None, num_neighbors=None, p=1):
       
        if data_dim is not None:
            #time series to consider
            Y = self.data[:,data_dim]
        else:
            Y = self.data
            
        N = len(Y)
        
        #create delay coordinate manifold
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points (=len(M_Y) I think)
        N2 = N - manifold_dim*tau
        
        #size of simplex
        if num_neighbors is None:
            k = manifold_dim+1
        else:
            k = num_neighbors
            
        half = int(N2/2)
        Y_pred = np.zeros(N)
        Y_pred[:half] = Y[:half]
        
        #how many points of real data we'll need
        anchor = (manifold_dim-1)*tau
        
        #will hold temporarily projected values as well as real data used
        Y_temp = np.zeros(anchor + p)
        
        point = np.zeros(manifold_dim)
        
        #given first half of data as library, try to continue series forward
        #stepping by 1 forward from real-data stencil
        #to predict index i, start by predicting i-p+1 with
        #stencil = indices i-p, i-p-tau, i-p-2tau, etc
        
        #for each data point we want to predict
        for i in range(half, N):
            #predict p steps into the future, one at a time, to get from i-p to i
            
            #oldest index used
            timeIndex = i-p-anchor
            
            
            #fill in real data we'll need
            Y_temp[:anchor] = Y[timeIndex:i-p]
            
            #predict each time step out to p:
            for P in range(p):
                
                #construct point we'll use
                for j in range(manifold_dim):
                    
                    #left entry is most delayed 
                    point[j] = Y_temp[P+j*tau]
                
                #new predicted value using first-half manifold
                Y_temp[anchor+P] = self.k_simplex_predict(point, M_Y, k, half, forwardStep=1)
            
            
            #record last value in our p-steps-out time series
            Y_pred[i] = Y_temp[anchor+p-1]
        
        if p ==1 or p==2 or p==5:
            #figure 1b and c, plotting real data against predictions 2 and 5 steps in the future 
            plt.scatter(Y[half:],Y_pred[half:])
            plt.xlabel('$\Delta Y_{Observed}$')
            plt.ylabel('$\Delta Y_{Predicted}$')
            plt.show()
            #this should be equivalent when p=1, but they aren't
        
        return Y_pred
    
    
    #tries to continue the time series one timestep at a time
    def predict(self, tau,  manifold_dim, data_dim = None, num_neighbors=None, plotTime = 200):
        
        if data_dim is not None:
            #time series to consider
            Y = self.data[:,data_dim]
        else:
            Y = self.data
            
        N = len(Y)
            
        #create delay coordinate manifold, N2 by manifold_dim
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points
        N2 = N - manifold_dim*tau
        
        #size of simplex
        if num_neighbors is None:
            k = manifold_dim+1
        else:
            k = num_neighbors
            
        half = int(N2/2)
        Y_pred = np.zeros(N)
        Y_pred[:half]= Y[:half]
        
        point = np.zeros(manifold_dim)
        #given first half of data, try to continue series
        for i in range(half, N):
            #grab manifold point to predict next
            #oldest index used
            timeIndex = i-1-(manifold_dim-1)*tau
            for j in range(manifold_dim):
                #left entry is most delayed, manifold_dim*tau, 
                point[j] = Y_pred[timeIndex]
                timeIndex += tau
                
            #new predicted value using first-half manifold
            new = self.k_simplex_predict(point, M_Y, k,half)
            
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
        
        plt.figure()
        plt.scatter(Y[half:half+plotTime],Y_pred[half:half+plotTime])
        plt.xlabel('$\Delta Y_{Observed}$')
        plt.ylabel('$\Delta Y_{Predicted}$')
        plt.show()
        

        
        #compare manifolds
        
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
        
        
        N = len(X)
        
        '''
        create manifold 
        each row is a data point in delay-coordinate space
        (N-dim*tau) x (dim)
        '''
        M_X = self.make_manifold(X, manifold_dim, tau)
        M_Y = self.make_manifold(Y, manifold_dim, tau)
        
        # number of manifold points
        N2 = N - manifold_dim*tau
        
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
    predict next entry (or entry p steps later) after the given manifold point 
    based on manifold up to time t 
    (can stop recording manifold before the point in question)
    with exponential interpolation weighting
    '''
    def k_simplex_predict(self, point, manifold, k, t, forwardStep=1):
        knn = self.k_nearest_neighbors(point, manifold,k,t)
        
        #exponentially weight by distance ratio
        knn[:,0]= np.exp(-knn[:,0]/knn[0,0])
        
        #normalize 
        knn[:,0]= knn[:,0]/np.sum(knn[:,0])
        
        
        #manifold-index of rows p steps after the simplex points
        nextrows = knn[:,1]+forwardStep
        nextrows = nextrows.astype(int)
        #weighted sum of next-points (last elements of destination rows on manifold)
        prediction = np.sum(knn[:,0]*manifold[nextrows,len(point)-1])
        return prediction
        
    '''
    returns [distance, index] for the k closest points to the passed point
    on the passed manifold up to time t
    '''
    def k_nearest_neighbors(self, point, manifold, k, t):
        
        #subtract our last point from our library
        diff = manifold[:t,:]-point
        
        #take norm of those differences
        dists = np.linalg.norm(diff,axis=1)
        
        #attach indices (for latest time in trio) to each distance
        d = np.zeros([t, 2]) 
        d[:,0]= dists
        d[:,1]= range(t)
        
        #sort based on first entry (distance), bring indices along for the ride
        d2 = d[d[:,0].argsort()]
        
        #if included a zero distance, return the next k points
        if d2[0,0]==0:
            return d2[1:k+1,:]
        else:
            return d2[0:k,:]
            
            
        
    def make_manifold(self, X, dim, tau):
        N = len(X)
        manifold = np.zeros([N-dim*tau , dim])
        
        for i in range(0, N - dim*tau):
            for j in range(0, dim):
                #upper left is x[0], beside it is x[tau], then x[2tau] 
                manifold[i,j] = X[i+tau*j]
                
        return manifold
        
    
'''
run prep.py, dynamical_system.py, Lorenz.py, manifolding.py, then:
    
lorrie = Lorenz(28,10,8/3)
sol = lorrie.integrate(0,50,0.01, np.array([1,1,1]))
lorrie.trajectory_plot()

manny = CCM(sol.y.T)

manny.predict(20,4, data_dim = 1, plotTime = 300)

manny.cross_map(1,1,0,3)

compare to

manny = CCM(np.random.rand(5000,2))
manny.cross_map(1,1,0,3)

'''
    
'''
run prep.py, Tent_map.py, manifolding.py, then:

figure 1    
tent = Tent_Map(1.9999)
X = tent.run(0.3,1001)
dX = X[1:]-X[:1000]
manny = manifolding(dX)
manny.corr_plot(1,3)
'''

'''
figure 2
Sineseries = np.sin(0.5*np.arange(300))+(-0.5+np.random.rand(300))
man2 = manifolding(Sineseries)
man2.corr_plot(1,3)
'''
    
    