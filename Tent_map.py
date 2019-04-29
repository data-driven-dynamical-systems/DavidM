class Tent_Map:
    def __init__(self, r):
        self.r = r
        
    def run(self, x0, N):
        x = np.zeros(N)
        r = self.r
        x[0]= x0
        for i in range(1,N):
            x[i] = r*min(x[i-1],1-x[i-1])
            
        return x
    
    
    '''
tent = Tent_Map(1.9999)
X = tent.run(0.3,1001)
dX = X[1:]-X[:1000]
    '''
