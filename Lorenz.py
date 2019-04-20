class Lorenz(dynamical_system):
    def __init__(self, rho, sigma, beta):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        
    def rhs(self, t, x):
        
        dxdt = np.zeros(3)
        
        dxdt[0] = self.sigma*(x[1]-x[0])
        dxdt[1] = x[0]*(self.rho-x[2])-x[1]
        dxdt[2] = x[0]*x[1] - self.beta*x[2]
        
        return dxdt