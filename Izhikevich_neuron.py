class Izhikevich_neuron:
    def __init__(self, my_type, I_fun):
        '''
        Creates an Izhikevich neuron of the form 
        dV = 0.04*V^2 + 5*V + 140 - u + Iinj(dt*i)
        du = a(bV-u)
        with reset conditions
        if(v>30)
            v= c
            u+=d
        
        Args:
           my_type = "integrator" or "resonator"
           b_fun = b(X,t)
        '''
        self.I_fun = I_fun
        
        if str(my_type) == "integrator":
            self.a = 0.02
            self.b = -0.1
            self.c = -55
            self.d = 6
            self.V_equil = -87.5
        elif  str(my_type) == "resonator":
            self.a = 0.1
            self.b = 0.26
            self.c = -60
            self.d = -1
            self.V_equil = -62.5
        '''
        add more types
        '''
        
    
    def run(self,tmax,dt):
        '''
        Implements forward Euler (piecewise Izhikevich makes other techniques hard/impossible?). 
        
        Args: 
            tmax: final time
            dt = timestep
            
        Returns:
            V at time points, i*tmax/N.
        '''
        
        a = self.a 
        b = self.b
        c = self.c 
        d = self.d
        I_fun = self.I_fun
        
        #dt = tmax/(N-1)
        N = int(np.ceil(tmax/dt))
        
        V = np.zeros(N)
        u = np.zeros(N)
        t = np.zeros(N)
        
        
        V[0]= self.V_equil
        u[0] = b*self.V_equil
        t[0] = 0
        for i in range(1,N):
            if V[i-1]==30: 
                '''
                Reset condition
                '''
                V[i] = c
                u[i] = u[i-1] + d
                t[i] = t[i-1] + dt
            else:
                '''
                Normal evolution
                '''
                V[i] = V[i-1] + dt*(0.04*V[i-1]**2 + 5*V[i-1] + 140 - u[i-1] + I_fun[i-1] )
                u[i] = u[i-1] + dt*a*(b*V[i-1]-u[i-1])
                t[i] = t[i-1] + dt
            
            if V[i]>30:
                '''
                Cap spike height
                '''
                V[i] = 30
       
        self.Vs = V
        self.us = u
        self.ts = t
        return V
    
    def voltage_plot(self):
        '''
        Plots the solution V(t)
        '''
        plt.plot(self.ts,self.Vs)
        plt.xlabel('$t$')
        plt.ylabel('$V$')
        plt.show()
        
    def phase_plot(self):
        plt.plot(self.Vs,self.us)
        plt.xlabel('$V$')
        plt.ylabel('$u$')
        plt.show()
        
'''       
tmax = 100
dt = 0.01
N = int(tmax/dt)
I = 40*np.ones(N)
izzy = Izhikevich_neuron("integrator", I)
izzy.run(tmax,dt)
izzy.voltage_plot()
izzy.phase_plot()
'''
    