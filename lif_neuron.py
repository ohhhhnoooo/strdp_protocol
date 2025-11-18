import numpy as np

class LIF_neuron:
    def __init__(self, type, tau_m=20.0, v_rest=-65.0, v_thresh=-10.0, v_reset=-65.0, r_m=1.0, dt=1.0):
        self.type = type
        
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.r_m = r_m
        self.dt = dt

        # current state
        self.v = v_rest
        self.I = 0.0
        self.spike = False

    # update state on dt + refactoring period
    def update(self, I_inj=0.0):
        self.I = I_inj

        #else
        dv = (-(self.v - self.v_rest) + self.r_m * self.I) * (self.dt / self.tau_m)

        self.v += dv

        # spike generation
        self.spike = False
        if self.v >= self.v_thresh:
            self.spike = True
            self.v = self.v_reset
        
    def get_voltage(self):
        return self.v

    def has_spiked(self):
        return self.spike

if __name__=="__main__":
    from matplotlib import pyplot as plt
    Tmax = 5000
    scale = 3
    T = np.linspace(0, Tmax, Tmax*scale)
    I = lambda t : 25*np.sin(t/1000*6.28) # pA 
    N = LIF_neuron(dt=T[1]-T[0])
    V = np.zeros_like(T)
    for i, t in enumerate(T):
        V[i]=N.get_voltage()
        N.update(I_inj=I(t))
    
    plt.plot(T, V)
    plt.show()
        

