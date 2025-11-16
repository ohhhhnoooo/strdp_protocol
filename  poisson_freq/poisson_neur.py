import numpy as np

class PoissonNeuron:    
    def __init__(self, rate, dt=1.0, type='pre'):
        self.rate = rate
        self.dt = dt
        self.type = type
        self.spike = False
        
    def update(self):
        p_spike = self.rate * (self.dt / 1000.0)
        self.spike = np.random.random() < p_spike
        
    def get_voltage(self):
        return 0.0 if not self.spike else 1.0
        
    def has_spiked(self):
        return self.spike