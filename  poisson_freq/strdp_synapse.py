import numpy as np
from lif_neuron import LIF_neuron

class STRDP_Synapse:
    def __init__(self, pre_neuron, post_neuron, weight=1.0, tau_s=5.0,
                A_plus=0.015, A_minus=0.01, tau_plus=7.0, tau_minus=5.0,
                w_max=1.0, w_min=0.0):
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = weight
        self.tau_s = tau_s
        self.mu = 0.0005

        self.I_syn = 0.0

        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min

        # trace
        self.trace_pre = 0.0
        self.trace_post = 0.0

        # history for graphics
        self.weight_history = [weight]
        self.trace_pre_history = []
        self.trace_post_history = []

        self.dt = self.pre.dt

        # for the method from article
        self.nu = 0.1
        self.alpha = 0.3
        self.b_minus = 65.0

        # last spikes
        self.last_pre_spike = -1e6
        self.last_post_spike = -1e6


    def stdp_method(self, t):
        spike_pre = self.pre.has_spiked()
        spike_post = self.post.has_spiked()

        if spike_pre:
            self.last_pre_spike = t
        if spike_post:
            self.last_post_spike = t

        if spike_post:
            if self.last_pre_spike == -1e6:
                return

            dt = t - self.last_pre_spike
            F_plus = self.nu * (self.w_max - self.weight)
            ltp = F_plus * np.exp(-dt / self.tau_plus)

            F_minus = self.nu * self.alpha * self.weight
            ltd = F_minus / (1 + np.exp(self.b_minus - dt / self.tau_minus))

            self.weight += (ltp - ltd) * self.dt

            self.weight = np.clip(self.weight, self.w_min, self.w_max)

        self.weight_history.append(self.weight)

    def update(self, t):
        # if new spike from pre
        if self.pre.spike:
            total_spike_input = 500.0
        else:
            total_spike_input = 0.0

        # update synaptic I
        dI_syn = (-self.I_syn + total_spike_input) * (self.dt / self.tau_s)
        self.I_syn += dI_syn

        self.stdp_method(t)

    def get_weight(self):
        return self.weight
    
    def get_current(self):
        return self.I_syn