from strdp_synapse import STRDP_Synapse
from lif_neuron import LIF_neuron
import numpy as np

class Network:
    def __init__(self, Tmax, scale, n_pre, n_post, I, weight_matrix):
        self.Tmax = Tmax
        self.scale = scale
        self.n_pre = n_pre
        self.n_post = n_post
        self.I = I
        self.weight_matrix = weight_matrix

        # time
        self.T = np.linspace(0, Tmax, Tmax * scale)
        self.dt = self.T[1] - self.T[0]

        # neurons
        self.pre_neurons = np.array([LIF_neuron(type='pre', dt=self.dt) for _ in range(n_pre)], dtype=object)
        self.post_neurons = np.array([LIF_neuron(type='post', dt=self.dt) for _ in range(n_post)], dtype=object)

        # synapses
        self.synapses = np.empty((n_pre, n_post), dtype=object)
        for i in range(n_pre):
            for j in range(n_post):
                self.synapses[i, j] = STRDP_Synapse(
                    self.pre_neurons[i], 
                    self.post_neurons[j], 
                    weight=weight_matrix[i, j]
                )

        # data saving
        self.V_pre_history = np.zeros((len(self.T), n_pre))
        self.V_post_history = np.zeros((len(self.T), n_post))
        self.weights_history = np.zeros((len(self.T), n_pre, n_post))
        self.I_syn_history = np.zeros((len(self.T), n_post))

        self.pre_spikes_history = np.zeros((len(self.T), n_pre), dtype=bool)
        self.post_spikes_history = np.zeros((len(self.T), n_post), dtype=bool)


    def run_simulation(self):
        for step, t in enumerate(self.T):
            for i, neuron in enumerate(self.pre_neurons):
                neuron.update(I_inj=self.I[i](t))
                self.V_pre_history[step, i] = neuron.get_voltage()
                self.pre_spikes_history[step, i] = neuron.has_spiked()

            for i in range(self.n_pre):
                for j in range(self.n_post):
                    self.synapses[i, j].update(t)
                    self.weights_history[step, i, j] = self.synapses[i, j].get_weight()
                    I_post = np.zeros(self.n_post)
        
            for i in range(self.n_pre):
                for j in range(self.n_post):
                    I_post[j] += self.synapses[i, j].I_syn * self.synapses[i, j].weight

            for j, neuron in enumerate(self.post_neurons):
                neuron.update(I_inj=I_post[j])
                self.V_post_history[step, j] = neuron.get_voltage()
                self.post_spikes_history[step, j] = neuron.has_spiked()
                self.I_syn_history[step, j] = I_post[j]


    def get_results(self):
        return {
            'time': self.T,
            'V_pre': self.V_pre_history,
            'V_post': self.V_post_history,
            'weights': self.weights_history,
            'I_syn': self.I_syn_history,
            'pre_spikes': self.get_spike_trains(self.pre_neurons),
            'post_spikes': self.get_spike_trains(self.post_neurons)
        }

    def get_spike_trains(self, neurons):
        spike_trains = []
        for neuron_idx, neuron in enumerate(neurons):
            if neuron.type == 'pre':
                spike_times = self.T[self.pre_spikes_history[:, neuron_idx]]
            else:
                spike_times = self.T[self.post_spikes_history[:, neuron_idx]]
            spike_trains.append(spike_times)
        return spike_trains

    def get_final_weights(self):
        final_weights = np.zeros((self.n_pre, self.n_post))

        for i in range(self.n_pre):
            for j in range(self.n_post):
                final_weights[i, j] = self.synapses[i, j].get_weight()
        return final_weights