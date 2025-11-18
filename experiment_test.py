from strdp_synapse import STRDP_Synapse
from lif_neuron import LIF_neuron
from neur_network_for_STRDP import Network
import numpy as np

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # params
    Tmax = 300
    scale = 3
    n_pre = 1
    n_post = 1
    
    # input I
    # sin
    # phases = np.linspace(2 * np.pi, 0, n_pre, endpoint=False)
    # I_list = [lambda t, phase=phase: 160 * (np.sin(t/1000 * 2 * np.pi + phase)) for phase in phases]   # pA
    # const
    I_list = [lambda t: 1000]
    # exp
    # I_list = [lambda t, phase=phase: 100 * (np.exp(-t/1000)) + phase for phase in phases]

    # I_list = [(lambda t: 200 * (np.sin(t/1000 * 2 * np.pi))), (lambda t: 200 * (np.cos(t/1000 * 2 * np.pi))), (lambda t: 150 * (np.sin(t/1000 * 2 * np.pi)))]

    # weights
    init_weight = 0.1
    weight_matrix = np.linspace(0.5, 0.5, n_pre).reshape(n_pre, n_post)
    # weight_matrix = np.random.uniform(0, 1.0, size=(n_pre, n_post))

    network = Network(Tmax, scale, n_pre, n_post, I_list, weight_matrix)
    network.run_simulation()
    results = network.get_results()
    
    # plots with results
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    
    for i in range(n_pre):
        axes[0].plot(results['time'], results['V_pre'][:, i], label=f'Pre {i+1}')
    axes[0].set_title('Presynaptic Voltages')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].legend()
    
    # post V
    for j in range(n_post):
        axes[1].plot(results['time'], results['V_post'][:, j], label=f'Post {j+1}')
    axes[1].set_title('Postsynaptic Voltages')
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].legend()
    
    # synapse weights
    for i in range(n_pre):
        for j in range(n_post):
            axes[2].plot(results['time'], results['weights'][:, i, j], 
                           label=f'W{i+1}{j+1}')
    axes[2].set_title('Synaptic Weights')
    axes[2].set_ylabel('Weight')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # final weights
    print("Initial weights matrix:")
    print(weight_matrix)
    print("\nFinal weights matrix:")
    print(network.get_final_weights())