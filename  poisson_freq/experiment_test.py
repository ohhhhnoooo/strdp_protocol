from network import Network
import numpy as np
from matplotlib import pyplot as plt

def get_weight_change(network, initial_weight):
    final_weights = network.get_final_weights()
    weight_change = final_weights[0, 0] - initial_weight

    return weight_change


def single_synapse(rate_value):
    # params
    Tmax = 2000
    scale = 3
    n_pre = 1
    n_post = 1

    pre_rates = [rate_value]

    initial_weight = 0.5
    weight_matrix = np.array([[initial_weight]])

    network = Network(Tmax, scale, n_pre, n_post, pre_rates, weight_matrix)
    network.run_simulation()

    frequency = compute_frequency(network, Tmax)
    weight_change = get_weight_change(network, initial_weight)

    # Отладочная информация
    results = network.get_results()
    pre_spikes = results['pre_spikes']
    actual_rate = len(pre_spikes[0]) / (Tmax / 1000.0)
    print(f"Задана частота: {rate_value} Гц, Измеренная частота: {actual_rate:.2f} Гц")

    return frequency, weight_change

def compute_frequency(network, Tmax):
    results = network.get_results()
    pre_spikes = results['pre_spikes']
    
    if len(pre_spikes) > 0 and len(pre_spikes[0]) > 0:
        frequency = len(pre_spikes[0]) / (Tmax / 1000.0)
    else:
        frequency = 0.0
    
    return frequency

def plot_results(I_list, frequencies, weight_changes):
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))

    ax1.plot(frequencies, weight_changes, 'go-', linewidth=2, markersize=6)
    ax1.set_xlabel('Частота спайков (Гц)')
    ax1.set_ylabel('Изменение веса синапса')
    ax1.set_xlim(xmax=1000)
    ax1.set_title('STRDP: Зависимость изменения веса от частоты')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    for i, (freq, w_change) in enumerate(zip(frequencies, weight_changes)):
        if i % 100 == 0:
            ax1.annotate(f'I={I_list[i]}', 
                        xy=(freq, w_change), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    rate_list = np.arange(5, 1001, 5)
    frequencies = []
    weight_changes = []
    
    print("Заданная частота | Измеренная частота | Δ веса")
    print("-" * 50)
    
    for rate in rate_list:
        freq, w_change = single_synapse(rate)
        frequencies.append(freq)
        weight_changes.append(w_change)
        print(f"{rate:15} | {freq:17.2f} | {w_change:8.4f}")
    
    plot_results(rate_list, frequencies, weight_changes)