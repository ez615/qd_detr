import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

N_EPOCH = 200
FUNCTIONS = ['linear', 'cosine', 'inverse sqrt', 'step', 'sigmoid', 'none']
COLORS = sns.color_palette()

epochs = np.arange(0, 200)

linear_sim = [(1 - epoch_i / N_EPOCH) for epoch_i in epochs]
linear_giou = [epoch_i / N_EPOCH for epoch_i in epochs]

cos_sim = [(1 + math.cos(epoch_i * math.pi / N_EPOCH)) / 2 for epoch_i in epochs]
cos_giou = [(1 - math.cos(epoch_i * math.pi / N_EPOCH)) / 2 for epoch_i in epochs]

inv_sqrt_sim = [1 / math.sqrt(epoch_i + 1) for epoch_i in epochs]
inv_sqrt_giou = [1 - (1 / math.sqrt(epoch_i + 1)) for epoch_i in epochs]

step_sim = np.concatenate([np.ones(100), np.zeros(100)])
step_giou = np.flip(step_sim)

sigmoid_sim = [1 / (1 + np.exp(epoch_i - N_EPOCH / 2)) for epoch_i in epochs]
sigmoid_giou = [1 / (1 + np.exp(-epoch_i + N_EPOCH / 2)) for epoch_i in epochs]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

### All graphs
axes[0].plot(epochs, linear_sim, color=COLORS[0])
axes[0].plot(epochs, cos_sim, color=COLORS[1])
axes[0].plot(epochs, inv_sqrt_sim, color=COLORS[2])
axes[0].plot(epochs, step_sim, color=COLORS[3])
axes[0].plot(epochs, sigmoid_sim, color=COLORS[4])

axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Sim Loss weight')

### mAP compare
mAPs = [41.1, 42.44, 41.48, 41.82, 41.17, 41.8]
axes[1].bar(range(len(FUNCTIONS)), mAPs, color=COLORS, width=0.5)
for i in range(len(FUNCTIONS)):
    axes[1].text(i, mAPs[i], mAPs[i], ha='center', va='bottom', size=8)

axes[1].set_xticks(range(len(FUNCTIONS)))
axes[1].set_xticklabels(FUNCTIONS)
axes[1].set_yticks(np.arange(41, 43, 0.5))
axes[1].grid(True, axis='y', linestyle='--')
axes[1].set_ylim(40.5, 43)
axes[1].set_ylabel('mAP')

fig.savefig('all_func_mAP.png')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

### each functions
axes[0, 0].plot(epochs, linear_sim, color=COLORS[0])
axes[0, 0].plot(epochs, linear_giou, color=COLORS[0], linestyle='dashed')
axes[0, 0].set_title(FUNCTIONS[0])

axes[0, 1].plot(epochs, cos_sim, color=COLORS[1])
axes[0, 1].plot(epochs, cos_giou, color=COLORS[1], linestyle='dashed')
axes[0, 1].set_title(FUNCTIONS[1])

axes[0, 2].plot(epochs, inv_sqrt_sim, color=COLORS[2])
axes[0, 2].plot(epochs, inv_sqrt_giou, color=COLORS[2], linestyle='dashed')
axes[0, 2].set_title(FUNCTIONS[2])

axes[1, 0].plot(epochs, step_sim, color=COLORS[3])
axes[1, 0].plot(epochs, step_giou, color=COLORS[3], linestyle='dashed')
axes[1, 0].set_title(FUNCTIONS[3])

axes[1, 1].plot(epochs, sigmoid_sim, color=COLORS[4])
axes[1, 1].plot(epochs, sigmoid_giou, color=COLORS[4], linestyle='dashed')
axes[1, 1].set_title(FUNCTIONS[4])

axes[1, 2].plot(epochs, np.ones(200), color=COLORS[5])
axes[1, 2].plot(epochs, np.ones(200), color=COLORS[5], linestyle='dashed')
axes[1, 2].set_title(FUNCTIONS[5])

fig.savefig('each_graphs.png')
plt.close()