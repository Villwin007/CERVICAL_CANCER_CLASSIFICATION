import matplotlib.pyplot as plt
import os

genotype = ['max_pool_3x3', 'dil_conv_5x5', 'sep_conv_3x3']

def visualize_genotype(genotype):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, len(genotype))
    ax.set_ylim(0, 1)
    ax.axis('off')
    for i, op in enumerate(genotype):
        ax.text(i + 0.5, 0.5, op, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round", facecolor='lightblue', edgecolor='black'))
        if i > 0:
            ax.arrow(i - 0.2 + 0.5, 0.5, 0.4, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
    plt.title("Final Genotype")
    os.makedirs("outputs/results", exist_ok=True)
    plt.savefig("outputs/results/genotype_visualization.png")
    print("Genotype visualization saved to outputs/results/genotype_visualization.png")

visualize_genotype(genotype)
