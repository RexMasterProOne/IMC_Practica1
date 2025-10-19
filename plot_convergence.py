import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(csv_file, title="Convergencia del entrenamiento"):
    # Leer CSV ignorando líneas en blanco entre semillas
    data = pd.read_csv(csv_file)

    # Obtener todas las semillas presentes
    seeds = data['seed'].unique()

    plt.figure(figsize=(10, 6))

    for seed in seeds:
        seed_data = data[data['seed'] == seed]

        # Train error → línea discontinua
        plt.plot(seed_data['epoch'], seed_data['train_error'],
                 linestyle='--', label=f"Seed {seed} - Train")

        # Test error → línea continua
        plt.plot(seed_data['epoch'], seed_data['test_error'],
                 linestyle='-', label=f"Seed {seed} - Test")

    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Guardamos y mostramos
    plt.savefig(csv_file.replace(".csv", ".png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python plot_convergence.py <archivo.csv>")
    else:
        plot_convergence(sys.argv[1])
