import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools, algorithms
from tqdm import trange
import random
import pandas as pd
import matplotlib.pyplot as plt

# 1) Simulación de datos: Ornstein–Uhlenbeck
def simulate_ou(theta=1.0, sigma=0.5, dt=0.01, steps=1000, paths=1000):
    x = np.zeros((paths, steps+1), dtype=np.float32)
    for t in range(steps):
        dw = np.random.normal(scale=np.sqrt(dt), size=paths)
        x[:, t+1] = x[:, t] - theta * x[:, t] * dt + sigma * dw
    return x

# 2) Preparación del dataset
data = simulate_ou()
X = data[:, :-1].reshape(-1,1)
y = data[:, 1: ].reshape(-1,1)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                          batch_size=256, shuffle=False)

# 3) Definición de la red
class Net(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(n_hidden, lr, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Entrenamiento
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
    # Evaluación
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            total_loss += loss_fn(model(xb), yb).item() * xb.size(0)
    return total_loss / len(val_loader.dataset)

# 4) Configuración DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("n_hidden", random.randint, 5, 100)
toolbox.register("log_lr", lambda: random.uniform(-4, -1))
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_hidden, toolbox.log_lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(ind):
    n_hidden, log_lr = ind
    lr = 10 ** log_lr
    mse = train_and_evaluate(n_hidden=n_hidden, lr=lr)
    return (mse,)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=[50, -2.5], sigma=[20, 1.0], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main_evolution(pop_size=20, gens=10):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    log = tools.Logbook()
    log.header = ["gen", "nevals"] + stats.fields

    # Bucle manual para poder imprimir por generación
    for gen in range(gens):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        hof.update(pop)

        record = stats.compile(pop)
        log.record(gen=gen, nevals=len(offspring), **record)
        print(f"Gen {gen} → avg: {record['avg']:.5f}, min: {record['min']:.5f} "
              f"(best n_hidden={hof[0][0]}, lr=10^{hof[0][1]:.2f})")

    return log, hof

if __name__ == "__main__":
    # Ejecutamos la evolución
    log, hof = main_evolution(pop_size=20, gens=10)

    # Guardar CSV de la evolución
    df_log = pd.DataFrame(log)
    df_log.to_csv("evolution_log.csv", index=False)
    print("✅ Evolución guardada en evolution_log.csv")

    # Gráfica de fitness vs generación
    gens = df_log["gen"]
    avg  = df_log["avg"]
    mn   = df_log["min"]

    plt.figure()
    plt.plot(gens, avg, label="Fitness medio")
    plt.plot(gens, mn,  label="Fitness mínimo")
    plt.xlabel("Generación")
    plt.ylabel("MSE")
    plt.title("Evolución del fitness")
    plt.legend()
    plt.show()

    # Imprimir mejor individuo final
    best = hof[0]
    print(f"\n🔥 Mejor individuo final: n_hidden={best[0]}, lr=10^{best[1]:.2f} "
          f"=> MSE={best.fitness.values[0]:.6f}")
