import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from deap import base, creator, tools, algorithms
from tqdm import trange
import random

# 1) Simulación de datos: Ornstein–Uhlenbeck (dx = -theta x dt + sigma dW)
def simulate_ou(theta=1.0, sigma=0.5, dt=0.01, steps=1000, paths=1000):
    x = np.zeros((paths, steps+1), dtype=np.float32)
    for t in range(steps):
        dw = np.random.normal(scale=np.sqrt(dt), size=paths)
        x[:, t+1] = x[:, t] - theta * x[:, t] * dt + sigma * dw
    return x

# Generar dataset (pares (x_t, x_{t+1})):
data = simulate_ou()
X = data[:, :-1].reshape(-1,1)
y = data[:, 1: ].reshape(-1,1)

# Dividir train/val
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_val,   y_val   = X[split:], y[split:]

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                          batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val)),
                          batch_size=256, shuffle=False)

# 2) Definición de la red (hiperparámetros a optimizar)
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
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    # Entrenamiento
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    # Evaluación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
    return val_loss / len(val_loader.dataset)

# 3) Setup DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # minimizamos MSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Individual: [n_hidden, log10_lr]
toolbox.register("n_hidden", random.randint, 5, 100)
toolbox.register("log_lr", lambda: random.uniform(-4, -1))
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_hidden, toolbox.log_lr), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(ind):
    n_hidden, log_lr = ind
    lr = 10**log_lr
    mse = train_and_evaluate(n_hidden=n_hidden, lr=lr)
    return (mse,)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=[50,-2.5], sigma=[20,1.0], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 4) Loop evolutivo
def main_evolution(pop_size=20, gens=10):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=0.5, mutpb=0.2,
                                   ngen=gens, stats=stats,
                                   halloffame=hof, verbose=True)
    best = hof[0]
    print(f"\n🔥 Mejor individuo: n_hidden={best[0]:.0f}, lr=10^{best[1]:.2f} => MSE={best.fitness.values[0]:.6f}")

if __name__ == "__main__":
    main_evolution()
