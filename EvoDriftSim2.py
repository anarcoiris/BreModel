import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from scipy.optimize import minimize

class DriftDiffusionSimulator:
    def __init__(self, historical_file, test_file):
        self.historical_file = historical_file
        self.test_file = test_file
        self.historical_df = None
        self.test_df = None
        self.n_days = None
        self.last_price = None
        self.last_date = None
        self.dt = 1
        self.simulations = 100

    def load_data(self):
        with open(self.historical_file, "r") as f:
            raw_data = json.load(f)
        df = pd.DataFrame(raw_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df["close"] = df["close"].astype(float)
        df.sort_index(inplace=True)

        self.historical_df = df
        self.last_price = df["close"].iloc[-1]
        self.last_date = df.index[-1]

        with open(self.test_file, "r") as f:
            test_data = json.load(f)
        df_test = pd.DataFrame(test_data)
        df_test["timestamp"] = pd.to_datetime(df_test["timestamp"], unit="ms")
        df_test.set_index("timestamp", inplace=True)
        df_test["close"] = df_test["close"].astype(float)
        df_test.sort_index(inplace=True)

        # Continuidad temporal
        df_test = df_test[df_test.index > self.last_date]

        # Ajustamos test al horizonte predicho
        self.n_days = min(len(df_test), 30)
        self.test_df = df_test.iloc[:self.n_days]

    def simulate_paths(self, mu, sigma):
        paths = np.zeros((self.n_days, self.simulations))
        paths[0] = self.last_price
        for t in range(1, self.n_days):
            z = np.random.normal(0, 1, self.simulations)
            paths[t] = paths[t - 1] * np.exp(mu * self.dt + sigma * np.sqrt(self.dt) * z)
        return paths

    def evaluate_model(self, mu_sigma):
        mu, sigma = mu_sigma
        paths = self.simulate_paths(mu, sigma)
        mean_pred = paths.mean(axis=1)

        # Seguridad: igualamos longitudes
        test_prices = self.test_df["close"].values[:len(mean_pred)]

        mse = np.mean((test_prices - mean_pred) ** 2)
        return mse

    def fit_model(self):
        initial_guess = [0.0, 0.01]
        bounds = [(-1, 1), (1e-6, 1)]
        result = minimize(self.evaluate_model, initial_guess, bounds=bounds)
        return result.x, result.fun

    def plot_results(self, mu, sigma):
        paths = self.simulate_paths(mu, sigma)
        mean_pred = paths.mean(axis=1)
        lower = np.percentile(paths, 10, axis=1)
        upper = np.percentile(paths, 90, axis=1)
        future_dates = self.test_df.index[:self.n_days]

        plt.figure(figsize=(20, 10))
        plt.plot(self.historical_df["close"], label="Histórico", color="black")
        for i in range(self.simulations):
            plt.plot(future_dates, paths[:, i], color="orange", alpha=0.2)
        plt.plot(future_dates, mean_pred, label="Predicción media", color="red")
        plt.scatter(self.last_date, self.last_price, color='purple', s=50, label='Último precio histórico')
        plt.plot(self.test_df["close"], label="Real futuro", color="green")
        plt.fill_between(future_dates, lower, upper, color="red", alpha=0.2, label="Rango 10–90%")
        plt.axvline(self.last_date, linestyle="--", color="gray")
        plt.title(f"Simulación Optimizada | μ = {mu:.4f}, σ = {sigma:.4f}")
        plt.xlabel("Fecha")
        plt.ylabel("Precio (USD)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("EvoSimulation_Result.png")
        plt.show()

if __name__ == "__main__":
    sim = DriftDiffusionSimulator("BTCUSDT_1D_train.json", "BTCUSDT_test_1D.json")
    sim.load_data()
    best_params, final_mse = sim.fit_model()
    print(f"Mejores parámetros encontrados: μ = {best_params[0]:.6f}, σ = {best_params[1]:.6f}, MSE = {final_mse:.2f}")
    sim.plot_results(best_params[0], best_params[1])
