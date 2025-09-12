# --- SCRIPT 1: CLASSICAL BENCHMARK (SELECTION + CVAR ALLOCATION) ---
import pandas as pd
import numpy as np
import cvxpy as cp
import time
import tracemalloc
import matplotlib.pyplot as plt
from scipy import stats
import argparse

# --- Step 1: Argument Parsing ---
parser = argparse.ArgumentParser(description="Classical Benchmark Script")
parser.add_argument(
    "--input_file",
    type=str,
    default="cleaned_daily_log_returns_100_assets_2025.csv",
    help="Path to the input CSV file",
)
parser.add_argument(
    "--max_universe_size",
    type=int,
    default=100,
    help="Maximum universe size to process (default: 100)",
)
args = parser.parse_args()

# --- Step 2: Load Cleaned Historical Data ---
try:
    returns_df = pd.read_csv(args.input_file, index_col=0, parse_dates=True)
    print("Successfully loaded historical returns data.")
    print(f"Data shape: {returns_df.shape}")
    print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# --- Configuration ---
TARGET_PORTFOLIO_SIZE = 15
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02
ALPHA = 0.95


# --- CVaR Optimization Function ---
def cvar_optimization(returns, alpha=0.95):
    """
    Perform CVaR optimization on the given returns.

    Parameters:
        returns (pd.DataFrame): Asset returns data.
        alpha (float): Confidence level for CVaR calculation.

    Returns:
        dict: Optimized weights for the assets, or None if optimization fails.
    """
    n_assets = returns.shape[1]
    n_samples = returns.shape[0]
    returns_data = returns.values
    weights = cp.Variable(n_assets)
    beta = cp.Variable()
    z = cp.Variable(n_samples)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0.05,
        weights <= 0.5,
        z >= 0,
        z >= -returns_data @ weights - beta,
    ]

    cvar = beta + (1 / (n_samples * (1 - alpha))) * cp.sum(z)

    try:
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve(solver=cp.ECOS, max_iters=1000)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return None

        if weights.value is None or np.any(np.isnan(weights.value)):
            return None

        normalized_weights = weights.value / np.sum(weights.value)
        return dict(zip(returns.columns, normalized_weights))
    except Exception as e:
        print(f"Error during CVaR optimization: {e}")
        return None


# --- Performance Metrics Calculation ---
def calculate_metrics(
    returns, risk_free=RISK_FREE_RATE, trading_days=TRADING_DAYS_PER_YEAR
):
    """
    Calculate financial performance metrics for a portfolio.

    Parameters:
        returns (pd.Series): Portfolio returns.
        risk_free (float): Risk-free rate for Sharpe and Sortino ratios.
        trading_days (int): Number of trading days in a year.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    metrics = {}
    metrics["Annual Return"] = returns.mean() * trading_days
    metrics["Annual Volatility"] = returns.std() * np.sqrt(trading_days)
    metrics["Sharpe Ratio"] = (metrics["Annual Return"] - risk_free) / metrics[
        "Annual Volatility"
    ]

    downside_returns = returns[returns < 0]
    downside_dev = (
        downside_returns.std() * np.sqrt(trading_days)
        if len(downside_returns) > 0
        else 0
    )
    metrics["Sortino Ratio"] = (
        (metrics["Annual Return"] - risk_free) / downside_dev
        if downside_dev > 0
        else np.nan
    )

    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    metrics["Max Drawdown"] = drawdowns.min()
    metrics["Calmar Ratio"] = (
        metrics["Annual Return"] / abs(metrics["Max Drawdown"])
        if metrics["Max Drawdown"] != 0
        else np.nan
    )

    metrics["VaR (95%)"] = returns.quantile(0.05)
    metrics["CVaR (95%)"] = returns[returns <= metrics["VaR (95%)"]].mean()

    return metrics


# --- Graph Plotting Functions ---
def plot_computational_performance(computational_results):
    """
    Plot computational performance metrics: Memory usage and execution time.

    Parameters:
        computational_results (list): List of dictionaries containing computational metrics.
    """
    universe_sizes = [result["Universe Size"] for result in computational_results]
    memory_usage = [result["Peak Memory (MB)"] for result in computational_results]
    execution_time = [result["Time (s)"] for result in computational_results]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Memory Usage Plot
    axs[0].plot(
        universe_sizes,
        memory_usage,
        marker="o",
        color="blue",
        label="Memory Usage (MB)",
    )
    axs[0].set_xlabel("Universe Size")
    axs[0].set_ylabel("Memory Usage (MB)")
    axs[0].set_title("Memory Usage vs Universe Size")
    axs[0].grid()
    axs[0].legend()

    # Execution Time Plot
    axs[1].plot(
        universe_sizes,
        execution_time,
        marker="o",
        color="green",
        label="Execution Time (s)",
    )
    axs[1].set_xlabel("Universe Size")
    axs[1].set_ylabel("Execution Time (s)")
    axs[1].set_title("Execution Time vs Universe Size")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def plot_financial_performance(financial_results):
    """
    Plot financial performance metrics: Annual Return, Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, CVaR.

    Parameters:
        financial_results (list): List of dictionaries containing financial metrics.
    """
    universe_sizes = [result["Universe Size"] for result in financial_results]
    metrics_to_plot = [
        ("Annual Return", "%"),
        ("Annual Volatility", "%"),
        ("Sharpe Ratio", ""),
        ("Sortino Ratio", ""),
        ("Calmar Ratio", ""),
        ("CVaR (95%)", "%"),
    ]

    fig, axs = plt.subplots(3, 2, figsize=(14, 12))

    for i, (metric, unit) in enumerate(metrics_to_plot):
        row, col = divmod(i, 2)
        values = [result[metric] for result in financial_results]

        # Convert percentages to 100-based scale for Annual Return, Volatility, and CVaR
        if unit == "%":
            values = [v * 100 for v in values]

        axs[row, col].plot(
            universe_sizes, values, marker="o", label=f"{metric} ({unit})"
        )
        axs[row, col].set_xlabel("Universe Size")
        axs[row, col].set_ylabel(f"{metric} ({unit})")
        axs[row, col].set_title(f"{metric} vs Universe Size")
        axs[row, col].grid()
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()


# --- Main Execution ---
print("\n--- Classical Benchmark Performance ---")
computational_results = []
financial_results = []
all_tickers = returns_df.columns.tolist()

# MODIFIED: Only include universe sizes divisible by 5
universe_sizes = [
    n for n in range(TARGET_PORTFOLIO_SIZE, min(len(all_tickers), args.max_universe_size) + 1)
    if n % 5 == 0
]

print(f"Running through universe sizes: {universe_sizes}")

for n_assets in universe_sizes:
    current_tickers = all_tickers[:n_assets]
    current_returns = returns_df[current_tickers]

    print(f"\n1. Universe Size: {n_assets}")

    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # Stage 1: Classical Asset Selection
        annualized_returns = current_returns.mean() * TRADING_DAYS_PER_YEAR
        annualized_volatility = current_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_ratios = (annualized_returns - RISK_FREE_RATE) / annualized_volatility
        sharpe_ratios = sharpe_ratios.replace([np.inf, -np.inf], np.nan).dropna()

        if len(sharpe_ratios) < TARGET_PORTFOLIO_SIZE:
            tracemalloc.stop()
            continue

        selected_assets = sharpe_ratios.nlargest(TARGET_PORTFOLIO_SIZE).index.tolist()
        print(f"2. Selected Assets: {selected_assets}")

        selected_returns = current_returns[selected_assets]

        # Stage 2: CVaR Weight Allocation
        cvar_weights = cvar_optimization(selected_returns, ALPHA)

        if cvar_weights is None:
            tracemalloc.stop()
            continue

        portfolio_returns = (selected_returns * pd.Series(cvar_weights)).sum(axis=1)
        metrics = calculate_metrics(portfolio_returns)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exec_time = time.perf_counter() - start_time

        print("3. Optimal Weights:")
        for ticker, weight in cvar_weights.items():
            print(f"   {ticker}: {weight:.2%}")

        print("4. Performance Metrics:")
        print(f"   Annual Return: {metrics['Annual Return']:.2%}")
        print(f"   Annual Volatility: {metrics['Annual Volatility']:.2%}")
        print(f"   Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"   Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['Max Drawdown']:.4f}")
        print(f"   Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
        print(f"   VaR (95%): {metrics['VaR (95%)']:.4f}")
        print(f"   CVaR (95%): {metrics['CVaR (95%)']:.4f}")

        print(f"5. Execution time: {exec_time:.4f} seconds")
        print(f"6. Memory usage: {peak_mem / (1024 * 1024):.2f} MB")

        computational_results.append(
            {
                "Universe Size": n_assets,
                "Time (s)": exec_time,
                "Peak Memory (MB)": peak_mem / (1024 * 1024),
            }
        )

        financial_results.append(
            {
                "Universe Size": n_assets,
                "Selected Assets": selected_assets,
                "Weights": cvar_weights,
                "Annual Return": metrics["Annual Return"],
                "Annual Volatility": metrics["Annual Volatility"],
                "Sharpe Ratio": metrics["Sharpe Ratio"],
                "Sortino Ratio": metrics["Sortino Ratio"],
                "Max Drawdown": metrics["Max Drawdown"],
                "Calmar Ratio": metrics["Calmar Ratio"],
                "VaR (95%)": metrics["VaR (95%)"],
                "CVaR (95%)": metrics["CVaR (95%)"],
            }
        )

    except Exception as e:
        print(f"Error during execution for universe size {n_assets}: {e}")
        tracemalloc.stop()
        continue
    finally:
        # Explicitly free memory
        del (
            current_tickers,
            current_returns,
            selected_assets,
            selected_returns,
            cvar_weights,
            portfolio_returns,
        )

# Plot the computational and financial metrics
plot_computational_performance(computational_results)
plot_financial_performance(financial_results)