# --- SCRIPT 3: MIQP BENCHMARK (COMPLEXITY ANALYSIS) ---
import pandas as pd
import numpy as np
import cvxpy as cp
import time
import tracemalloc
import matplotlib.pyplot as plt
import argparse
import math

# --- Step 1: Argument Parsing ---
parser = argparse.ArgumentParser(description="MIQP Complexity Analysis Script")
parser.add_argument(
    "--input_file",
    type=str,
    default="cleaned_daily_log_returns_100_assets_2008.csv",
    help="Path to the input CSV file",
)
parser.add_argument(
    "--max_universe_size",
    type=int,
    default=45, 
    help="Maximum universe size to process before termination (default: 45)",
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

# --- Complexity Analysis Configuration ---
# SIGNIFICANTLY INCREASED THRESHOLDS TO DEMONSTRATE EXPONENTIAL GROWTH
MAX_ALLOWED_TIME = 1800  # Increased from 300 to 1800 seconds (30 minutes)
TIME_THRESHOLD = 300     # Increased from 60 to 300 seconds (5 minutes)
MAX_MEMORY_MB = 8192     # Increased from 4096 to 8192 MB (8 GB)

# --- MIQP CVaR Optimization Function ---
def miqp_cvar_optimization(returns, target_cardinality, alpha=0.95):
    """
    Perform cardinality-constrained CVaR optimization using Mixed-Integer Quadratic Programming.
    """
    n_assets = returns.shape[1]
    n_samples = returns.shape[0]
    returns_data = returns.values

    # Define optimization variables
    weights = cp.Variable(n_assets)  # Continuous weights
    selection = cp.Variable(n_assets, boolean=True)  # Binary selection variables
    beta = cp.Variable()  # VaR variable
    z = cp.Variable(n_samples)  # Auxiliary variables for CVaR

    # Big-M constraint to link selection and weights
    M = 1.0

    # Constraints
    constraints = [
        cp.sum(weights) == 1,  # Fully invested
        weights >= 0,  # No short selling
        weights <= M * selection,  # Link weights to selection
        cp.sum(selection) == target_cardinality,  # Cardinality constraint
        z >= 0,
        z >= -returns_data @ weights - beta,  # CVaR constraint
    ]

    # Additional weight bounds if selected
    for i in range(n_assets):
        constraints.append(weights[i] <= 0.5)  # Maximum weight per asset
        constraints.append(weights[i] >= 0.05 * selection[i])  # Minimum weight if selected

    # Objective: Minimize CVaR
    cvar = beta + (1 / (n_samples * (1 - alpha))) * cp.sum(z)
    problem = cp.Problem(cp.Minimize(cvar), constraints)

    try:
        # Solve with MIQP solver - increased max_iters for larger problems
        problem.solve(solver=cp.ECOS_BB, max_iters=5000, verbose=False)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"  MIQP optimization failed with status: {problem.status}")
            return None, None

        # Extract results
        selected_mask = np.array(selection.value > 0.5)
        selected_assets = returns.columns[selected_mask].tolist()
        selected_weights = weights.value[selected_mask]
        
        # Normalize weights
        if np.sum(selected_weights) > 0:
            normalized_weights = selected_weights / np.sum(selected_weights)
            final_weights = dict(zip(selected_assets, normalized_weights))
            return selected_assets, final_weights
        else:
            return None, None

    except Exception as e:
        print(f"  MIQP optimization error: {str(e)}")
        return None, None

# --- Performance Metrics Calculation ---
def calculate_metrics(
    returns, risk_free=RISK_FREE_RATE, trading_days=TRADING_DAYS_PER_YEAR
):
    """
    Calculate financial performance metrics for a portfolio.
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
    Plot computational performance metrics.
    """
    if not computational_results:
        print("No computational results to plot.")
        return
        
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

def plot_complexity_analysis(computational_results):
    """
    Plot demonstrating the exponential computational complexity of the MIQP problem.
    """
    if not computational_results:
        print("No results to plot for complexity analysis.")
        return

    df = pd.DataFrame(computational_results)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Time on primary Y-axis
    color = 'tab:red'
    ax1.set_xlabel('Universe Size')
    ax1.set_ylabel('Time (s) - Log Scale', color=color)
    ax1.set_yscale('log')
    ax1.plot(df['Universe Size'], df['Time (s)'], marker='o', color=color, 
             label='Solution Time (s)', linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Create a second Y-axis for Number of Combinations
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Possible Portfolios (Log Scale)', color=color)
    ax2.set_yscale('log')
    ax2.plot(df['Universe Size'], df['Possible Portfolios'], marker='s', 
             color=color, linestyle='--', label='Possible Portfolios', 
             linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Exponential Growth of MIQP Computational Complexity\n(Selecting {TARGET_PORTFOLIO_SIZE} from N Assets)')
    fig.tight_layout()
    plt.show()

def plot_financial_performance(financial_results):
    """
    Plot financial performance metrics: Annual Return, Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, CVaR.

    Parameters:
        financial_results (list): List of dictionaries containing financial metrics.
    """
    if not financial_results:
        print("No financial results to plot.")
        return
        
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
print("\n=== MIQP COMPLEXITY ANALYSIS ===")
print("Demonstrating exponential growth of computational requirements")
print("=" * 50)

computational_results = []
financial_results = []
all_tickers = returns_df.columns.tolist()

# Test progressively larger universe sizes to show complexity growth
# Extended to larger sizes to better demonstrate exponential growth
universe_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45]
universe_sizes = [n for n in universe_sizes if n <= args.max_universe_size]

experiment_terminated = False

for n_assets in universe_sizes:
    if experiment_terminated:
        break
        
    current_tickers = all_tickers[:n_assets]
    current_returns = returns_df[current_tickers]

    possible_combinations = math.comb(n_assets, TARGET_PORTFOLIO_SIZE)
    print(f"\n--- Universe Size: {n_assets} ---")
    print(f"Possible portfolios: C({n_assets}, {TARGET_PORTFOLIO_SIZE}) = {possible_combinations:,}")

    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # Check if previous run exceeded thresholds
        if computational_results:
            last_result = computational_results[-1]
            if (last_result["Time (s)"] > TIME_THRESHOLD or 
                last_result["Peak Memory (MB)"] > MAX_MEMORY_MB * 0.75):
                print(f"  Previous run exceeded practical thresholds.")
                print(f"  Terminating experiment to avoid intractable runtimes.")
                experiment_terminated = True
                tracemalloc.stop()
                break

        print("  Running MIQP optimization...")
        selected_assets, cvar_weights = miqp_cvar_optimization(
            current_returns, TARGET_PORTFOLIO_SIZE, ALPHA
        )

        # Check for timeout during optimization
        exec_time_so_far = time.perf_counter() - start_time
        if exec_time_so_far > MAX_ALLOWED_TIME:
            print(f"  !!! SOLVER TIMEOUT !!! Exceeded max allowed time ({MAX_ALLOWED_TIME}s).")
            experiment_terminated = True
            tracemalloc.stop()
            break

        if cvar_weights is None:
            print("  MIQP optimization failed.")
            tracemalloc.stop()
            continue

        # Calculate performance metrics
        portfolio_returns = (current_returns[selected_assets] * pd.Series(cvar_weights)).sum(axis=1)
        metrics = calculate_metrics(portfolio_returns)

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exec_time = time.perf_counter() - start_time

        print(f"  Selected Assets: {selected_assets}")
        print("  Optimal Weights:")
        for ticker, weight in cvar_weights.items():
            print(f"    {ticker}: {weight:.2%}")

        print("  Performance Metrics:")
        print(f"    Annual Return: {metrics['Annual Return']:.2%}")
        print(f"    Annual Volatility: {metrics['Annual Volatility']:.2%}")
        print(f"    Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"    Sortino Ratio: {metrics['Sortino Ratio']:.2f}")
        print(f"    Max Drawdown: {metrics['Max Drawdown']:.4f}")
        print(f"    Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
        print(f"    VaR (95%): {metrics['VaR (95%)']:.4f}")
        print(f"    CVaR (95%): {metrics['CVaR (95%)']:.4f}")

        print(f"  Execution time: {exec_time:.2f} seconds")
        print(f"  Memory usage: {peak_mem / (1024 * 1024):.2f} MB")

        # Store computational results
        computational_results.append({
            "Universe Size": n_assets,
            "Time (s)": exec_time,
            "Peak Memory (MB)": peak_mem / (1024 * 1024),
            "Possible Portfolios": possible_combinations,
            "Status": "Success"
        })

        # Store financial results
        financial_results.append({
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
        })

        # Check if we should terminate after this successful run
        if exec_time > TIME_THRESHOLD:
            print(f"  Warning: Run time ({exec_time:.1f}s) exceeded threshold ({TIME_THRESHOLD}s).")
            print(f"  Next universe size will likely be intractable.")
            # Don't terminate immediately - allow one more run to show the jump
            if len(computational_results) > 3:  # Only terminate after we have enough data points
                experiment_terminated = True

    except MemoryError:
        print(f"  !!! MEMORY ERROR !!! Universe size {n_assets} exhausted available memory.")
        experiment_terminated = True
        tracemalloc.stop()
    except Exception as e:
        print(f"  Error for universe size {n_assets}: {e}")
        tracemalloc.stop()
        if "memory" in str(e).lower():
            experiment_terminated = True

# --- Results Summary ---
print("\n" + "="*60)
print("COMPLEXITY ANALYSIS SUMMARY")
print("="*60)

if computational_results:
    df_comp = pd.DataFrame(computational_results)
    print("\nComputational Performance:")
    print(df_comp[['Universe Size', 'Time (s)', 'Peak Memory (MB)', 'Possible Portfolios']].to_string(index=False))
    
    # Calculate complexity growth factors
    if len(df_comp) > 1:
        print(f"\nComplexity Growth Factors:")
        for i in range(1, len(df_comp)):
            time_growth = df_comp['Time (s)'].iloc[i] / df_comp['Time (s)'].iloc[i-1]
            comb_growth = df_comp['Possible Portfolios'].iloc[i] / df_comp['Possible Portfolios'].iloc[i-1]
            print(f"  {df_comp['Universe Size'].iloc[i-1]} -> {df_comp['Universe Size'].iloc[i]}: "
                  f"Time ×{time_growth:.1f}, Combinations ×{comb_growth:.1f}")

# Plot the results
print("\nGenerating complexity analysis plots...")
plot_computational_performance(computational_results)
plot_complexity_analysis(computational_results)
plot_financial_performance(financial_results)

print("\n=== EXPERIMENT COMPLETE ===")
if experiment_terminated:
    print("Experiment was terminated early due to computational constraints.")
    print("This demonstrates the practical intractability of large-scale MIQP problems.")
else:
    print("All planned universe sizes were completed successfully.")