# --- SCRIPT 2: COMMUNAL QUBO (SELECTION + ALLOCATION) ---
import pandas as pd
import numpy as np
import cvxpy as cp
import time
import tracemalloc
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg as la
from pyqubo import Array, Constraint
import neal
from joblib import Parallel, delayed

# --- Step 1: Load Your Cleaned Historical Data ---
try:
    returns_df = pd.read_csv(
        "cleaned_daily_log_returns_100_assets_2008.csv", index_col=0, parse_dates=True
    )
    print("Successfully loaded historical returns data.")
    print(f"Data shape: {returns_df.shape}")
    print(f"Date range: {returns_df.index.min()} to {returns_df.index.max()}")
    print("Assets:", returns_df.columns.tolist())
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit()

# --- Configuration ---
TARGET_PORTFOLIO_SIZE = 5
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02
ALPHA = 0.95  # Confidence level for CVaR

# --- CVaR Optimization Function ---
def cvar_optimization(returns, alpha=0.95):
    """
    Perform CVaR optimization using direct CVXPY implementation.
    """
    n_assets = returns.shape[1]
    n_samples = returns.shape[0]

    if n_assets == 0:
        print("CVaR optimization called with no assets.")
        return None

    returns_data = returns.values
    weights = cp.Variable(n_assets)
    beta = cp.Variable()
    z = cp.Variable(n_samples)

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0.05,  # Min 5% allocation
        weights <= 0.5,   # Max 50% allocation
        z >= 0,
        z >= -returns_data @ weights - beta
    ]

    cvar = beta + (1 / (n_samples * (1 - alpha))) * cp.sum(z)

    try:
        problem = cp.Problem(cp.Minimize(cvar), constraints)
        problem.solve(solver=cp.ECOS, max_iters=1000)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"CVaR optimization failed with status: {problem.status}")
            return None

        if weights.value is None or np.any(np.isnan(weights.value)):
            print("CVaR optimization returned invalid weights")
            return None

        normalized_weights = weights.value / np.sum(weights.value)
        return dict(zip(returns.columns, normalized_weights))

    except Exception as e:
        print(f"CVaR optimization error: {str(e)}")
        return None

# --- Performance Metrics Calculation ---
def calculate_metrics(returns, risk_free=RISK_FREE_RATE, trading_days=TRADING_DAYS_PER_YEAR):
    """
    Calculate financial performance metrics for a portfolio.
    """
    metrics = {}
    metrics["Annual Return"] = returns.mean() * trading_days
    metrics["Annual Volatility"] = returns.std() * np.sqrt(trading_days)
    metrics["Sharpe Ratio"] = (metrics["Annual Return"] - risk_free) / metrics["Annual Volatility"]
    downside_returns = returns[returns < 0]
    downside_dev = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 0
    metrics["Sortino Ratio"] = (metrics["Annual Return"] - risk_free) / downside_dev if downside_dev > 0 else np.nan
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    metrics["Max Drawdown"] = drawdowns.min()
    metrics["Calmar Ratio"] = metrics["Annual Return"] / abs(metrics["Max Drawdown"]) if metrics["Max Drawdown"] != 0 else np.nan
    metrics["VaR (95%)"] = returns.quantile(0.05)
    metrics["CVaR (95%)"] = returns[returns <= metrics["VaR (95%)"]].mean()
    return metrics

# --- QUBO Optimization for Asset Selection ---
def solve_qubo_for_community(community_assets, mean_returns, cov_matrix):
    """
    Select the best asset from a community using QUBO optimization.
    """
    num_assets = len(community_assets)
    if num_assets == 1:
        return community_assets[0]

    try:
        x = Array.create('x', shape=num_assets, vartype='BINARY')
        mu = mean_returns.loc[community_assets].values
        sigma = cov_matrix.loc[community_assets, community_assets].values
        lambda_param = 0.5
        return_term = sum(mu[i] * x[i] for i in range(num_assets))
        risk_term = sum(sigma[i, j] * x[i] * x[j] for i in range(num_assets) for j in range(num_assets))
        hamiltonian = -return_term + lambda_param * risk_term
        constraint = Constraint((sum(x) - 1)**2, label="one_asset")
        model = (hamiltonian + constraint).compile()
        qubo, _ = model.to_qubo()
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(qubo, num_reads=100)
        best_solution = sampleset.first.sample
        
        selected_index = -1
        for i in range(num_assets):
            if best_solution.get(f'x[{i}]', 0) == 1:
                selected_index = i
                break
        
        if selected_index == -1: # Fallback if QUBO fails
             sharpe_ratios = mu / np.sqrt(np.diag(sigma))
             selected_index = np.argmax(sharpe_ratios)
             
        return community_assets[selected_index]

    except Exception as e:
        print(f"QUBO optimization failed for community {community_assets}: {str(e)}")
        mu = mean_returns.loc[community_assets].values
        return community_assets[np.argmax(mu)] # Fallback to highest return

# --- Community Detection and Asset Selection ---
def detect_communities_and_select_assets(returns_df, target_communities):
    """
    Detect communities and select the best asset from each using QUBO.
    """
    corr_matrix = returns_df.corr()
    distance_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix.values, 0)
    graph = ig.Graph.Weighted_Adjacency(distance_matrix.values, mode='undirected', attr="weight", loops=False)

    # --- Resolution Search Loop ---
    print(f"  Searching for a resolution to find {target_communities} communities...")
    best_partition = None
    resolution = 1.0  # Initial guess
    
    # Try to find the exact number of communities
    for _ in range(20): # Limit search to 20 iterations
        partition = la.find_partition(
            graph, 
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=resolution,
            seed=42
        )
        num_communities = len(partition)
        
        if num_communities == target_communities:
            print(f"  Found exactly {num_communities} communities with resolution {resolution:.2f}.")
            best_partition = partition
            break
        elif num_communities > target_communities:
            resolution -= 0.1 # Decrease resolution for fewer communities
        else: # num_communities < target_communities
            resolution += 0.1 # Increase resolution for more communities

    # If exact number not found, find closest partition and merge if necessary
    if best_partition is None:
        print("  Could not find exact number of communities. Finding closest and merging.")
        # Search for a partition that has MORE than the target to allow merging
        resolution = 1.0
        for _ in range(20):
            partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution, seed=42)
            if len(partition) >= target_communities:
                best_partition = partition
                break
            resolution += 0.2
        
        if best_partition is None:
             print("  Failed to find a suitable community partition. Skipping.")
             return None, None
             
    # --- Organize assets by community ---
    communities = {}
    for idx, comm_id in enumerate(best_partition.membership):
        asset_name = returns_df.columns[idx]
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(asset_name)

    # If we have more communities than the target, merge the smallest ones
    if len(communities) > target_communities:
        print(f"  Detected {len(communities)} communities. Merging smallest to reach {target_communities}.")
        communities_list = sorted(communities.values(), key=len)
        while len(communities_list) > target_communities:
            smallest_comm = communities_list.pop(0)
            next_smallest_comm = communities_list[0]
            communities_list[0] = next_smallest_comm + smallest_comm
        communities = {i: assets for i, assets in enumerate(communities_list)}

    if len(communities) < target_communities:
        print(f"  Could not form {target_communities} communities. Found only {len(communities)}.")
        return None, None
        
    # --- Select best asset from each community ---
    mean_returns = returns_df.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
    
    selected_assets = Parallel(n_jobs=-1)(
        delayed(solve_qubo_for_community)(assets, mean_returns, cov_matrix) 
        for assets in communities.values()
    )
    
    return communities, list(set(selected_assets))


# --- Main Pipeline Function ---
def run_communal_qubo_pipeline(returns_df, target_portfolio_size):
    """
    Complete communal QUBO pipeline: community detection + QUBO selection + CVaR allocation
    """
    communities, selected_assets = detect_communities_and_select_assets(returns_df, target_portfolio_size)
    
    if selected_assets is None or len(selected_assets) < target_portfolio_size:
        print(f"  Asset selection failed to produce {target_portfolio_size} assets.")
        return None, communities, selected_assets
    
    selected_returns = returns_df[selected_assets]
    cvar_weights = cvar_optimization(selected_returns, ALPHA)
    
    return cvar_weights, communities, selected_assets

# --- Step 2: Assess Computational and Financial Performance ---
print("\n--- Assessing Performance of Communal QUBO + CVaR ---")

computational_results = []
financial_results = []
all_tickers = returns_df.columns.tolist()
universe_sizes = [
    n for n in range(TARGET_PORTFOLIO_SIZE, len(all_tickers) + 1)
    if n % 5 == 0
]

for n_assets in universe_sizes:
    current_tickers = all_tickers[:n_assets]
    current_returns = returns_df[current_tickers]
    
    print(f"\n1. Universe Size: {n_assets}")
    
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        cvar_weights, communities_found, selected_assets = run_communal_qubo_pipeline(
            current_returns, TARGET_PORTFOLIO_SIZE
        )
        
        # Check for failures in the pipeline
        if cvar_weights is None or selected_assets is None or len(selected_assets) < TARGET_PORTFOLIO_SIZE:
            print("  Communal QUBO pipeline failed for this universe size, skipping.")
            tracemalloc.stop()
            continue
            
        portfolio_returns = (current_returns[selected_assets] * pd.Series(cvar_weights)).sum(axis=1)
        metrics = calculate_metrics(portfolio_returns)
        
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exec_time = time.perf_counter() - start_time

        print(f"2. Selected Assets: {selected_assets}")
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

        computational_results.append({
            "Universe Size": n_assets,
            "Time (s)": exec_time,
            "Peak Memory (MB)": peak_mem / (1024 * 1024),
        })
        
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
            "CVaR (95%)": metrics["CVaR (95%)"]
        })
        
    except Exception as e:
        print(f"  An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        tracemalloc.stop()
        continue

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
    axs[0].plot(universe_sizes, memory_usage, marker='o', color='blue', label="Memory Usage (MB)")
    axs[0].set_xlabel("Universe Size")
    axs[0].set_ylabel("Memory Usage (MB)")
    axs[0].set_title("Memory Usage vs Universe Size")
    axs[0].grid()
    axs[0].legend()

    # Execution Time Plot
    axs[1].plot(universe_sizes, execution_time, marker='o', color='green', label="Execution Time (s)")
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

        axs[row, col].plot(universe_sizes, values, marker='o', label=f"{metric} ({unit})")
        axs[row, col].set_xlabel("Universe Size")
        axs[row, col].set_ylabel(f"{metric} ({unit})")
        axs[row, col].set_title(f"{metric} vs Universe Size")
        axs[row, col].grid()
        axs[row, col].legend()

    plt.tight_layout()
    plt.show()

plot_computational_performance(computational_results)
plot_financial_performance(financial_results)