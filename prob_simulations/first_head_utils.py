import numpy as np
import pandas as pd

def simulate_coin_tosses_until_head(p: float=0.5) -> int:
    """
    Simulate tossing a fair coin until we get a head.
    
    Parameters:
    p (float): Probability of getting heads (default 0.5 for fair coin)
    
    Returns:
    int: Number of tosses required to get the first head
    """

    from scipy.stats import bernoulli

    tosses = 0
    while True:
        tosses += 1
        # Generate a Bernoulli random variable (1 = head, 0 = tail)
        result = bernoulli.rvs(p)
        if result == 1:  # Got a head
            break
    return tosses

# Let's also verify our simulation with a more detailed analysis
def fair_coin_detailed_analysis(n_experiments=50000):
    """Perform detailed analysis of the coin toss simulation"""
    
    # Run the simulation
    tosses_results = [simulate_coin_tosses_until_head() for _ in range(n_experiments)]
    
    # Calculate statistics
    sample_mean = np.mean(tosses_results)
    sample_var = np.var(tosses_results, ddof=1)
    sample_std = np.std(tosses_results, ddof=1)
    
    # Theoretical values
    theoretical_mean = 2.0
    theoretical_var = 2.0
    theoretical_std = np.sqrt(2.0)
    
    # Create summary table
    summary_df = pd.DataFrame({
        'Metric': ['Mean', 'Variance', 'Standard Deviation'],
        'Theoretical': [theoretical_mean, theoretical_var, theoretical_std],
        'Simulated': [sample_mean, sample_var, sample_std],
        'Difference': [abs(sample_mean - theoretical_mean), 
                      abs(sample_var - theoretical_var),
                      abs(sample_std - theoretical_std)]
    })
    
    print(f"Simulation Results (n = {n_experiments:,} experiments)")
    print("=" * 60)
    print(summary_df.round(4))
    
    # Calculate frequency of each outcome
    unique, counts = np.unique(tosses_results, return_counts=True)
    frequencies = counts / len(tosses_results)
    
    print(f"\nTop 10 most frequent outcomes:")
    print("-" * 40)
    for i in range(min(10, len(unique))):
        theoretical_prob = (0.5)**(unique[i]-1) * 0.5
        print(f"{unique[i]} tosses: {frequencies[i]:.4f} (theoretical: {theoretical_prob:.4f})")
    
    return tosses_results, summary_df
