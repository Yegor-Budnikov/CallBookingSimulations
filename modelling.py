import numpy as np
from math import comb

# Function to calculate P(X = x) using the convolution of binomial distributions
def calculate_distribution(n_list, p_list, max_clients):
    distribution = np.zeros(max_clients + 1)
    # Initialize distribution for the first channel
    for x in range(n_list[0] + 1):
        distribution[x] = comb(n_list[0], x) * (p_list[0] ** x) * ((1 - p_list[0]) ** (n_list[0] - x))
    # Convolve distributions for remaining channels
    for i in range(1, len(n_list)):
        new_distribution = np.zeros(max_clients + 1)
        for x in range(max_clients + 1):
            for y in range(n_list[i] + 1):
                if x + y <= max_clients:
                    new_distribution[x + y] += (
                        distribution[x]
                        * comb(n_list[i], y)
                        * (p_list[i] ** y)
                        * ((1 - p_list[i]) ** (n_list[i] - y))
                    )
        distribution = new_distribution
    # Normalize the distribution
    distribution /= np.sum(distribution)
    return distribution

# Function to calculate expected excess customers E[Y]
def expected_excess_customers(S, distribution):
    max_clients = len(distribution) - 1
    expectation = sum((x - S) * distribution[x] for x in range(S + 1, max_clients + 1))
    return expectation

# Function to simulate client arrivals based on the booking sequence and channel probabilities
def simulate_arrivals(sequence, p_list):
    return [1 if np.random.random() < p_list[channel] else 0 for channel in sequence]

# Simulation function with a customizable stop criterion
def simulate_clients_with_custom_stop(p_list, P_list, S, E_STOP=1, max_steps=1000):
    N = len(p_list)  # Number of channels
    n_list = [0] * N  # Initial number of booked clients per channel
    sequence = []  # Record the sequence of added clients
    stop_criteria_met = False

    for step in range(max_steps):
        # Randomly select a channel based on P_list
        channel = np.random.choice(N, p=P_list)
        n_list[channel] += 1  # Temporarily add the client for testing
        
        # Calculate distribution and expected excess customers
        max_clients = sum(n_list)
        distribution = calculate_distribution(n_list, p_list, max_clients)
        E_Y = expected_excess_customers(S, distribution)
        
        if E_Y >= E_STOP:
            # Remove the client if it violates the custom threshold
            n_list[channel] -= 1
            stop_criteria_met = True
            break

        # Add the client to the sequence if the criteria is not violated
        sequence.append(channel)

    return sequence, n_list, E_Y, stop_criteria_met

# Function to calculate metrics with reserve sellers
def calculate_metrics_with_reserve(arrivals, sequence, S, S_reserve):
    # Total number of arrivals
    total_arrivals = sum(arrivals)

    # Excess clients (if arrivals exceed sellers)
    excess_clients = max(0, total_arrivals - S)

    # Reserve sellers handling excess clients
    reserve_processed = min(excess_clients, S_reserve)
    remaining_excess_clients = max(0, excess_clients - S_reserve)

    # Idle sellers (if sellers exceed arrivals)
    idle_sellers = max(0, S - total_arrivals)

    return total_arrivals, remaining_excess_clients, idle_sellers, reserve_processed

# Full simulation function
def full_simulation_with_reserve(p_list, P_list, S, S_reserve, E_STOP, n_iterations=100):
    results = {
        "Total Booked": [],
        "Total Arrivals": [],
        "Excess Clients": [],
        "Idle Sellers": [],
        "Processed Clients": [],
        "Reserve Processed Clients": [],
    }

    for _ in range(n_iterations):
        # Step 1: Generate sequence of bookings with custom stop criterion
        sequence, final_n_list, final_E_Y, stopped = simulate_clients_with_custom_stop(p_list, P_list, S, E_STOP)
        
        # Step 2: Simulate arrivals
        arrivals = simulate_arrivals(sequence, p_list)
        
        # Step 3: Calculate metrics
        total_arrivals, remaining_excess_clients, idle_sellers, reserve_processed = calculate_metrics_with_reserve(
            arrivals, sequence, S, S_reserve
        )
        processed_clients = total_arrivals - remaining_excess_clients  # Clients who were processed

        # Record results
        results["Total Booked"].append(len(sequence))
        results["Total Arrivals"].append(total_arrivals)
        results["Excess Clients"].append(remaining_excess_clients)
        results["Idle Sellers"].append(idle_sellers)
        results["Processed Clients"].append(processed_clients)
        results["Reserve Processed Clients"].append(reserve_processed)
    
    # Aggregate metrics
    metrics = {
        "Booked Clients": {
            "Mean": np.mean(results["Total Booked"]),
            "Min": np.min(results["Total Booked"]),
            "Max": np.max(results["Total Booked"]),
        },
        "Arrived Clients": {
            "Mean": np.mean(results["Total Arrivals"]),
            "Min": np.min(results["Total Arrivals"]),
            "Max": np.max(results["Total Arrivals"]),
        },
        "Excess Clients": {
            "Mean": np.mean(results["Excess Clients"]),
            "Min": np.min(results["Excess Clients"]),
            "Max": np.max(results["Excess Clients"]),
            "Total": np.sum(results["Excess Clients"]),
        },
        "Idle Sellers": {
            "Mean": np.mean(results["Idle Sellers"]),
            "Min": np.min(results["Idle Sellers"]),
            "Max": np.max(results["Idle Sellers"]),
            "Total": np.sum(results["Idle Sellers"]),
        },
        "Processed Clients": {
            "Mean": np.mean(results["Processed Clients"]),
            "Min": np.min(results["Processed Clients"]),
            "Max": np.max(results["Processed Clients"]),
            "Total": np.sum(results["Processed Clients"]),
        },
        "Reserve Processed Clients": {
            "Mean": np.mean(results["Reserve Processed Clients"]),
            "Min": np.min(results["Reserve Processed Clients"]),
            "Max": np.max(results["Reserve Processed Clients"]),
            "Total": np.sum(results["Reserve Processed Clients"]),
        },
    }

    return metrics


p_list = [0.29, 0.26, 0.45, 0.43, 0.44, 0.31]
P_list = [0.627, 0.0551, 0.0469, 0.1053, 0.0364, 0.1293]
S = 30
S_reserve = 10
E_STOP = 0.75
N_ITERATIONS = 1000

simulation_results = full_simulation_with_reserve(p_list, P_list, S, S_reserve, E_STOP, N_ITERATIONS)
print(simulation_results)
