import pandas as pd
import numpy as np
from math import comb

# Function to calculate P(X = x) using the convolution of binomial distributions
def calculate_distribution(n_list, p_list, max_clients):
    distribution = np.zeros(max_clients + 1)
    for x in range(n_list[0] + 1):
        distribution[x] = comb(n_list[0], x) * (p_list[0] ** x) * ((1 - p_list[0]) ** (n_list[0] - x))
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
    distribution /= np.sum(distribution)
    return distribution

# Function to calculate expected excess customers E[Y]
def expected_excess_customers(S, distribution):
    max_clients = len(distribution) - 1
    return sum((x - S) * distribution[x] for x in range(S + 1, max_clients + 1))

# Function to calculate metrics with reserve sellers
def calculate_metrics_with_reserve(arrivals, S, S_reserve):
    total_arrivals = sum(arrivals)
    excess_clients = max(0, total_arrivals - S)
    reserve_processed = min(excess_clients, S_reserve)
    remaining_excess_clients = max(0, excess_clients - S_reserve)
    idle_sellers = max(0, S - total_arrivals)
    processed_clients = total_arrivals - remaining_excess_clients
    return total_arrivals, processed_clients, idle_sellers, reserve_processed, remaining_excess_clients

# Function to simulate bookings from a file with proper sequence reset and no double-counting
def simulate(data, p_list, S, S_reserve, E_STOP):
    N = len(p_list)
    n_list = [0] * N
    simulations = []
    current_sequence = []
    metrics = []
    step = 0  # Step counter for debug logging

    for _, row in data.iterrows():
        channel = int(row['traffic_source'].split()[-1]) - 1
        n_list[channel] += 1
        arrived = int(row['was_at_meeting']) if not pd.isna(row['was_at_meeting']) else 0

        # Calculate distribution and expected excess customers BEFORE appending
        max_clients = sum(n_list)
        distribution = calculate_distribution(n_list, p_list, max_clients)
        E_Y = expected_excess_customers(S, distribution)

        if E_Y >= E_STOP:
            # Log metrics and reset BEFORE adding the new event
            arrivals = [arrival[1] for arrival in current_sequence]
            total_arrivals, processed_clients, idle_sellers, reserve_processed, remaining_excess_clients = calculate_metrics_with_reserve(
                arrivals, S, S_reserve
            )
            metrics.append({
                "Processed Clients": processed_clients,
                "Idle Sellers": idle_sellers,
                "Reserve Processed Clients": reserve_processed,
                "Excess Clients": remaining_excess_clients,
                "Final E[Y]": E_Y,
                "Arrived Clients": total_arrivals
            })
            simulations.append({
                "sequence": current_sequence.copy(),
                "final_n_list": n_list.copy(),
                "final_E_Y": E_Y,
                "stopped_due_to_criterion": True
            })
            current_sequence = []
            n_list = [0] * N

        # Append current event AFTER reset logic
        current_sequence.append((channel, arrived))
        step += 1

    if current_sequence:
        arrivals = [arrival[1] for arrival in current_sequence]
        total_arrivals, processed_clients, idle_sellers, reserve_processed, remaining_excess_clients = calculate_metrics_with_reserve(
            arrivals, S, S_reserve
        )
        metrics.append({
            "Processed Clients": processed_clients,
            "Idle Sellers": idle_sellers,
            "Reserve Processed Clients": reserve_processed,
            "Excess Clients": remaining_excess_clients,
            "Final E[Y]": E_Y,
            "Arrived Clients": total_arrivals
        })
        simulations.append({
            "sequence": current_sequence,
            "final_n_list": n_list,
            "final_E_Y": E_Y,
            "stopped_due_to_criterion": False
        })

    return simulations, metrics

# Summarize metrics across all simulations
def summarize_metrics(metrics):
    summary = {
        "Arrived Clients": {
            "Mean": np.mean([m["Arrived Clients"] for m in metrics]),
            "Min": np.min([m["Arrived Clients"] for m in metrics]),
            "Max": np.max([m["Arrived Clients"] for m in metrics]),
            "Total": np.sum([m["Arrived Clients"] for m in metrics]),
        },
        "Processed Clients": {
            "Mean": np.mean([m["Processed Clients"] for m in metrics]),
            "Min": np.min([m["Processed Clients"] for m in metrics]),
            "Max": np.max([m["Processed Clients"] for m in metrics]),
            "Total": np.sum([m["Processed Clients"] for m in metrics]),
        },
        "Idle Sellers": {
            "Mean": np.mean([m["Idle Sellers"] for m in metrics]),
            "Min": np.min([m["Idle Sellers"] for m in metrics]),
            "Max": np.max([m["Idle Sellers"] for m in metrics]),
            "Total": np.sum([m["Idle Sellers"] for m in metrics]),
        },
        "Reserve Processed Clients": {
            "Mean": np.mean([m["Reserve Processed Clients"] for m in metrics]),
            "Min": np.min([m["Reserve Processed Clients"] for m in metrics]),
            "Max": np.max([m["Reserve Processed Clients"] for m in metrics]),
            "Total": np.sum([m["Reserve Processed Clients"] for m in metrics]),
        },
        "Excess Clients": {
            "Mean": np.mean([m["Excess Clients"] for m in metrics]),
            "Min": np.min([m["Excess Clients"] for m in metrics]),
            "Max": np.max([m["Excess Clients"] for m in metrics]),
            "Total": np.sum([m["Excess Clients"] for m in metrics]),
        },
        "Final E[Y]": {
            "Mean": np.mean([m["Final E[Y]"] for m in metrics]),
            "Min": np.min([m["Final E[Y]"] for m in metrics]),
            "Max": np.max([m["Final E[Y]"] for m in metrics]),
        },
    }
    return summary

# Summarize booked clients across simulations
def summarize_booked_clients(simulations):
    booked_clients = [len(sim["sequence"]) for sim in simulations]
    summary = {
        "Booked Clients": {
            "Mean": np.mean(booked_clients),
            "Min": np.min(booked_clients),
            "Max": np.max(booked_clients),
            "Total": np.sum(booked_clients),
        }
    }
    return summary

# Example usage
file_path = 'Data for an interview - Sheet1.csv'
data = pd.read_csv(file_path)

# Parameters
p_list = [0.29, 0.26, 0.45, 0.43, 0.44, 0.31]
S = 30
S_reserve = 10
E_STOP = 0.75

# Run the simulation
simulations_results, metrics_results = simulate(data, p_list, S, S_reserve, E_STOP)

# Summarize the metrics
metrics_summary = summarize_metrics(metrics_results)
booked_clients_summary = summarize_booked_clients(simulations_results)

# Combine the summaries
metrics_summary.update(booked_clients_summary)

# Display the final summary
print(metrics_summary)
