#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt

def parse_scaling_results(filename):
    ms = []
    ns = []
    times = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip non-data lines, such as headers or divider lines
            # Check if the line starts with a digit (M)
            if re.match(r'^\d+', line):
                # Assume each line has at least 3 numbers: M, N, Time(s)
                parts = line.split()
                # parts look like ["500", "4", "0.000145", "1.446e-14", "5.586e-16"]
                if len(parts) >= 3:
                    m_val = int(parts[0])
                    n_val = int(parts[1])
                    time_val = float(parts[2])
                    ms.append(m_val)
                    ns.append(n_val)
                    times.append(time_val)
    return ms, ns, times

def main():
    # Read data from file
    filename = "scaling_results.txt"
    ms, ns, times = parse_scaling_results(filename)

    # Find all unique n values
    unique_ns = sorted(set(ns))

    # For each n, plot a (M, Time) curve
    plt.figure(figsize=(8, 6))
    for n_val in unique_ns:
        # Find the corresponding (m, time) pairs for this n value
        m_list = []
        t_list = []
        for i in range(len(ms)):
            if ns[i] == n_val:
                m_list.append(ms[i])
                t_list.append(times[i])
        # Sort by M in ascending order to avoid cluttered lines
        # Zip, sort, then unzip
        combined = sorted(zip(m_list, t_list), key=lambda x: x[0])
        m_sorted, t_sorted = zip(*combined)

        plt.plot(m_sorted, t_sorted, marker='o', label=f"N={n_val}")

    plt.xlabel("M (number of rows)")
    plt.ylabel("Time (seconds)")
    plt.title("TSQR Scaling: Time vs M for different N")
    plt.xscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
