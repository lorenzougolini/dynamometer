import csv
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Read data from the CSV file
time_values = []
force_values = []

with open("datafile.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for time_str, value_str in row.items():
            time_values.append(float(time_str))
            force_values.append(float(value_str))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time_values, force_values, marker='x', linestyle='-', color='r', label="Recorded Data")
plt.xlabel("Time (seconds)")
plt.ylabel("Force (Kgs)")
plt.title("Recorded Force over Time")
plt.grid(True)
plt.legend()
plt.show()