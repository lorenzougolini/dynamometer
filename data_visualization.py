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

z_scores = zscore(force_values)
threshold = 2
substituted_values_nearest = force_values[:]
for i, z in enumerate(z_scores):
    if abs(z) > threshold:
        # Find the nearest non-outlier value by looking left and right
        left = i - 1 if i > 0 else i
        right = i + 1 if i < len(force_values) - 1 else i
        nearest_value = force_values[left] if abs(force_values[left] - force_values[i]) < abs(force_values[right] - force_values[i]) else force_values[right]
        substituted_values_nearest[i] = nearest_value

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(time_values, substituted_values_nearest, marker='x', linestyle='-', color='r', label="Recorded Data")
plt.xlabel("Time (seconds)")
plt.ylabel("Force (Kgs)")
plt.title("Recorded Force over Time")
plt.grid(True)
plt.legend()
plt.show()