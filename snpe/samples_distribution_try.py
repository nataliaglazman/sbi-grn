import csv
import numpy as np

# Generate samples from a normal distribution
mu, sigma = 0, 0.1  # mean and standard deviation
num_samples = 1000
samples = [np.random.normal(mu, sigma, num_samples), np.random.normal(10, 1, num_samples)]

# Write samples to a CSV file
with open('samples_try_2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sample'])
    for sample in samples:
        writer.writerow([sample[0], sample[1]])