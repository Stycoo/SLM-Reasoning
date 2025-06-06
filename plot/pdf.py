# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters for the normal distribution
# mu, sigma = 0, 1  
# data = np.random.normal(mu, sigma, 1000)

# # Plot histogram normalized to a probability density
# plt.hist(data, bins=30, density=True, alpha=0.5)

# # Compute and overlay the theoretical PDF
# x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
# pdf = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
# plt.plot(x, pdf)

# # Labels and title
# plt.title("Probability Density Plot")
# plt.xlabel("Value")
# plt.ylabel("Density")

# # Save the figure to disk
# output_path = 'plot/figs/probability_density_plot.png'
# plt.savefig(output_path, dpi=300)

import matplotlib.pyplot as plt
import seaborn as sns

# Example list: values from 0 to 10
data = list(range(11))  # [0, 1, 2, ..., 10]

plt.figure(figsize=(8, 4))
sns.kdeplot(data, fill=True, bw_adjust=0.5)  # bw_adjust controls smoothness

plt.title("Density Distribution of Integer List")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)
# plt.show()

output_path = 'plot/figs/probability_density_plot.png'
plt.savefig(output_path, dpi=300)
