import matplotlib.pyplot as plt
import numpy as np

# Data for the "best" and "worst" cases (number of examples)
num_examples = np.array([10, 20, 40, 60, 100, 200, 400, 600, 800])

# Success rates (in percentages)
best_success_rates = np.array([50.75, 75.37, 73.13, 77.61, 75.37, 73.13, 81.34, 74.63, 78.36])
worst_success_rates = np.array([53.73, 61.94, 65.67, 73.13, 58.21, 69.40, 73.13, 72.39, 72.39])

# Calculate error rates as 100 - success rate
best_error_rates = 100 - best_success_rates
worst_error_rates = 100 - worst_success_rates

plt.figure(figsize=(10, 6))
plt.semilogx(num_examples, best_success_rates, marker='o', label='Best Success Rate')
plt.semilogx(num_examples, worst_success_rates, marker='o', label='Worst Success Rate')

plt.xlabel('Number of Examples (log scale)')
plt.ylabel('Success Rate (%)')
plt.ylim(0, 100)
plt.title('Success Rate vs. Number of Examples (Log-Log Plot)')
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig('bestworst_success_rate_semilog.png')
