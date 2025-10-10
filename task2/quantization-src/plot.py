import matplotlib.pyplot as plt
import pandas as pd

# Read data from the CSV file
df = pd.read_csv('../results/evaluation_data.csv')

# Plotting
plt.figure(figsize=(10, 6))

# Plot each metric
plt.plot(df['Model'], df['Accuracy'], label='Accuracy', marker='o')
plt.plot(df['Model'], df['Precision'], label='Precision', marker='o')
plt.plot(df['Model'], df['Recall'], label='Recall', marker='o')
plt.plot(df['Model'], df['F1'], label='F1', marker='o')

# Customizing the plot
plt.title('Model Evaluation Metrics Comparison')
plt.xlabel('Model')
plt.ylabel('Metric Value')
plt.xticks(rotation=45, ha='right')
plt.legend()

# Show plot
plt.tight_layout()
plt.savefig("./model_metrics_comparison.png")
