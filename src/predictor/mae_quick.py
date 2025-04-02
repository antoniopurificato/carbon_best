import pandas as pd
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('src/results_csv/cifar10_476.csv')  

# Compute MAE for ACC and EN
mae_acc = mean_absolute_error(df['true_ACC'], df['predicted_ACC'])
mae_en = mean_absolute_error(df['true_EN'], df['predicted_EN'])

print(f"MAE for ACC: {mae_acc:.4f}")
print(f"MAE for EN:  {mae_en:.4f}")
