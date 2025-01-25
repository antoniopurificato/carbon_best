import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_ndcg_sensitivity(ndcg_df):
    """
    Plot the NDCG evolution for varying weights.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ndcg_df["acc_w"], ndcg_df["ndcg"], marker="o", label="NDCG")
    plt.xlabel("Accuracy_weight(1-Emissions_w)")
    plt.ylabel("NDCG")
    plt.title("NDCG with varying weights")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_pareto_fronts(ground_truth_pareto, predicted_pareto):
    plt.scatter(ground_truth_pareto["true_EM"], ground_truth_pareto["true_ACC"], color="blue", label="True Pareto Front")
    plt.scatter(predicted_pareto["true_EM"], predicted_pareto["true_ACC"], color="red", label="Predicted Pareto Front")
    plt.xlabel("True Emissions")
    plt.ylabel("True Accuracy")
    plt.title("Comparison of True and Predicted Pareto Fronts")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_absolute_difference(n_samples, colors, item_to_plot, labels):
    # Plot absolute differences for  VAL_ACC
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        color = colors[i % len(colors)]
        time_steps = np.arange(item_to_plot.shape[1])

        # Plot absolute differences
        plt.plot(time_steps, item_to_plot[i], color=color, label=f"Exp: {labels[i]}")


    plt.title("Absolute Difference - VAL_ACC")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Difference")
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid(True)
    plt.show()

def plot_losses(losses_df):
    plt.figure(figsize=(12, 6))
    plt.plot(losses_df["epoch"], losses_df["val_loss"], label="Validation Loss", marker="o")
    plt.plot(losses_df["epoch"], losses_df["train_loss"], label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot individual components if needed
    plt.figure(figsize=(12, 6))
    plt.plot(losses_df["epoch"], losses_df["val_acc_loss"], label="Validation Accuracy Loss", marker="o")
    plt.plot(losses_df["epoch"], losses_df["train_acc_loss"], label="Training Accuracy Loss", marker="o")
    plt.plot(losses_df["epoch"], losses_df["val_em_loss"], label="Validation EM Loss", marker="o")
    plt.plot(losses_df["epoch"], losses_df["train_em_loss"], label="Training EM Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Individual Loss Components Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)