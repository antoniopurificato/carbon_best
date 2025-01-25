import numpy as np
from copy import deepcopy
import random

def remove_samples_per_user(datasets, num_users, percentage):
    """
    Removes a percentage of unique samples (sids) per user from train, val, and test datasets.

    Args:
        datasets (dict): Dictionary containing 'train', 'val', and 'test' datasets. Each dataset is a list of user data.
        num_users (int): Number of users in the datasets.
        percentage (float): Percentage of unique samples to remove for each user.

    Returns:
        dict: A deep copy of the original dataset with the specified percentage of unique samples removed for each user.
    """
    random.seed(42)  # Ensure reproducibility
    modified_datasets = deepcopy(datasets)

    for user_index in range(num_users):
        # Extract unique sample IDs (sids) for train, val, and test sets
        train_sids = modified_datasets['train'][user_index]['sid']
        val_sids = modified_datasets['val'][user_index]['sid']
        test_sids = modified_datasets['test'][user_index]['sid']
        
        total_samples = len(train_sids)
        unique_sids = list(set(train_sids))  # Get unique sample IDs
        
        # Calculate the number of samples to remove
        num_samples_to_remove = min(len(unique_sids), int(total_samples * percentage))
        
        # Select random unique sample IDs to remove
        sids_to_remove = random.sample(unique_sids, num_samples_to_remove)
        
        # Remove the selected sample IDs from all three datasets
        for sid in sids_to_remove:
            train_sids.remove(sid)
            val_sids.remove(sid)
            test_sids.remove(sid)
    
    return modified_datasets

def calculate_avg_length(datasets, num_users):
    """
    Calculates the average length of the 'train' dataset across all users.

    Args:
        datasets (dict): Dictionary containing 'train', 'val', and 'test' datasets. Each dataset is a list of user data.
        num_users (int): Number of users in the dataset.

    Returns:
        float: The average number of samples per user in the 'train' dataset.
    """
    total_samples = 0
    for user_index in range(num_users):
        total_samples += len(datasets['train'][user_index]['sid'])
    return total_samples / num_users

def generate_learning_rates(start_order, end_order, num_samples, seed=42):
    """
    Generates a list of learning rates using a logarithmic scale with added noise.

    Args:
        start_order (float): Start exponent for log scale (e.g., -3 for 10^-3).
        end_order (float): End exponent for log scale (e.g., -5 for 10^-5).
        num_samples (int): Number of learning rates to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: Array of learning rates with added random noise.
    """
    if seed is not None:
        np.random.seed(seed)  # Set random seed for reproducibility
    
    # Generate logarithmically spaced values
    base_values = np.logspace(start_order, end_order, num_samples)
    
    # Add small random noise to the values
    learning_rates = base_values + np.random.rand(num_samples) * 1e-4
    return learning_rates
