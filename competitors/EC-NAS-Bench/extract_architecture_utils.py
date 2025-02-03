import sys
import os
import json
import random
import pickle
import csv
from pathlib import Path
from math import inf
import pandas as pd
from collections import defaultdict
from typing import List
import numpy as np

random.seed(0)

####### Taken and adapted from ecnas/baselines/core/pareto.py and ecnas/ecnas/utils/data/experiments.py #######

def pareto(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    # print(costs.shape)
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto


def pareto_index(costs: np.ndarray, index_list):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)

    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self

    index_return = index_list[is_pareto]

    return is_pareto, index_return


def nDS_index(costs, index_list):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :list of indeces
    :return: list of all fronts, sorted indeces
    """

    dominating_list = []
    index_return_list = []
    fronts = []
    while costs.size > 0:
        dominating, index_return = pareto_index(costs, index_list)
        fronts.append(costs[dominating])
        costs = costs[~dominating]
        index_list = index_list[~dominating]
        dominating_list.append(dominating)
        index_return_list.append(index_return)

    return fronts, index_return_list


def crowdingDist(fronts, index_list):
    """
    Implementation of the crowding distance
    :param front: (n_points, m_cost_values) array
    :return: sorted_front and corresponding distance value of each element in the sorted_front
    """
    dist_list = []
    index_return_list = []

    for g in range(len(fronts)):
        front = fronts[g]
        index_ = index_list[g]

        sorted_front = np.sort(front.view([("", front.dtype)] * front.shape[1]), axis=0).view(np.float)

        _, sorted_index = (list(t) for t in zip(*sorted(zip([f[0] for f in front], index_))))

        normalized_front = np.copy(sorted_front)

        for column in range(normalized_front.shape[1]):
            ma, mi = np.max(normalized_front[:, column]), np.min(normalized_front[:, column])
            normalized_front[:, column] -= mi
            normalized_front[:, column] /= ma - mi

        dists = np.empty((sorted_front.shape[0],), dtype=np.float)
        dists[0] = np.inf
        dists[-1] = np.inf

        for elem_idx in range(1, dists.shape[0] - 1):
            dist_left = np.linalg.norm(normalized_front[elem_idx] - normalized_front[elem_idx - 1])
            dist_right = np.linalg.norm(normalized_front[elem_idx + 1] - normalized_front[elem_idx])
            dists[elem_idx] = dist_left + dist_right

        dist_list.append((sorted_front, dists))
        _, index_sorted_max = (list(t) for t in zip(*sorted(zip(dists, sorted_index))))
        index_sorted_max.reverse()

        index_return_list.append(index_sorted_max)

    return dist_list, index_return_list


def nDS(costs: np.ndarray):
    """
    Implementation of the non-dominated sorting method
    :param costs: (n_points, m_cost_values) array
    :return: list of all fronts
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # Stepwise compute the pareto front without all prior dominating points
    my_costs = costs.copy()
    remain = np.ones(len(my_costs), dtype=np.bool)
    fronts = []
    while np.any(remain):
        front_i = pareto(my_costs)
        fronts.append(my_costs[front_i, :])
        my_costs[front_i, :] = np.inf
        remain = np.logical_and(remain, np.logical_not(front_i))
    return fronts


def computeHV2D(front: np.ndarray, ref: List[float]):
    """
    Compute the Hypervolume for the pareto front  (only implement it for 2D)
    :param front: (n_points, m_cost_values) array for which to compute the volume
    :param ref: coordinates of the reference point
    :returns: Hypervolume of the polygon spanned by all points in the front + the reference point
    """

    front = np.asarray(front)
    assert front.ndim == 2
    assert len(ref) == 2

    # We assume all points already sorted
    list_ = [ref]
    for x in front:
        elem_at = len(list_) - 1
        list_.append([list_[elem_at][0], x[1]])  # add intersection points by keeping the x constant
        list_.append(x)
    list_.append([list_[-1][0], list_[0][1]])
    sorted_front = np.array(list_)

    def shoelace(x_y):  # taken from https://stackoverflow.com/a/58515054
        x_y = np.array(x_y)
        x_y = x_y.reshape(-1, 2)

        x = x_y[:, 0]
        y = x_y[:, 1]

        S1 = np.sum(x * np.roll(y, -1))
        S2 = np.sum(y * np.roll(x, -1))

        area = 0.5 * np.absolute(S1 - S2)

        return area

    return shoelace(sorted_front)


# In[2]:



def load_pickle_data(fn):
    return pickle.load(open(fn, "rb"))


def filter_data(data, dvs):
    return data[data["mean"].isin(dvs) == False]


def convert_data(data):
    accs = data[data["metric_name"] == "val_acc"]["mean"].values
    energies = data[data["metric_name"] == "energy"]["mean"].values
    
    accs *= 0.01  # Convert to percentage
    energies *= 1000  # Convert to mJ
    accs = accs.tolist()
    energies = energies.tolist()
    app_accs = accs.copy()
    app_energies = energies.copy()
    for i in range(len(accs)):

        if abs(accs[i]) < 0.75:
            app_accs.remove(accs[i])
            app_energies.remove(energies[i])  

    accs = np.array(app_accs)
    energies = np.array(app_energies)
    return np.array([accs, energies]).T


def create_front(front, max_acc):
    if max_acc:
        front = front[front[:, 0].argsort()[::-1]]
        front = front[-1:, :]
    else:
        is_front = pareto(front)
        front = front[is_front == 1]
    return front


def load_experiments(fn, n_trials, dummy_values=None, max_acc=False):
    base_fn = fn + "_"
    max_samples = 0
    dvs = [inf, -inf, 0.0, 1.0] if dummy_values is None else dummy_values + [inf, -inf, 0.0, 1.0]
    all_experiments = []
    for i in range(n_trials):
        data = load_pickle_data(base_fn + str(i) + ".pickle")

        if dummy_values is None:
            data = filter_data(data, dvs)
        front = convert_data(data)
        front = create_front(front, max_acc)
        sols = list(zip(front[:, 0], front[:, 1]))
        all_experiments.append(sols)
        if len(sols) > max_samples:
            max_samples = len(sols)
    
    dummy_pair = (0.0, 0.0) if dummy_values is None else tuple(dummy_values[:2])

    for i in range(n_trials):
        diff = max_samples - len(all_experiments[i])
        all_experiments[i].extend([dummy_pair] * diff)
    X = np.ndarray((n_trials, max_samples, 2))

    for i in range(n_trials):
        X[i] = np.array(all_experiments[i])
    return X


def load_experiments_full(fn, n_trials, dummy_values=None):
    return load_experiments(fn, n_trials, dummy_values, max_acc=False)


def load_not_front(fn, n_trials, dummy_values=None):
    return load_experiments(fn, n_trials, dummy_values, max_acc=True)




################# OUR UTILS FUNCTIONS #################

def obtain_max_balanced_from_pareto(pareto_results):
    couples = [item for sublist in pareto_results for item in sublist]

    filtered_couples = [c for c in couples if not (c[0] == 0 or c[1] == 0)] #remove couples with 0
    original_couples = np.array(filtered_couples)

    # Get the abs value of the first element
    for c in filtered_couples:
        c[0] = abs(c[0])

    # Normalize the second element
    second_elements = [c[1] for c in filtered_couples]
    min_second = min(second_elements)
    max_second = max(second_elements)
    for c in filtered_couples:
        c[1] = (c[1] - min_second) / (max_second - min_second)


    filtered_couples = np.array(filtered_couples)

    # Multi-objective solutions
    # 1. Max the first element
    max_first_index = np.argmax(filtered_couples[:, 0])
    max_first_solution = original_couples[max_first_index]  # Retrieve original values

    # 2. Balance (0.5 weight for both)
    balanced_scores = filtered_couples[:, 0] * 0.5 + (1 - filtered_couples[:, 1]) * 0.5
    balanced_index = np.argmax(balanced_scores)
    balanced_solution = original_couples[balanced_index]  # Retrieve original values


    return max_first_solution, balanced_solution


def obtain_arm_list(max_first_solution, balanced_solution):
    arm_to_is1 = defaultdict(list)
    arm_to_is2 = defaultdict(list)


    for i in range(10):
        file_path = f'EC-NAS-Bench/ecnas/experiments/semoa/7v_SEMOA_Real_{i}.pickle'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Pivot the dataframe to have metrics as columns for each arm
        pivot_df = data.pivot(index='arm_name', columns='metric_name', values='mean')
        
        # Check if the val_acc and energy match the solution values
        mask = (
            (pivot_df['val_acc'] * 0.01 == max_first_solution[0]) &
            (pivot_df['energy'] * 1000 == max_first_solution[1])
        )
        mask1 = (
            (pivot_df['val_acc'] * 0.01 == balanced_solution[0]) &
            (pivot_df['energy'] * 1000 == balanced_solution[1])
        )
        
        # Extract matching arms for each solution
        matching_arms = pivot_df[mask].index.tolist()
        matching_arms1 = pivot_df[mask1].index.tolist()
        
        # Collect all matching arms along with the current i
        for arm in matching_arms:
            arm_to_is1[arm].append(i)

        for arm in matching_arms1:
            arm_to_is2[arm].append(i)
        
    # Deduplicate the indices for each arm and convert to a regular dictionary
    arm_to_is = {arm: list(set(indices)) for arm, indices in arm_to_is1.items()}
    arm_to_is1 = {arm: list(set(indices)) for arm, indices in arm_to_is2.items()}

    # Convert to a list of tuples if needed (arm_name, list_of_indices)
    arm_list_max = list(arm_to_is.items())
    arm_list_balanced = list(arm_to_is1.items())

    return arm_list_max, arm_list_balanced


# Function to read the CSV and extract the adjacency matrix and operations
def parse_csv(arm_name, indices):
        adjacency_matrix = np.zeros((7, 7), dtype=int)  # Initial assumption of 7x7 adjacency matrix
        operations = []
        num_nodes = 0  # Will dynamically determine the number of nodes


        # Open and read the CSV file
        file_path = f'EC-NAS-Bench/ecnas/experiments/semoa/7v_SEMOA_Real_{indices[0]}.csv'
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Go through each row (there may be multiple rows in the CSV)
            for row in reader:
                if row["arm_name"] != arm_name:
                    continue
                budget = row['budget']
                # Dynamically count the number of operation nodes in the current row
                num_nodes = max(num_nodes, len([key for key in row.keys() if key.startswith("op_node_")]))

                # Extract adjacency matrix from edge_*_* columns
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):  # Only upper triangle since it's undirected
                        edge_key = f"edge_{i}_{j}"
                        if row[edge_key] == "1":
                            adjacency_matrix[i, j] = 1
                            adjacency_matrix[j, i] = 1  # Since it's undirected

                # Extract operations from op_node_* columns
                for i in range(num_nodes):  # Dynamically handle the number of nodes
                    op_key = f"op_node_{i}"
                    if op_key in row:
                        operations.append(row[op_key])
                    else:
                        # Handle case where operation might be missing (e.g., some node might not have an operation in the CSV)
                        operations.append("input")  # Default to "input" if missing

                return adjacency_matrix, operations, budget


def extract_archi_and_save_to_csv(max_solution, balanced_solution):

    # generate a number to extract randomly and architecture from the lists
    generate_random_number_1 = random.randint(0, len(arm_list) - 1)
    generate_random_number_2 = random.randint(0, len(arm_list1) - 1)

    # In[25]:

    name, indices = max_solution[generate_random_number_1]
    adjacency_matrix, operations, budget = parse_csv(name, indices)
    # save to a json file both adjacency matrix and operations
    with open('EC-NAS-Bench/Archi/max.json', 'w') as f:
        json.dump({"adjacency_matrix": adjacency_matrix.tolist(), "operations": operations, "budget": budget}, f)


    name, indices = balanced_solution[generate_random_number_2]
    adjacency_matrix, operations, budget = parse_csv(name, indices)
    # save to a json file both adjacency matrix and operations
    with open('EC-NAS-Bench/Archi/balanced.json', 'w') as f:
        json.dump({"adjacency_matrix": adjacency_matrix.tolist(), "operations": operations, 'budget': budget}, f)

    print("Architecture saved to json files")
    return


