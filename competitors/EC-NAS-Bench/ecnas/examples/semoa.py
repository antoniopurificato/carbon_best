
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines import save_experiment
from baselines.methods.semoa import SEMOA

from baselines.problems.ecnas import ecnasSearchSpace
from baselines.problems import get_ecnas
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
    num_nodes = 7
    ops_choices = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]

    # Parameters ecnas
    N_init = 10
    min_budget = 4
    max_budget = 108
    max_function_evals = 100

    trials = 10

    for run in tqdm(range(trials)):
        np.random.seed(run)
        search_space = ecnasSearchSpace(num_nodes, ops_choices)
        experiment = get_ecnas(num_nodes, ops_choices, "SEMOA")

        ea = SEMOA(
            search_space,
            experiment,
            population_size=10,
            num_generations=max_function_evals,
            min_budget=min_budget,
            max_budget=max_budget,
        )
        ea.optimize()

        res = experiment.fetch_data().df
        save_experiment(res, f"/home/hl-neumann/Scrivania/ec-nas/EC-NAS-Bench/ecnas/experiments/semoa/{num_nodes}v_{experiment.name}_Real_{run}.pickle")

        
        arm_data = []
        for arm_name, arm in experiment.arms_by_name.items():
            # Combine arm name and its parameters
            arm_config = {"arm_name": arm_name, **arm.parameters}
            arm_data.append(arm_config)

        # Convert to a pandas DataFrame
        arm_df = pd.DataFrame(arm_data)

        # Save to CSV
        csv_path = f"EC-NAS-Bench/ecnas/experiments/semoa/{num_nodes}v_{experiment.name}_Real_{run}.csv"
        arm_df.to_csv(csv_path, index=False)

