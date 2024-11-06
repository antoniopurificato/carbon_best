class MultiCriteriaOptimizer:
    def __init__(self, 
                beta=0.5, 
                epsilon=1e-8):
        """
        Initializes the MultiCriteriaOptimizer with the specified beta and epsilon values.

        Args:
            beta (float): Weight parameter for accuracy vs. emissions in multi-objective calculations (default=0.5).
            epsilon (float): A small value to avoid division by zero (default=1e-8).

        Attributes:
            disagreement_point_accuracy (float): Stores the disagreement point for accuracy after normalization.
            disagreement_point_emissions (float): Stores the disagreement point for emissions after normalization.
            max_accuracy (float): Stores the maximum normalized accuracy.
            min_emissions (float): Stores the minimum normalized emissions.
        """
        self.beta = beta
        self.epsilon = epsilon
        self.disagreement_point_accuracy = None
        self.disagreement_point_emissions = None
        self.max_accuracy = None
        self.min_emissions = None

    @staticmethod
    def min_max_normalize(value, min_value, max_value):
      """
        Normalizes the value between the min and max range.

        Args:
            value (float): The value to be normalized.
            min_value (float): The minimum value of the range.
            max_value (float): The maximum value of the range.

        Returns:
            float: Normalized value between 0 and 1.
        """
        return 0.0 if max_value - min_value == 0 else (value - min_value) / (max_value - min_value)

    def compute_nash_bargaining(self, row):
        """
        Computes the Nash Bargaining score for a given row.

        Args:
            row (pd.Series): A row from the DataFrame containing normalized accuracy and emissions.

        Returns:
            float: The Nash Bargaining score.
        """
        accuracy, emissions = row['VA_norm'], row['Emis_norm']
        return ((accuracy - self.disagreement_point_accuracy) ** self.beta) * \
               ((self.disagreement_point_emissions - emissions) ** (1 - self.beta))

    def compute_ks_proportion(self, row):
         """
        Computes the K-S Proportion for a given row.

        Args:
            row (pd.Series): A row from the DataFrame containing normalized accuracy and emissions.

        Returns:
            float: The K-S proportion score.

        """
        accuracy, emissions = row['VA_norm'], row['Emis_norm']
        delta_accuracy = self.max_accuracy - self.disagreement_point_accuracy
        delta_emissions = self.disagreement_point_emissions - self.min_emissions

        if delta_accuracy == 0 or delta_emissions == 0:
            raise ValueError("Delta accuracy or delta emissions cannot be zero.")

        norm_accuracy_gain = (accuracy - self.disagreement_point_accuracy) / delta_accuracy
        norm_emissions_gain = (self.disagreement_point_emissions - emissions) / delta_emissions

        return min(norm_accuracy_gain, norm_emissions_gain)

    def compromise_programming(self, accuracy, emissions, ideal_accuracy, ideal_emissions, p=2):
        """
        Computes the Compromise Programming distance for a given solution.

        Args:
            accuracy (float): Normalized accuracy value.
            emissions (float): Normalized emissions value.
            ideal_accuracy (float): Ideal (maximum) accuracy.
            ideal_emissions (float): Ideal (minimum) emissions.
            p (int): The power parameter for compromise programming (default=2).

        Returns:
            float: The Compromise Programming score.
        """
        return ((self.beta * abs(accuracy - ideal_accuracy) ** p) +
                ((1 - self.beta) * abs(emissions - ideal_emissions) ** p)) ** (1 / p)

    def find_pareto_optimal_solutions(self, df):
        """
        Identifies Pareto-optimal solutions from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the normalized accuracy and emissions columns.

        Returns:
            pd.DataFrame: The DataFrame containing only the Pareto-optimal solutions.
        """       
        pareto_optimal_indices = []
        for idx, row in df.iterrows():
            if not any(
                (other_row['VA_norm'] >= row['VA_norm']) and
                (other_row['Emis_norm'] <= row['Emis_norm']) and
                (other_idx != idx) and
                ((other_row['VA_norm'] > row['VA_norm']) or (other_row['Emis_norm'] < row['Emis_norm']))
                for other_idx, other_row in df.iterrows()
            ):
                pareto_optimal_indices.append(idx)

        return df.loc[pareto_optimal_indices]

    def rank_and_sort_solutions_df(self, df_filt, name_acc_var, name_emiss_var):
       """
        Ranks and sorts solutions based on Nash Bargaining, K-S Proportion, and Compromise Programming methods.
        Also applies Borda Count to assign an overall ranking.

        Args:
            df_filt (pd.DataFrame): The filtered DataFrame containing the solutions.
            name_acc_var (str): The column name representing accuracy.
            name_emiss_var (str): The column name representing emissions.

        Returns:
            pd.DataFrame: A sorted DataFrame ranked by multiple criteria and Borda count.
        """
        df_filtered = df_filt.copy()

        # Min/Max values for normalization
        min_accuracy_orig, max_accuracy_orig = df_filtered[name_acc_var].min(), df_filtered[name_acc_var].max()
        min_emissions_orig, max_emissions_orig = df_filtered[name_emiss_var].min(), df_filtered[name_emiss_var].max()

        # Normalize VA (accuracy) and emissions
        df_filtered['VA_norm'] = df_filtered[name_acc_var].apply(
            lambda x: self.min_max_normalize(x, min_accuracy_orig, max_accuracy_orig))
        df_filtered['Emis_norm'] = df_filtered[name_emiss_var].apply(
            lambda x: self.min_max_normalize(x, min_emissions_orig, max_emissions_orig))

        # Set disagreement points and ideal scenarios
        self.disagreement_point_accuracy = df_filtered['VA_norm'].min()
        self.disagreement_point_emissions = df_filtered['Emis_norm'].max()
        self.max_accuracy = df_filtered['VA_norm'].max()
        self.min_emissions = df_filtered['Emis_norm'].min()

        # Apply Nash Bargaining, K-S, and Compromise Programming
        df_filtered['NB Score'] = df_filtered.apply(self.compute_nash_bargaining, axis=1)
        df_filtered['KS Proportion'] = df_filtered.apply(self.compute_ks_proportion, axis=1)
        df_filtered['CP Score'] = df_filtered.apply(lambda row: self.compromise_programming(
            row['VA_norm'], row['Emis_norm'], self.max_accuracy, self.min_emissions), axis=1)

        # Pareto-optimal solutions
        pareto_optimal_indices = self.find_pareto_optimal_solutions(df_filtered).index
        df_filtered['Pareto Optimal'] = df_filtered.index.isin(pareto_optimal_indices)

        # Rankings
        df_filtered['NB rank'] = df_filtered['NB Score'].rank(ascending=False)
        df_filtered['KS rank'] = df_filtered['KS Proportion'].rank(ascending=False)
        df_filtered['CP rank'] = df_filtered['CP Score'].rank(ascending=True)

        # Apply Borda Count
        num_models = len(df_filtered)
        df_filtered["Borda points"] = sum(num_models - df_filtered[f"{method} rank"]
                                          for method in ["NB", "KS", "CP"])

        df_filtered["Borda rank"] = df_filtered["Borda points"].rank(ascending=False)

        # Sort by Borda rank and epoch (lower epochs preferred in case of a tie)
        return df_filtered.sort_values(by=['Borda rank', 'epoch'], ascending=[True, True]).reset_index(drop=True)