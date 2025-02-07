from typing import Dict, List, Union
import pandas as pd
import numpy as np

def get_unique_columns(df, prefix_sep="__"):
    """Get the unique columns of the dataframe by disragarding string
    that comes after postfix "__". SO IMPORTANT: Only one-hot encoded
    columns should have "__" as postfix.

    example:
    df columns: ['age', 'gender_1', 'gender_2', 'salary']
    output: ['age', 'gender', 'salary']

    Args:
        df (_type_): Dataframe conatining one-hot encoded columns.

    Returns:
        list: unique column names without postfixes.
    """
    columns_seen = []

    for column in df.columns:
        if len(columns_seen) == 0:
            columns_seen.append(column)
        if not columns_seen[-1].startswith(column.split(prefix_sep)[0]):
            columns_seen.append(column.split(prefix_sep)[0])

    return columns_seen

class DataSplitter:
    def __init__(
        self, 
        array_to_split: Union[pd.DataFrame, np.ndarray], 
        columns: List[str] = None,
        prefix_sep: str = "__"
    ):
        self.data = array_to_split
        self.columns = columns
        self.prefix_sep = prefix_sep

        if not isinstance(array_to_split, pd.DataFrame):
            # Create a DataFrame
            temp_df = pd.DataFrame(data=array_to_split, columns=columns)
            self.data = temp_df

        self.cat_one_hot_cols = set()

        # Get unique feature index range to include one-hot encoded columns in each client
        self.columns_unique = get_unique_columns(self.data, self.prefix_sep)
        print(f"[DataSplitter] {self.columns_unique=}")

        # create mapping of original columns that are one-hot encoded
        # in the dataframe
        for original_col in self.columns_unique:
            for col in self.data.columns:
                if col.startswith(original_col+self.prefix_sep):
                    self.cat_one_hot_cols.add(original_col)
                    break
        print(f"{self.cat_one_hot_cols=}")

    def divide_number(self, num, clients):
        res = np.zeros(clients, dtype=int)
        quotient = num // clients
        remainder = num % clients

        res += quotient
        res[:remainder] += 1

        return res.tolist()
        
    def split_data(self, list_of_cols_per_client: Union[List[List[str]], List[float]]) -> Union[List[pd.DataFrame], List[np.ndarray]]:
        """
        Split the data into multiple subsets based on the provided percentages.
        
        :param list_of_cols_per_client: A list of column lists representing the columns to keep from data for each client.
        :return: A list of Pandas dataframes representing the data in each subset for each client.
        """
        if isinstance(list_of_cols_per_client, list) and isinstance(list_of_cols_per_client[0], float):            
            num_unique_cols = len(self.columns_unique)
            split_ratios = self.divide_number(num_unique_cols, len(list_of_cols_per_client))
            print(split_ratios)

            split_cols = []

            prev_col = -1

            for num_cols in split_ratios:
                if prev_col == -1:
                    prev_col = 0

                client_cols = self.columns_unique[prev_col : prev_col+num_cols]
                print(len(client_cols))
                print(f"[DataSplitter] {client_cols=}")
                prev_col += num_cols

                # Filter the true column names from self.data
                true_cols = []
                for true_col in client_cols:
                    for col in self.columns:
                        # make sure that columns that are one-hot encoded
                        # are added to list corresponding to their original column
                        # this is done to prevent,e.g., 'education-num' to be added
                        # to the list of columns corresponding to 'education' so that only
                        # exact one-hot encoded columns of 'education' are added.
                        if true_col in self.cat_one_hot_cols:
                            if col.startswith(true_col+self.prefix_sep):
                                true_cols.append(col)    
                        else:
                            # if not one-hot encoded then should be the same as the original column
                            if col == true_col:
                                true_cols.append(col)
                print(f"[DataSplitter] {true_cols=}")

                split_cols.append(true_cols)
            
            print(split_cols)

            # verify columns in one-hot encoded df are in the split_cols list
            flattened_split_cols = [item for sublist in split_cols for item in sublist]
            for original_col in self.data.columns:
                count_col = flattened_split_cols.count(original_col)
                count_col_data = self.columns.count(original_col)
                assert count_col == count_col_data, f"Number of columns in each split should be equal to the number of columns in the original data. {count_col=} {count_col_data=}, {original_col=}"

            split_data = []
            for client_idx in range(len(list_of_cols_per_client)):
                split_data.append(self.data[split_cols[client_idx]])

            return split_data
        else:
            assert set(self.columns) == set([col for cols in list_of_cols_per_client for col in cols]), "The columns in the list of columns per client should be a subset of the columns in the dataset."

            split_data = [self.data.loc[:, cols_to_keep] for cols_to_keep in list_of_cols_per_client]
            return split_data

