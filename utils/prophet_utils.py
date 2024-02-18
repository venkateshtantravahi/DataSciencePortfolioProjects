import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns


def clean_dataframe(df, date_column: str, unwanted_columns: List):
    """
    Converts the date column from object type to datetime as well as drops of the unwanted columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - date_column (str): The name of the column containing date information.
    - unwanted_columns (List): List with names of the unwanted columns

    Returns:
    - pd.DataFrame: The dataframe with the updated date column.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df.drop(columns = unwanted_columns, inplace=True)
    return df


def validate_active_cases(df, active_col='Active', confirmed_col='Confirmed', deaths_col='Deaths', recovered_col='Recovered'):
    """
    Validates and adjusts the active cases column based on confirmed, deaths, and recovered columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - active_col (str): The name of the column containing active case numbers.
    - confirmed_col (str): The name of the column containing confirmed case numbers.
    - deaths_col (str): The name of the column containing death numbers.
    - recovered_col (str): The name of the column containing recovered case numbers.

    Returns:
    - pd.DataFrame: The dataframe with validated and adjusted active cases.
    """
    df['Active_check'] = df[confirmed_col] - df[deaths_col] - df[recovered_col]
    df[active_col] = df.apply(lambda row: row['Active_check'] if row[active_col] != row['Active_check'] else row[active_col], axis=1)
    df.drop(columns=['Active_check'], inplace=True)
    return df

def remove_duplicates(df, subset):
    """
    Removes duplicate rows from the dataset based on the given subset of columns.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - subset (list): List of column names to check for duplicate rows.

    Returns:
    - pd.DataFrame: The dataframe with duplicates removed.
    """
    return df.drop_duplicates(subset=subset)


def plot_cumulative_trends(df, date_col, data_cols, title):
    """
    Plots the cumulative trends of confirmed, deaths, and recovered cases.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - date_col (str): The name of the column containing date information.
    - data_cols (list): List of column names to be plotted.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(14, 7))
    for data_col in data_cols:
        sns.lineplot(x=date_col, y=data_col, data=df, label=data_col)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.show()

def plot_daily_changes(df, date_col, data_cols, title):
    """
    Plots the daily changes in confirmed, deaths, and recovered cases.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - date_col (str): The name of the column containing date information.
    - data_cols (list): List of column names to be plotted.
    - title (str): The title of the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(14, 7))
    for data_col in data_cols:
        sns.lineplot(x=date_col, y=df[data_col].diff(), data=df, label=f'Daily {data_col}')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Change')
    plt.legend()
    plt.show()