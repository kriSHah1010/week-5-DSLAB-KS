# apputil.py
# ----------------------------------
# Utility functions for Titanic dataset analysis
# Exercises 1, 2, and Bonus Question
# ----------------------------------

import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Optional, Any, Union

# Default URL for the Titanic dataset
DEFAULT_TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ==========================================================
# Exercise 1: Survival Patterns
# ==========================================================
def survival_demographics(url: str = DEFAULT_TITANIC_URL) -> pd.DataFrame:
    """
    Analyze survival patterns on the Titanic dataset
    grouped by Passenger Class, Sex, and Age Group.
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.DataFrame: A DataFrame showing survival statistics by pclass, sex, and age_group.
    """
    # Load the dataset
    df = pd.read_csv(url)

    # Clean Sex column 
    df['sex'] = df['Sex'].str.lower().str.strip()

    # Create Age Groups using pd.cut - EXACTLY as specified in exercise
    # Child (up to 12), Teen (13-19), Adult (20-59), Senior (60+)
    bins = [0, 12, 19, 59, 200]  # Using these bins to match: (0,12], (12,19], (19,59], (59,200]
    labels = ["Child", "Teen", "Adult", "Senior"]
    
    # Create age_group column as categorical with ordered=True
    df["age_group"] = pd.cut(
        df["Age"], 
        bins=bins, 
        labels=labels, 
        right=True,  # Right inclusive: (0,12], (12,19], (19,59], (59,200]
        ordered=True
    )
    
    # Ensure it's categorical dtype with all categories
    df["age_group"] = pd.Categorical(df["age_group"], categories=labels, ordered=True)
    
    # Use lowercase column names
    df['pclass'] = df['Pclass']
    
    # Create all possible combinations FIRST
    # Get unique values for each grouping column
    all_pclasses = sorted(df['pclass'].unique())
    all_sexes = sorted(df['sex'].unique())
    all_age_groups = labels  # All categories including those that might not exist
    
    # Create a MultiIndex with all possible combinations
    import itertools
    all_combinations = pd.MultiIndex.from_product(
        [all_pclasses, all_sexes, all_age_groups],
        names=['pclass', 'sex', 'age_group']
    )
    
    # Now do the groupby with observed=True
    grouped = df.groupby(["pclass", "sex", "age_group"], observed=True).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()
    
    # Reindex to include all combinations
    grouped.set_index(['pclass', 'sex', 'age_group'], inplace=True)
    grouped = grouped.reindex(all_combinations).reset_index()
    
    # Calculate survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Fill NaN values - IMPORTANT for groups with no members
    grouped["n_passengers"] = grouped["n_passengers"].fillna(0).astype(int)
    grouped["n_survivors"] = grouped["n_survivors"].fillna(0).astype(int)
    grouped["survival_rate"] = grouped["survival_rate"].fillna(0)
    
    # Sort for easy interpretation
    grouped = grouped.sort_values(["pclass", "sex", "age_group"])
    
    # Ensure the required columns are present and in correct order
    required_columns = ["pclass", "sex", "age_group", "n_passengers", "n_survivors", "survival_rate"]
    grouped = grouped[required_columns]
    
    return grouped


# ==========================================================
# Exercise 2: Family Size and Wealth
# ==========================================================
def family_groups(url: str = DEFAULT_TITANIC_URL) -> pd.DataFrame:
    """
    Returns a DataFrame showing family size, passenger class, and fare statistics.
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.DataFrame: DataFrame with family size, class, and fare statistics.
    """
    df = pd.read_csv(url)

    # 1. Calculate Family Size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    
    # Use lowercase pclass
    df['pclass'] = df['Pclass']
    
    # Group by family_size and pclass
    grouped = df.groupby(['family_size', 'pclass']).agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()

    # Sort for clarity
    grouped = grouped.sort_values(['pclass', 'family_size'])
    
    # Ensure correct column names
    required_columns = ['family_size', 'pclass', 'n_passengers', 'avg_fare', 'min_fare', 'max_fare']
    grouped = grouped[required_columns]
    
    return grouped


def last_names(url: str = DEFAULT_TITANIC_URL) -> pd.Series:
    """
    Returns a Series of the last names and their counts.
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.Series: A Series of last names and their counts.
    """
    df = pd.read_csv(url)

    # Extract Last Name (Surname)
    df['LastName'] = df['Name'].str.split(',').str[0].str.strip()

    # Count the occurrences of each last name
    name_counts = df['LastName'].value_counts()

    return name_counts


# ==========================================================
# Bonus Question: Older Passenger Division
# ==========================================================
def determine_age_division(url: str = DEFAULT_TITANIC_URL) -> pd.DataFrame:
    """
    Add a new column 'older_passenger' that marks whether each passenger
    is older than the median age of their passenger class.
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.DataFrame: The original DataFrame with the added 'older_passenger' column.
    """
    df = pd.read_csv(url)

    # Make sure we use lowercase 'age' column
    # First ensure the Age column exists
    if 'Age' in df.columns:
        # Use lowercase column name
        df['age'] = df['Age']
    
    # Use lowercase pclass
    df['pclass'] = df['Pclass']
    
    # Calculate median age per class
    median_by_class = df.groupby("pclass")["age"].transform("median")

    # Boolean column: True if age > median for class
    df["older_passenger"] = df["age"] > median_by_class

    return df


def visualize_age_division(df: pd.DataFrame):
    """
    Visualize the effect of being older/younger than the median
    within each passenger class on survival.
    """
    # Make sure we have the right column names
    if 'pclass' not in df.columns and 'Pclass' in df.columns:
        df['pclass'] = df['Pclass']
    if 'older_passenger' not in df.columns:
        # Calculate it if not present
        median_by_class = df.groupby("pclass")["age"].transform("median")
        df["older_passenger"] = df["age"] > median_by_class
    
    # Group by class and older_passenger
    grouped = df.groupby(["pclass", "older_passenger"], observed=True).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Plot with Plotly
    fig = px.bar(
        grouped,
        x="pclass",
        y="survival_rate",
        color="older_passenger",
        barmode="group",
        text="n_passengers",
        title="Survival Rate by Passenger Class and Age Relative to Class Median"
    )

    return fig


# ==========================================================
# Visualization functions (for app.py)
# ==========================================================
def visualize_demographic():
    """
    Create a Plotly visualization for demographic analysis.
    """
    df = survival_demographics()
    
    # Example visualization: Survival rate by class, sex, and age group
    fig = px.bar(
        df,
        x="age_group",
        y="survival_rate",
        color="sex",
        facet_col="pclass",
        barmode="group",
        title="Survival Rate by Age Group, Sex, and Passenger Class",
        labels={"survival_rate": "Survival Rate", "age_group": "Age Group"}
    )
    
    return fig


def visualize_families():
    """
    Create a Plotly visualization for family analysis.
    """
    df = family_groups()
    
    # Example visualization: Average fare by family size and class
    fig = px.scatter(
        df,
        x="family_size",
        y="avg_fare",
        color="pclass",
        size="n_passengers",
        hover_data=["min_fare", "max_fare"],
        title="Average Fare by Family Size and Passenger Class",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "pclass": "Passenger Class"
        }
    )
    
    return fig
