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
        pd.DataFrame: A DataFrame showing survival statistics by Pclass, Sex, and AgeGroup.
    """
    # Load the dataset
    df = pd.read_csv(url)

    # Clean Sex column 
    df['Sex'] = df['Sex'].str.lower().str.strip()

    # Create Age Groups using pd.cut
    # Note: Based on the exercise description, age groups should be:
    # Child (up to 12), Teen (13-19), Adult (20-59), Senior (60+)
    # So bins should be: [0, 12, 19, 59, 200]
    # right=False makes it left-inclusive: (0,12], (12,19], (19,59], (59,200]
    # But for "up to 12" we want to include 12, so let's use right=True with adjusted bins
    bins = [-1, 12, 19, 59, 200]  # Using -1 to include 0-year-olds
    labels = ["Child", "Teen", "Adult", "Senior"]
    
    # Create AgeGroup as categorical with ordered=True
    df["AgeGroup"] = pd.cut(
        df["Age"], 
        bins=bins, 
        labels=labels, 
        right=True, 
        ordered=True
    )
    
    # For groups with NaN Age, we should include them
    # Use observed=True to include all combinations
    grouped = df.groupby(["Pclass", "Sex", "AgeGroup"], observed=True).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()

    # Calculate survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Fill NaN values
    grouped["n_passengers"] = grouped["n_passengers"].fillna(0)
    grouped["n_survivors"] = grouped["n_survivors"].fillna(0)
    grouped["survival_rate"] = grouped["survival_rate"].fillna(0)
    
    # Sort for easy interpretation (as requested)
    grouped = grouped.sort_values(["Pclass", "Sex", "AgeGroup"])
    
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
    
    # Group by family_size and Pclass
    grouped = df.groupby(['family_size', 'Pclass']).agg(
        n_passengers=('PassengerId', 'count'),
        avg_fare=('Fare', 'mean'),
        min_fare=('Fare', 'min'),
        max_fare=('Fare', 'max')
    ).reset_index()

    # Sort for clarity (by class then family size)
    grouped = grouped.sort_values(['Pclass', 'family_size'])
    
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

    # Calculate median age per class
    median_by_class = df.groupby("Pclass")["Age"].transform("median")

    # Boolean column: True if Age > median for class
    df["older_passenger"] = df["Age"] > median_by_class

    return df


def visualize_age_division(df: pd.DataFrame):
    """
    Visualize the effect of being older/younger than the median
    within each passenger class on survival.
    """
    # Group by class and older_passenger
    grouped = df.groupby(["Pclass", "older_passenger"], observed=True).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Plot with Plotly
    fig = px.bar(
        grouped,
        x="Pclass",
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
        x="AgeGroup",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        title="Survival Rate by Age Group, Sex, and Passenger Class",
        labels={"survival_rate": "Survival Rate", "AgeGroup": "Age Group"}
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
        color="Pclass",
        size="n_passengers",
        hover_data=["min_fare", "max_fare"],
        title="Average Fare by Family Size and Passenger Class",
        labels={
            "family_size": "Family Size",
            "avg_fare": "Average Fare",
            "Pclass": "Passenger Class"
        }
    )
    
    return fig
