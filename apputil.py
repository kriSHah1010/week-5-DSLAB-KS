# apputil.py
# ----------------------------------
# Utility functions for Titanic dataset analysis
# Exercises 1, 2, and Bonus Question
# ----------------------------------

import pandas as pd
import plotly.express as px
from typing import List, Dict, Optional, Any

# FIX: The 'url' parameter is now a required positional argument 
# without a default value or type hint in the signature.

# ==========================================================
# Exercise 1: Survival Patterns
# ==========================================================
def survival_demographics(url):
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
    bins = [0, 12, 19, 59, 200]
    labels = ["Child", "Teen", "Adult", "Senior"]
    # ordered=True ensures the 'AgeGroup' column is a Categorical dtype (Test 1.4)
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True, ordered=True)

    # Group by Pclass, Sex, AgeGroup
    # observed=True is important for categorical grouping to include all possible groups (Test 1.2)
    grouped = df.groupby(["Pclass", "Sex", "AgeGroup"], observed=True).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()

    # Calculate survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Fill NaN values (e.g., if a group has 0 passengers)
    grouped["n_passengers"] = grouped["n_passengers"].fillna(0)
    grouped["n_survivors"] = grouped["n_survivors"].fillna(0)
    grouped["survival_rate"] = grouped["survival_rate"].fillna(0)
    
    return grouped


# ==========================================================
# Exercise 2: Family and Names
# ==========================================================
def family_groups(url):
    """
    Returns a DataFrame showing the last name, family size, and survival rate 
    for each family group (Last Name combined with Family Size).
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.DataFrame: DataFrame with last name, family size, and survival rate.
    """
    df = pd.read_csv(url)

    # 1. Calculate Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 2. Extract Last Name
    df['LastName'] = df['Name'].str.split(',').str[0].str.strip()

    # Filter for families with more than 1 member to analyze groups
    df_families = df[df['FamilySize'] > 1].copy()

    # Group by Last Name
    grouped = df_families.groupby('LastName').agg(
        n_passengers=('PassengerId', 'count'),
        n_survivors=('Survived', 'sum'),
        FamilySize=('FamilySize', 'first') 
    ).reset_index()

    # Calculate survival rate
    grouped['survival_rate'] = grouped['n_survivors'] / grouped['n_passengers']

    # Return required columns: LastName, FamilySize, survival_rate
    return grouped[['LastName', 'FamilySize', 'survival_rate']]


def last_names(url):
    """
    Returns a Series of the 10 most common last names and their counts.
    
    Args:
        url (str): The URL/path to the Titanic CSV dataset.
        
    Returns:
        pd.Series: A Series of the top 10 last names and their counts.
    """
    df = pd.read_csv(url)

    # Extract Last Name (Surname)
    df['LastName'] = df['Name'].str.split(',').str[0].str.strip()

    # Count the occurrences of each last name and get the top 10
    top_names = df['LastName'].value_counts().head(10)

    return top_names


# ==========================================================
# Bonus Question: Older Passenger Division
# ==========================================================
def determine_age_division(url):
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
