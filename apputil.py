# apputil.py
# -----------------------------------
# Utility functions for Titanic dataset analysis
# Exercises 1, 2, and Bonus Question
# -----------------------------------

import pandas as pd
import plotly.express as px


# ==========================================================
# Exercise 1: Survival Patterns
# ==========================================================
def survival_demographics(url: str):
    """
    Analyze survival patterns on the Titanic dataset
    grouped by Passenger Class, Sex, and Age Group.
    """
    # Load the dataset
    df = pd.read_csv(url)

    # Create Age Groups using pd.cut
    bins = [0, 12, 19, 59, 200]  # boundaries for child, teen, adult, senior
    labels = ["Child", "Teen", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

    # Group by Pclass, Sex, AgeGroup
    grouped = df.groupby(["Pclass", "Sex", "AgeGroup"]).agg(
        n_passengers=("PassengerId", "count"),
        n_survivors=("Survived", "sum")
    ).reset_index()

    # Calculate survival rate
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    # Sort results for readability
    grouped = grouped.sort_values(by=["Pclass", "Sex", "AgeGroup"])

    return grouped


def visualize_demographic(results: pd.DataFrame):
    """
    Create a visualization for demographic survival analysis.
    Example: Survival Rate by Class, Sex, and Age Group.
    """
    fig = px.bar(
        results,
        x="AgeGroup",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        text="n_passengers",
        title="Titanic Survival Rate by Class, Sex, and Age Group"
    )
    fig.update_layout(yaxis_title="Survival Rate", xaxis_title="Age Group")
    return fig


# ==========================================================
# Exercise 2: Family Groups and Last Names
# ==========================================================
def family_groups(url: str):
    """
    Explore the relationship between family size, passenger class, and ticket fare.
    """
    df = pd.read_csv(url)

    # Family size = SibSp + Parch + passenger themselves
    df["family_size"] = df["SibSp"] + df["Parch"] + 1

    # Group by family_size and class
    grouped = df.groupby(["family_size", "Pclass"]).agg(
        n_passengers=("PassengerId", "count"),
        avg_fare=("Fare", "mean"),
        min_fare=("Fare", "min"),
        max_fare=("Fare", "max"),
    ).reset_index()

    # Sort by class then family size
    grouped = grouped.sort_values(by=["Pclass", "family_size"])
    return grouped


def last_names(url: str):
    """
    Extract last names from the Name column and return their frequency.
    """
    df = pd.read_csv(url)

    # Last name = part before the first comma
    df["LastName"] = df["Name"].apply(lambda x: x.split(",")[0].strip())

    # Count occurrences
    last_name_counts = df["LastName"].value_counts()
    return last_name_counts


def visualize_families(results: pd.DataFrame):
    """
    Create a visualization for family size and fare patterns.
    """
    fig = px.scatter(
        results,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        size="n_passengers",
        hover_data=["min_fare", "max_fare"],
        title="Average Fare vs. Family Size by Passenger Class"
    )
    fig.update_layout(
        xaxis_title="Family Size",
        yaxis_title="Average Ticket Fare"
    )
    return fig


# ==========================================================
# Bonus Question: Older Passenger Division
# ==========================================================
def determine_age_division(url: str):
    """
    Add a new column 'older_passenger' that marks whether each passenger
    is older than the median age of their passenger class.
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
    grouped = df.groupby(["Pclass", "older_passenger"]).agg(
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
        title="Survival Rate: Older vs. Younger Passengers Within Class"
    )
    fig.update_layout(
        xaxis_title="Passenger Class",
        yaxis_title="Survival Rate"
    )
    return fig
