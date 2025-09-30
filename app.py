# app.py
# Titanic dataset exploration with Exercises 1, 2, and Bonus Question

import streamlit as st
from apputil import (
    survival_demographics,
    visualize_demographic,
    family_groups,
    last_names,
    visualize_families,
    determine_age_division,
    visualize_age_division,
)

# Titanic dataset URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"

# -----------------------
# App Title
# -----------------------
st.title("🚢 Titanic Data Exploration Project")
st.markdown(
    """
    This project explores different aspects of the Titanic dataset using **Python, Pandas, Plotly, and Streamlit**.  
    We focus on three main exercises:
    - **Exercise 1:** Survival patterns by class, sex, and age group.  
    - **Exercise 2:** Family groups, ticket fares, and surnames.  
    - **Bonus Question:** Age division relative to class median and survival outcome.  

    Use the tabs below to navigate through each exercise.
    """
)

# -----------------------
# Tabs Layout
# -----------------------
tab1, tab2, tab3 = st.tabs(["Exercise 1: Survival Patterns", "Exercise 2: Family Groups", "Bonus Question: Age Division"])


# -----------------------
# Tab 1 - Exercise 1
# -----------------------
with tab1:
    st.header("Exercise 1: Survival Patterns by Class, Sex, and Age Group")

    # Run analysis
    results = survival_demographics(url)

    # Show results table
    st.subheader("📊 Survival Demographics Table")
    st.write(
        "This table shows the number of passengers, number of survivors, "
        "and survival rate grouped by **class, sex, and age group**."
    )
    st.dataframe(results)

    # Student research question
    st.subheader("❓ Research Question")
    st.write("Did women in first class have a significantly higher survival rate compared to men across other classes?")

    # Visualization
    st.subheader("📈 Visualization")
    st.write("The chart below compares **survival rates** across age groups, split by class and gender.")
    fig1 = visualize_demographic(results)
    st.plotly_chart(fig1)


# -----------------------
# Tab 2 - Exercise 2
# -----------------------
with tab2:
    st.header("Exercise 2: Family Size, Class, and Ticket Fare")

    # Run family groups analysis
    family_results = family_groups(url)

    # Show results table
    st.subheader("📊 Family Groups Table")
    st.write(
        "This table summarizes passengers by **family size and class**, "
        "showing the average, minimum, and maximum ticket fares."
    )
    st.dataframe(family_results)

    # Last names frequency analysis
    st.subheader("🧑‍👩‍👦 Last Names Frequency")
    st.write(
        "To validate family groups, we also check for repeated last names. "
        "If many people share the same surname, it suggests they traveled together."
    )
    last_name_counts = last_names(url)
    st.write("Top 20 most common last names:")
    st.write(last_name_counts.head(20))

    st.info(
        "👉 From the last names we can see repeated surnames, "
        "which aligns with the family_size analysis in the table above."
    )

    # Student research question
    st.subheader("❓ Research Question")
    st.write("Did larger families in 3rd class pay higher fares on average compared to smaller families?")

    # Visualization
    st.subheader("📈 Visualization")
    st.write(
        "This scatter plot compares **family size and average fare**, "
        "colored by class. The bubble size shows the number of passengers in each group."
    )
    fig2 = visualize_families(family_results)
    st.plotly_chart(fig2)


# -----------------------
# Tab 3 - Bonus Question
# -----------------------
with tab3:
    st.header("Bonus Question: Older vs Younger Passengers (by Class Median Age)")

    # Run age division analysis
    df_age = determine_age_division(url)

    # Show updated table
    st.subheader("📊 Sample of Updated Data")
    st.write(
        "A new Boolean column **`older_passenger`** has been added. "
        "It is `True` if the passenger is older than the median age of their class, "
        "and `False` otherwise."
    )
    st.dataframe(df_age[["Pclass", "Age", "older_passenger"]].head(20))

    # Student research question
    st.subheader("❓ Research Question")
    st.write("Did younger passengers in each class survive at higher rates compared to older passengers?")

    # Visualization
    st.subheader("📈 Visualization")
    st.write(
        "This bar chart compares **survival rates** of older vs younger passengers within each class."
    )
    fig3 = visualize_age_division(df_age)
    st.plotly_chart(fig3)

    st.info(
        "👉 The results give us insight into whether being above or below "
        "the median age for a class affected survival chances."
    )
