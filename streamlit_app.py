import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import json


# Function to convert and normalize JSON columns
def normalize_column_with_conversion(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    def convert_to_json(x):
        if pd.notnull(x):
            try:
                x = x.replace("'", '"')
                return json.loads(x)
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    # Convert each element in the column to valid JSON
    col_json = df[col_name].apply(convert_to_json)
    normalized_df = pd.json_normalize(col_json)
    normalized_df.columns = [f"{col_name}_{subcol}" for subcol in normalized_df.columns]
    return normalized_df


# Page title
st.set_page_config(page_title="Interactive Data Explorer", page_icon="📊")
st.title("📊 Interactive Data Explorer")

with st.expander("About this dashboard"):
    st.markdown("**What is this?**")
    st.info(
        "This app allows us to visualise predictions from a number of models. "
        "1) FFH's current predictions which start with `ffh_` "
        "2) Predicting the running mean, which start `mean_` "
        "3) Sussex's current predictions which begin `predictions_` "
        "and 4) The actual data that was observed, which start `actual_`."
        "\n"
        "To get more detail about a point that's plotted, just roll your mouse over it."
    )
    st.markdown("**How to use the dashboard?**")
    st.warning(
        "To engage with the app, select features of your interest in the drop-down selection box for each axis and then as a result, this should generate an updated Scatter plot."
    )

st.subheader("Which predictions are the most accurate?")

# Load data
data_path = Path(".") / "data" / "2022_1_5_1vnozc9l.csv"

df = pd.read_csv(data_path)
# Normalize each JSON column
info_df = normalize_column_with_conversion(df, "info")
ffh_df = normalize_column_with_conversion(df, "ffh")
mean_df = normalize_column_with_conversion(df, "mean")
predicted_df = normalize_column_with_conversion(df, "predicted")
actual_df = normalize_column_with_conversion(df, "actual")

# Drop the original JSON columns
df.drop(["info", "ffh", "mean", "predicted", "actual"], axis=1, inplace=True)

# Concatenate the normalized columns with the main DataFrame
df = pd.concat([df, info_df, ffh_df, mean_df, predicted_df, actual_df], axis=1)
# df = df.drop("Unnamed: 0", axis=1)

# Define aggregation functions
agg_funcs = {
    col: "first" if df[col].dtype == "object" else "sum"
    for col in df.columns
    if col != "info_player_id"
}
# Group by 'info_player_id' and aggregate
df = df.groupby("info_player_id").agg(agg_funcs).reset_index()
# df.year = df.year.astype("int")


# Extract columns with required prefixes
columns = [
    col for col in df.columns if col.startswith(("ffh", "mean", "predicted", "actual"))
]

# Streamlit app
st.title("Scatter Plot of Selected Columns")
st.write("Select any two columns to create a scatter plot.")

# Select boxes for choosing columns
x_axis = st.selectbox("Select X-axis column", columns, index=21)
y_axis = st.selectbox("Select Y-axis column", columns, index=22)

# Generate scatter plot using Altair
if x_axis and y_axis:
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(x=x_axis, y=y_axis, tooltip=["info_player_name", x_axis, y_axis])
        .properties(
            title=f"Scatter Plot of {x_axis} vs {y_axis}", width=600, height=400
        )
    )

    st.altair_chart(chart, use_container_width=True)
