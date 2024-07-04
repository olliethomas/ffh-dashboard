import re
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from sklearn.metrics import r2_score  # type: ignore


# Function to calculate R-squared values over evaluation weeks
def calculate_r_squared(df, columns, eval_step_col, actual_col):
    eval_steps = df[eval_step_col].unique()
    r_squared_data = []
    for week in eval_steps:
        temp_df = df[(df[eval_step_col] <= week) & (df[actual_col] > 0.01)]
        # temp_df = df[(df[eval_step_col] <= week)]
        agg_funcs = {
            col: "first" if temp_df[col].dtype == "object" else "sum"
            for col in temp_df.columns
            if col != "info_player_id"
        }
        # Group by 'info_player_id' and aggregate
        temp_df = temp_df.groupby("info_player_id").agg(agg_funcs).reset_index()
        for col in columns:
            r2 = (
                r2_score(temp_df[actual_col], temp_df[col])
                if not temp_df.empty
                else None
            )
            r_squared_data.append({"Week": week, "Metric": col, "R-squared": r2})

    return pd.DataFrame(r_squared_data)


# Function to convert and normalize JSON columns
def normalize_column_with_conversion(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    def convert_to_json(x):
        if pd.notnull(x):
            # Remove all " and \n
            x = re.sub(r'[\'"\n]', " ", x)

            # Find all key, value pairs
            x = re.findall(r"([^{,:]+):([^,:}]+)", x)

            # Reconstruct a dictionary with type conversion
            def convert_value(value):
                value = value.strip()
                try:
                    # Try converting to int
                    return int(value)
                except ValueError:
                    try:
                        # Try converting to float
                        return float(value)
                    except ValueError:
                        # Return as string if it cannot be converted
                        return value

            return {key.strip(): convert_value(value) for key, value in x}
        else:
            return {}

    # Convert each element in the column to valid JSON
    col_json = df[col_name].apply(convert_to_json)
    normalized_df = pd.json_normalize(col_json)
    normalized_df.columns = [f"{col_name}_{subcol}" for subcol in normalized_df.columns]

    # Convert columns to appropriate numeric types where possible
    for column in normalized_df.columns:
        try:
            normalized_df[column] = pd.to_numeric(
                normalized_df[column], downcast="float"
            )
        except ValueError as e:
            pass

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
data_path = Path(".") / "data" / "2023_1_12_tkbm4o97.csv"

if "xgb" in data_path.name:
    df = pd.read_csv(data_path)
    # Title of the app
    st.title("Actual vs Pred Scatterplot")

    # Altair scatterplot
    scatter = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(y="actual", x="pred", tooltip=["player", "pred", "actual"])
        .interactive()
    )

    # Display the chart
    st.altair_chart(scatter, use_container_width=True)

else:

    # Streamlit app
    st.title("Scatter Plot of Selected Columns")

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

    # Add a slider for maximum value of info_eval_step
    max_info_eval_week = st.slider(
        "Set maximum value of Evaluation Week",
        min_value=int(df["info_eval_step"].min()) + 1,
        max_value=int(df["info_eval_step"].max()) + 1,
        value=int(df["info_eval_step"].max()) + 1,
    )

    # Filter DataFrame based on selected maximum info_eval_step
    df = df[df["info_eval_step"] < max_info_eval_week]

    # df = df.drop("Unnamed: 0", axis=1)

    # Define aggregation functions
    agg_funcs = {
        col: "first" if df[col].dtype == "object" else "sum"
        for col in df.columns
        if col != "info_player_id"
    }
    # Group by 'info_player_id' and aggregate
    agg_df = df.groupby("info_player_id").agg(agg_funcs).reset_index()
    # df.year = df.year.astype("int")

    # Extract columns with required prefixes
    x_columns = [
        col
        for col in agg_df.columns
        if col.startswith(("ffh", "mean", "predicted", "actual"))
    ]
    y_columns = [
        col
        for col in agg_df.columns
        if col.startswith(("ffh", "mean", "predicted", "actual"))
    ]

    # Add multiselect for positions
    positions = agg_df["info_player_position"].unique().tolist()
    selected_positions = st.multiselect(
        "Select positions to include in the plot", positions, default=positions
    )
    # Check if any positions are selected
    if not selected_positions:
        st.warning("Please select at least one position to generate the plot.")
    else:
        # Filter DataFrame based on selected positions
        agg_df = agg_df[agg_df["info_player_position"].isin(selected_positions)]

        st.write("Select any two columns to create a scatter plot.")
        # Select boxes for choosing columns
        x_axis = st.selectbox("Select X-axis column", x_columns, index=22)
        y_axis = st.selectbox("Select Y-axis column", y_columns, index=25)

        # Generate scatter plot using Altair
        if x_axis and y_axis:
            # Calculate combined domain
            combined_domain = [
                min(agg_df[x_axis].min(), agg_df[y_axis].min()),
                max(agg_df[x_axis].max(), agg_df[y_axis].max()),
            ]

            # Scatter plot
            scatter = (
                alt.Chart(agg_df)
                .mark_circle(size=60)
                .encode(
                    x=alt.X(x_axis, scale=alt.Scale(domain=combined_domain)),
                    y=alt.Y(y_axis, scale=alt.Scale(domain=combined_domain)),
                    tooltip=[
                        "info_player_name",
                        "info_player_position",
                        x_axis,
                        y_axis,
                    ],
                )
                .interactive()
            )

            # Line y=x
            line = (
                alt.Chart(
                    pd.DataFrame({x_axis: combined_domain, y_axis: combined_domain})
                )
                .mark_line(color="red")
                .encode(x=x_axis, y=y_axis)
            )

            # Combine scatter plot and line
            chart = alt.layer(scatter, line).properties(
                title=f"Scatter Plot of {x_axis} vs {y_axis}",
                width=600,
                height=600,  # Ensuring the plot is square
            )

            st.altair_chart(chart, use_container_width=True)

            # Calculate R-squared values
            r_squared_columns = ["predicted_xgnp", "mean_npxg", "ffh_xg"]
            r_squared_df = calculate_r_squared(
                df[df["info_player_position"].isin(selected_positions)],
                r_squared_columns,
                "info_eval_step",
                "actual_npxg",
            )

            # Generate line chart for R-squared values
            line_chart = (
                alt.Chart(r_squared_df)
                .mark_line(point=True)
                .encode(x="Week:O", y="R-squared:Q", color="Metric:N")
                .properties(
                    title="R-squared Values Over Evaluation Weeks",
                    width=600,
                    height=400,
                )
            )

            st.altair_chart(line_chart, use_container_width=True)
