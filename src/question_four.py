import pandas as pd
import plotly.express as px
from plotly.offline import plot

from src.constants import EFFICIENT_DAYS_CSV_NAME, FINAL_PARAMETERS
from src.feature_engineering import read_features_and_targets
from src.modeling import make_test_predictions

if __name__ == "__main__":
    print(
        f"{'#'*12}\tQuestion 4. What feature(s)/predictor(s) were most important in determining energy efficiency?\t{'#'*12}"
    )

    features_and_targets = read_features_and_targets()

    features = FINAL_PARAMETERS["features"][:]
    features.append("entries")

    predictions, feature_importances = make_test_predictions(
        features_and_targets,
        return_features=True,
        save_predictions=False,
        features=features,
        limit_training_to_covid_data=True,
    )

    feature_stats = (
        feature_importances.groupby("feature")["importance"].mean().reset_index()
    )

    features_to_readable = dict(
        date_time_dayofweek="Day of Week",
        date_time_dayofyear="Day of Year",
        date_time_fracday="Time of Day",
        date_time_trend="Trend",
        humidity="Humidity",
        temp="Temperature",
        entries="Occupancy",
    )
    feature_stats["feature"] = feature_stats["feature"].map(features_to_readable)

    feature_plot = px.bar(
        feature_stats,
        x="feature",
        y="importance",
        labels=dict(feature=""),
        template="presentation",
        title="Importance of Variables",
    ).update_xaxes(categoryorder="total descending")

    plot(feature_plot, filename="plots/feature_values.html")

    efficient_days = list(
        pd.read_csv(EFFICIENT_DAYS_CSV_NAME, parse_dates=["date"])["date"].dt.date
    )
    row_count = predictions["meter"].nunique()

    predictions["date"] = predictions.index.date

    prediction_fig = (
        px.line(
            predictions[predictions["date"].isin(efficient_days)].reset_index(),
            x="date_time",
            y=["predicted", "actual"],
            facet_row="meter",
            height=row_count * 800,
            title="Predicted vs Actual Covid Consumption",
            template="presentation",
        )
        .update_yaxes(matches=None)
        .update_xaxes(showticklabels=True)
    )

    plot(
        prediction_fig,
        filename="plots/efficient_days_predictions_and_actual_during_covid_consumption_line_plot.html",
    )
