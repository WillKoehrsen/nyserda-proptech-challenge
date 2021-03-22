import pandas as pd
import plotly.express as px
from plotly.offline import plot

from src.constants import EFFICIENT_DAYS_CSV_NAME, FINAL_PARAMETERS
from src.feature_engineering import read_features_and_targets
from src.modeling import (
    one_day_ahead_rolling_validation,
    process_validation_predictions,
)

if __name__ == "__main__":
    print(
        f"{'#'*12}\tQuestion 3. What is the mean absolute error for your model?\t{'#'*12}"
    )

    features_and_targets = read_features_and_targets()

    predictions = one_day_ahead_rolling_validation(
        features_and_targets, **FINAL_PARAMETERS
    )

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
        filename="predictions_and_actual_during_covid_consumption_line_plot.html",
    )

    metrics_by_meter = process_validation_predictions(predictions)

    metric_bar_plot = px.bar(
        metrics_by_meter[metrics_by_meter["meter"] != "Building"],
        x="meter",
        y="mae",
        template="presentation",
        title="Mean Absolute Error by Tenant Meter",
        labels=dict(meter="Tenant Meter", mae="Mean Absolute Error (kWh)"),
    ).update_layout(dict(font=dict(size=26)))

    plot(metric_bar_plot, filename="plots/mean_absolute_error_by_meter_bar_plot.html")
