import plotly.express as px
from plotly.offline import plot

from src.constants import FINAL_PARAMETERS
from src.feature_engineering import read_features_and_targets
from src.modeling import (
    one_day_ahead_rolling_validation,
    process_validation_predictions,
)

if __name__ == "__main__":
    print(
        f"\n\n{'#'*12}\tQuestion 3. What is the mean absolute error for your model?\t{'#'*12}"
    )

    features_and_targets = read_features_and_targets()

    predictions = one_day_ahead_rolling_validation(
        features_and_targets, **FINAL_PARAMETERS
    )

    metrics_by_meter = process_validation_predictions(predictions)

    metric_bar_plot = px.bar(
        metrics_by_meter[metrics_by_meter["meter"] != "Building"],
        x="meter",
        y="mae",
        template="presentation",
        title="Mean Absolute Error by Tenant Meter",
        labels=dict(meter="Tenant Meter", mae="Mean Absolute Error (kWh)"),
    )

    plot(
        metric_bar_plot,
        show_link=True,
        filename="plots/mean_absolute_error_by_meter_bar_plot.html",
    )
