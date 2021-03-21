import pandas as pd
import plotly.express as px
from plotly.offline import plot

from src.constants import COVID_START_DATE, FINAL_PARAMETERS
from src.feature_engineering import read_features_and_targets
from src.modeling import make_test_predictions

if __name__ == "__main__":

    features_and_targets = read_features_and_targets()

    predictions = make_test_predictions(
        features_and_targets, **FINAL_PARAMETERS
    ).assign(set="Forecast")

    monday_data = (
        features_and_targets[
            (features_and_targets.index.day_of_week == 0)
            & (features_and_targets["date"] > COVID_START_DATE + pd.Timedelta(weeks=16))
        ]
        .groupby(["meter", "date_time_fracday"])["consumption"]
        .mean()
        .reset_index()
        .assign(set="Monday Averaged")
        .rename(columns=dict(consumption="forecasted_consumption"))
    )
    monday_data.index = predictions.index

    combined = pd.concat([predictions, monday_data]).sort_values(["meter"])
    row_count = combined["meter"].nunique()

    comparison_fig = (
        px.line(
            combined.reset_index().sort_values(["date_time", "meter"]),
            x="date_time",
            y="forecasted_consumption",
            color="set",
            height=row_count * 450,
            facet_row="meter",
            template="presentation",
            title="Comparison of Forecasted Consumption to Prior Monday Consumption",
            labels=dict(forecasted_consumption="consumption"),
        )
        .update_yaxes(matches=None)
        .update_xaxes(showticklabels=True)
    )

    plot(comparison_fig, filename="plots/forecasted_consumption_comparison.html")
