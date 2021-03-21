import plotly.express as px
from plotly.offline import plot

from src.constants import FINAL_PARAMETERS
from src.feature_engineering import read_features_and_targets
from src.modeling import make_test_predictions

if __name__ == "__main__":

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
