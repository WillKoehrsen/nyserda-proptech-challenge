import pandas as pd
import numpy as np
import tqdm

import plotly.express as px
from plotly.offline import plot
from datetime import timedelta
from datetime import date as create_date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.utils import (
    remove_time_gaps,
    FEATURES_FILE,
    TARGETS_FILE,
)

TARGET_HIGH_LIMIT = 1000

# Important features from random forest model
important_features = [
    "date_time_FracYear",
    "date_time_FracMonth",
    "humidity",
    "date_time_FracDay",
    "date_time_Trend",
    "temp",
    "date_time_FracWeek",
]


def read_features_and_targets():
    """
    Read in processed features and targets from files.

    Returns:
        features, targets: Pandas dataframes
    """
    return (
        pd.read_csv(FEATURES_FILE, parse_dates=["date_time"]),
        pd.read_csv(TARGETS_FILE, parse_dates=["date_time"]),
    )


def plot_interpolated_series(comparison, meter_name):
    fig = px.line(
        comparison,
        x="date_time",
        y="consumption",
        color="set",
        template="presentation",
        title=f"{meter_name}",
    )
    # Blue for interpolated
    fig.data[0].mode = "lines+markers"
    # Orange and transparent for original
    fig.data[1].opacity = 0.1
    plot(
        fig,
        auto_open=False,
        show_link=True,
        include_plotlyjs=False,
        filename=f"./plots/{meter_name}_interpolation.html",
    )


def interpolate_zero_values(original_measurement_series):
    measurement_series = original_measurement_series.copy().dropna()

    zero_before = measurement_series.shift(1) == 0

    measurement_series[zero_before] = np.NaN
    measurement_series = measurement_series.replace({0: np.NaN})

    measurement_series = measurement_series.interpolate(method="time")

    comparison = (
        pd.concat(
            [
                measurement_series.rename("interpolated"),
                original_measurement_series.rename("original"),
            ],
            axis="columns",
        )
        .reset_index()
        .melt(id_vars=["date_time"], var_name="set", value_name="consumption")
        .dropna()
    )
    meter_name = original_measurement_series.name
    plot_interpolated_series(comparison, meter_name)

    return measurement_series


def create_targets(targets_from_file):
    targets = (
        pd.concat(
            [
                remove_time_gaps(targets_from_file, target_col)
                for target_col in targets_from_file.drop(
                    columns=["date_time", "set"]
                ).columns
            ]
        )
        .pivot_table(index=["date_time", "set"])
        .reset_index()
    )
    meter_cols = targets.select_dtypes("number").columns
    meters = targets[meter_cols].copy()
    melted = targets.melt(
        id_vars=["date_time", "set"], value_name="consumption", var_name="meter"
    ).dropna()

    interpolated_values = pd.concat(
        [
            interpolate_zero_values(targets.set_index("date_time")[meter])
            for meter in meter_cols
        ],
        axis=1,
    )
    return interpolated_values


def prepare():
    """
    Get data ready for modeling

    Returns:
        [type]: [description]
    """
    features, targets_from_file = read_features_and_targets()
    targets = create_targets(targets_from_file)
    targets = (
        pd.concat(
            [
                remove_time_gaps(targets, target_col)
                for target_col in targets.drop(columns=["date_time", "set"]).columns
            ]
        )
        .pivot_table(index=["date_time", "set"])
        .reset_index()
    )
    fig = px.line(
        targets, x="date_time", y=targets.drop(columns=["set", "date_time"]).columns
    )
    plot(fig, filename="targets.html")
    return features, targets


def model_with_features(features, targets, feature_list):
    """
    Build a supervised regression model with the specified features.
    Returns the predictions for each meter and set.
    """
    # Each meter forms one column of the targets (consumption values)
    meter_cols = [
        column for column in targets.columns if "-0" in column or column == "building"
    ]

    all_feature_imps_list = []
    all_results_list = []

    for meter in tqdm.tqdm(meter_cols, desc="meters"):
        meter_targets = remove_time_gaps(targets, target_col=meter)
        dataset = (
            meter_targets
            # Merge the targets with their respective features
            .merge(
                features,
                on="date_time",
                how="left",
            )
        ).set_index("date_time")

        training_dataset = dataset[dataset["set"] == "training"]

        # Train the model on the features and training target
        model = RandomForestRegressor(n_jobs=-1, max_depth=50, n_estimators=100).fit(
            X=training_dataset[feature_list], y=training_dataset[meter]
        )

        validation_dataset = dataset[dataset["set"] == "validation"]

        assert (
            training_dataset.index.max() + timedelta(minutes=15)
            <= validation_dataset.index.min()
        )

        prediction_dataset = dataset[dataset["set"] == "prediction"]

        assert (
            validation_dataset.index.max() + timedelta(minutes=15)
            <= prediction_dataset.index.min()
        )

        validations = model.predict(validation_dataset[feature_list])
        predictions = model.predict(prediction_dataset[feature_list])

        validation_df = (
            pd.DataFrame(
                dict(predicted=validations, actual=validation_dataset[meter]),
                index=validation_dataset.index,
            )
            .assign(meter=meter)
            .assign(set="validation")
        )

        prediction_df = (
            pd.DataFrame(
                dict(predicted=predictions, actual=prediction_dataset[meter]),
                index=prediction_dataset.index,
            )
            .assign(meter=meter)
            .assign(set="prediction")
        )

        results_df = pd.concat([validation_df, prediction_df])

        feature_imps = pd.DataFrame.from_dict(
            dict(zip(feature_list, model.feature_importances_)),
            orient="index",
            columns=["importance"],
        ).assign(meter=meter)

        all_feature_imps_list.append(feature_imps)
        all_results_list.append(results_df)

    all_feature_imps = (
        pd.concat(all_feature_imps_list)
        .reset_index()
        .rename(columns=dict(index="feature"))
    )

    all_results = pd.concat(all_results_list)
    all_results["date"] = all_results.index.date

    return dict(all_results=all_results, all_feature_imps=all_feature_imps)


def run_modeling():
    new_results_dict = model_with_features(
        features=features,
        targets=targets,
        feature_list=[
            "date_time_FracYear",
            "date_time_FracMonth",
            "humidity",
            "date_time_FracDay",
            "temp",
            "date_time_FracWeek",
        ],
    )

    new_all_results, new_all_feature_imps = (
        new_results_dict["all_results"],
        new_results_dict["all_feature_imps"],
    )

    new_scores_by_meter = new_all_results.groupby(["meter", "set"]).apply(
        lambda x: calculate_corrected_mape(actual=x["actual"], predicted=x["predicted"])
    )


# all_results = (
#     all_results.reset_index(drop=False)
#     .merge(occupancy_data.reset_index(drop=True), how="left", on="date")
#     .set_index("date_time")
# )


def calculate_corrected_mape(actual, predicted):
    errors = predicted - actual

    # The percentage error undefined when the actual measurement is 0.
    pe = errors / actual

    # When both the actual and predicted are 0, clearly the error is 0.
    pe[(actual == 0) & (predicted == 0)] = 0

    # Where the actual value is 0, but the predicted is not 0, divide by the absolute
    # mean actual value as an approximation
    pe[(actual == 0) & (predicted != 0)] = errors / actual.abs().mean()

    # The mean percentage error encodes information about directionality of errors
    mpe = 100 * pe.mean()

    # The mean absolute percentage error does not count directionality and thus may be
    # better for comparing model performance.
    mape = 100 * pe.abs().mean()

    return dict(mpe=round(mpe, 2), mape=round(mape, 2))
