import pandas as pd
import tqdm

import plotly.express as px
from plotly.offline import plot
from datetime import timedelta
from datetime import date as create_date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.utils import prepare_data

# Percentage of training data used for validation
VALIDATION_PERCENTAGE = 0.3

FEATURES_FILE = "data/modeling/features.csv"
TARGETS_FILE = "data/modeling/targets.csv"


def create_and_save_features_and_targets():
    """
    Create features and targets for machine learning from meter usage data.

    Targets are consumption for each meter.
    """
    data_dict = prepare_data()
    occupancy_data = data_dict["occupancy_data"]

    # First date with greater than 50% reduction in occupancy from baseline (first obs)
    prediction_start_date = occupancy_data.index.date[
        occupancy_data["baseline_change"] < -50
    ][0]

    print(
        f"The first day with greater than 50% reduction in occupancy is {prediction_start_date}."
    )

    tenant_usage_data = data_dict["tenant_data"]
    print("Tenant usage data head:\n\n", tenant_usage_data.head())

    validation_dates = (
        tenant_usage_data.groupby("meter")
        .apply(
            lambda x: x[x["date"] < prediction_start_date]
            .reset_index()["date_time"]
            .astype("int64")
            .quantile(1 - VALIDATION_PERCENTAGE)
        )
        .astype("datetime64[ns]")
        .dt.date
    ).rename("validation_date")

    print("Validation dates:\n\n", validation_dates)

    # Assign training/validation/prediction label to data
    tenant_usage_data.loc[
        tenant_usage_data["date"] >= prediction_start_date, "set"
    ] = "prediction"

    tenant_usage_data = tenant_usage_data.merge(
        validation_dates, left_on="meter", right_index=True, how="left"
    )

    tenant_usage_data.loc[
        (tenant_usage_data["date"] >= tenant_usage_data["validation_date"])
        & (tenant_usage_data["date"] < prediction_start_date),
        "set",
    ] = "validation"

    tenant_usage_data.loc[
        tenant_usage_data["date"] < tenant_usage_data["validation_date"], "set"
    ] = "training"

    print(
        "Set value counts for each meter:\n\n",
        tenant_usage_data.groupby("meter")["set"].value_counts(),
    )

    target_cols = [
        "meter",
        "consumption",
        "max_demand",
        "min_demand",
        "avg_demand",
        "name",
    ]

    # All feature columns
    feature_cols = list(
        set(tenant_usage_data.columns)
        - set(target_cols)
        - set(["set", "date", "entries", "baseline_change", "validation_date"])
    )

    # Get the features with set and meter columns for subsetting
    features = tenant_usage_data[
        feature_cols + ["set", "meter", "entries", "baseline_change"]
    ].copy()

    print("Features head:\n\n", features.head())

    features.reset_index().drop_duplicates(subset=["date_time"]).drop(
        columns=["set", "meter"]
    ).to_csv(FEATURES_FILE, index=False)

    # For the targets, each meter has its own column
    targets = tenant_usage_data.pivot_table(
        index=["date_time", "set"], values=["consumption"], columns=["meter"]
    )
    targets.columns = targets.columns.droplevel(0)

    print("Targets head:\n\n", targets.head())

    targets.reset_index().to_csv("data/modeling/targets.csv", index=False)

    print(f"Features saved to {FEATURES_FILE}")
    print(f"Targets saved to {TARGETS_FILE}")


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


features, targets = read_features_and_targets()


def remove_time_gaps(targets, target_col):
    """

    Remove anomalous consumption measurements resulting from gaps in data.

    Args:
        targets ([type]): [description]
        target_col ([type]): [description]
    """
    series = (
        targets.set_index("date_time")[[target_col, "set"]].dropna(subset=[target_col])
    ).sort_index()

    series["time_diff"] = series.index.to_series().diff()
    series["time_diff_minutes"] = series["time_diff"] / pd.Timedelta(minutes=1)

    print(
        "Time difference value counts:\n\n",
        series["time_diff_minutes"].value_counts().sort_values(),
    )

    anomalous = series[series["time_diff"] != pd.Timedelta(minutes=15)]
    new_series = series.drop(anomalous.index)
    assert (new_series["time_diff"] == pd.Timedelta(minutes=15)).all()
    return new_series[[target_col, "set"]].reset_index()


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
            .merge(features, on="date_time", how="left",)
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

