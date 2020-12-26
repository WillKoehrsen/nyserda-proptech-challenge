import pandas as pd
import tqdm

import plotly.express as px
from plotly.offline import plot
from datetime import timedelta
from datetime import date as create_date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from src.utils import prepare_data

data_dict = prepare_data()

occupancy_data = data_dict["occupancy_data"]

# First date with greater than 50% reduction in occupancy from baseline (first obs)
prediction_start_date = occupancy_data.index.date[
    occupancy_data["baseline_change"] < -50
][0]

print(
    f"The first day with greater than 50% reduction in occupancy is {prediction_start_date}."
)

meter_data = data_dict["meter_data"]
meter_data.head()

tenant_usage_data = data_dict["tenant_data"]
tenant_usage_data.head()

validation_dates = (
    tenant_usage_data.groupby("meter")
    .apply(
        lambda x: x[x["date"] < prediction_start_date]
        .reset_index()["date_time"]
        .astype("int64")
        .quantile(0.75)
    )
    .astype("datetime64[ns]")
    .dt.date
).rename("validation_date")

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


target_cols = ["meter", "consumption", "max_demand", "min_demand", "avg_demand", "name"]

feature_cols = list(
    set(tenant_usage_data.columns)
    - set(target_cols)
    - set(["date", "entries", "baseline_change", "validation_date"])
)

important_features = [
    "date_time_Dayofyear",
    "date_time_FracDay",
    "date_time_FracWeek",
    "date_time_Month",
    "humidity",
    "temp",
]

# Get the features with set and meter columns for subsetting
features = tenant_usage_data[important_features + ["set", "meter"]].copy()

# For the targets, each meter has its own column
targets = tenant_usage_data.pivot_table(
    index=["date_time"], values=["consumption"], columns=["meter"]
)
targets.columns = targets.columns.droplevel(0)

meter_cols = targets.columns
all_feature_imps_list = []
all_results_list = []

for meter in tqdm.tqdm(meter_cols, desc="meters"):
    dataset = (
        targets[[meter]]
        .dropna()
        .merge(
            features[features["meter"] == meter],
            left_index=True,
            right_index=True,
            how="left",
        )
    )

    training_dataset = dataset[dataset["set"] == "training"]

    model = RandomForestRegressor(n_jobs=-1, max_depth=30, n_estimators=60).fit(
        X=training_dataset[important_features], y=training_dataset[meter]
    )

    validation_dataset = dataset[dataset["set"] == "validation"]

    assert (
        training_dataset.index.max() + timedelta(minutes=15)
        == validation_dataset.index.min()
    )

    prediction_dataset = dataset[dataset["set"] == "prediction"]

    assert (
        validation_dataset.index.max() + timedelta(minutes=15)
        == prediction_dataset.index.min()
    ) or (validation_dataset.index.max() < prediction_dataset.index.min())

    validations = model.predict(validation_dataset[important_features])
    predictions = model.predict(prediction_dataset[important_features])

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

    results_df["pct_off"] = (
        100 * (results_df["actual"] - results_df["predicted"]) / results_df["actual"]
    )

    feature_imps = pd.DataFrame.from_dict(
        dict(zip(important_features, model.feature_importances_)),
        orient="index",
        columns=["importance"],
    ).assign(meter=meter)

    all_feature_imps_list.append(feature_imps)
    all_results_list.append(results_df)

    validation_mape = (
        100
        * (
            (validation_df["predicted"] - validation_df["actual"])
            / validation_df["actual"]
        )
        .replace({np.inf: pd.NA})
        .abs()
        .mean()
    )
    prediction_mape = (
        100
        * (
            (prediction_df["predicted"] - prediction_df["actual"])
            / prediction_df["actual"]
        )
        .replace({np.inf: pd.NA})
        .abs()
        .mean()
    )

    print(
        f"Meter: {meter}. Validation MAPE: {round(validation_mape, 2)}%. Prediction MAPE: {round(prediction_mape, 2)}%"
    )

all_feature_imps = (
    pd.concat(all_feature_imps_list).reset_index().rename(columns=dict(index="feature"))
)
all_results = pd.concat(all_results_list)
all_results["date"] = all_results.index.date

all_results["mae"] = all_results["predicted"] - all_results["actual"]

all_results = (
    all_results.reset_index(drop=False)
    .merge(occupancy_data.reset_index(drop=True), how="left", on="date")
    .set_index("date_time")
)


def calculate_corrected_mape(actual, predicted):
    errors = predicted - actual
    ape = errors / actual
    ape[(actual == 0) & (predicted == 0)] = 0
    ape[(actual == 0) & (predicted != 0)] = errors / actual.median()
    return dict(
        mpe=round(100 * ape.median(), 2), mape=round(100 * ape.abs().median(), 2)
    )

