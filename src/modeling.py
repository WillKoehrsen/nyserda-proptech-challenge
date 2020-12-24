import pandas as pd
import tqdm

import plotly.express as px
from plotly.offline import plot
from datetime import date as create_date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

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

target_cols = ["meter", "consumption", "max_demand", "min_demand", "avg_demand", "name"]
feature_cols = list(
    set(tenant_usage_data.columns)
    - set(target_cols)
    - set(["date", "entries", "baseline_change"])
)
features = tenant_usage_data[feature_cols].copy()

targets = tenant_usage_data.pivot_table(
    index=["date_time"], values=["consumption"], columns=["meter"]
)
targets.columns = targets.columns.droplevel(0)

meter_cols = targets.columns
all_feature_coefs_list = []
all_predictions_list = []

for meter in tqdm.tqdm(meter_cols, desc="meters"):
    dataset = (
        targets[[meter]]
        .dropna()
        .merge(features, left_index=True, right_index=True, how="left")
    )

    training_dataset = dataset[dataset.index.date < prediction_start_date]
    model = LinearRegression().fit(
        X=training_dataset[feature_cols], y=training_dataset[meter]
    )

    prediction_dataset = dataset[dataset.index.date >= prediction_start_date]
    predictions = model.predict(prediction_dataset[feature_cols])
    prediction_df = pd.DataFrame(
        dict(predicted=predictions, actual=prediction_dataset[meter]),
        index=prediction_dataset.index,
    ).assign(meter=meter)
    prediction_df["pct_off"] = (
        100
        * (prediction_df["actual"] - prediction_df["predicted"])
        / prediction_df["predicted"]
    )

    feature_coefs = pd.DataFrame.from_dict(
        dict(zip(feature_cols, model.coef_)), orient="index", columns=["coef"]
    ).assign(meter=meter)

    all_feature_coefs_list.append(feature_coefs)
    all_predictions_list.append(prediction_df)

all_feature_coefs = pd.concat(all_feature_coefs_list)
all_predictions = pd.concat(all_predictions_list)
all_predictions["date"] = all_predictions.index.date

all_predictions = (
    all_predictions.reset_index(drop=False)
    .merge(occupancy_data.reset_index(drop=True), on="date")
    .set_index("date_time")
)


fig_actual = px.line(
    all_predictions.reset_index(),
    x="date_time",
    y="actual",
    color="meter",
    title="Actual Consumption by Meter",
)
plot(fig_actual, show_link=True)

fig_predicted = px.line(
    all_predictions.reset_index(),
    x="date_time",
    y="predicted",
    color="meter",
    title="Predicted by meter",
    show_link=True,
)
plot(fig_predicted, show_link=True)
