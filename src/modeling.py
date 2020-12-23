import pandas as pd
from datetime import date as create_date
from sklearn.linear_model import LinearRegression

from src.utils import prepare_data

data_dict = prepare_data()

meter_data = data_dict["meter_data"]
meter_data.head()

tenant_usage_data = data_dict["tenant_data"]
tenant_usage_data.head()

target_cols = ["meter", "consumption", "max_demand", "min_demand", "avg_demand", "name"]
feature_cols = list(set(tenant_usage_data.columns) - set(target_cols))
features = tenant_usage_data[feature_cols].copy()

targets = tenant_usage_data.pivot_table(
    index=["date_time"], values=["consumption"], columns=["meter"]
)
targets.columns = targets.columns.droplevel(0)

meter_cols = targets.columns
all_feature_coefs = []

occupancy_data = data_dict["occupancy_data"]
baseline_entries = occupancy_data["entries"][0]
occupancy_data["baseline_change"] = (
    100 * (occupancy_data["entries"] - baseline_entries) / baseline_entries
)
prediction_start_date = occupancy_data.index.date[
    occupancy_data["baseline_change"] < -50
][0]

for meter in meter_cols:
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
    )
    prediction_df["pct_off"] = (
        prediction_df["actual"] - prediction_df["predicted"]
    ) / prediction_df["predicted"]

    feature_coefs = pd.DataFrame.from_dict(
        dict(zip(feature_cols, model.coef_)), orient="index", columns=["coef"]
    ).assign(meter=meter)

    all_feature_coefs.append(feature_coefs)

all_feature_coefs = pd.concat(all_feature_coefs)
