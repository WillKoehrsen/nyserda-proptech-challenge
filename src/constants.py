import pandas as pd

# Base csv file names
ELECTRIC_INTERVAL_DATA_CSV_NAME = "data/csvs/electric.csv"
TENANT_ELECTRIC_INTERVAL_DATA_CSV_NAME = "data/csvs/tenant_usage.csv"
METER_TO_LOCATION_CSV_NAME = "data/csvs/meter_to_location.csv"
OCCUPANCY_DATA_CSV_NAME = "data/csvs/occupancy.csv"
STEAM_DATA_CSV_NAME = "data/csvs/steam.csv"

# Modeling related csv file names
DATASET_CSV_NAME = "data/modeling/dataset.csv"
FEATURES_TARGET_CSV_NAME = "data/modeling/features_targets.csv"
TEST_DATASET_CSV_NAME = "data/modeling/test_dataset.csv"
FINAL_TEST_PREDICTIONS_CSV_NAME = "data/modeling/forecasted_consumption.csv"
EFFICIENT_DAYS_CSV_NAME = "data/modeling/most_efficient_days.csv"

# params for reading in daily data or interval data
DATE_READ_SETTINGS = dict(parse_dates=["date"], index_col="date")
DATE_TIME_READ_SETTINGS = dict(parse_dates=["date_time"], index_col="date_time")

# Number of days for one-day-ahead validation
VALIDATION_DATE_COUNT = 20

# Start of covid, based on when occupancy goes below 500
COVID_START_DATE = pd.to_datetime("2020-03-16")

# Default parameters for building the random forest
RANDOM_FOREST_HYPERPARAMETERS = dict(n_estimators=60, max_depth=15, n_jobs=-1)


DEFAULT_FEATURES = [
    "temp",
    "humidity",
    "date_time_fracday",
    "date_time_fracweek",
    "date_time_fracyear",
    "date_time_trend",
]

TARGET = "consumption"
TESTING_DATE = pd.to_datetime("2020-08-31")

COLUMNS_TO_KEEP = DEFAULT_FEATURES + [
    "consumption",
    "avg_demand",
    "meter",
    "baseline_change",
    "date",
]

FINAL_PARAMETERS = dict(
    features=[
        "temp",
        "humidity",
        "date_time_fracday",
        "date_time_dayofweek",
        "date_time_dayofyear",
        "date_time_trend",
    ],
    limit_training_to_covid_data=True,
)
