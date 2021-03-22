import pandas as pd
import plotly.express as px
from plotly.offline import plot
from sklearn.ensemble import RandomForestRegressor

from src.constants import (
    COVID_START_DATE,
    DATE_READ_SETTINGS,
    DATE_TIME_READ_SETTINGS,
    DEFAULT_FEATURES,
    EFFICIENT_DAYS_CSV_NAME,
    FINAL_TEST_PREDICTIONS_CSV_NAME,
    OCCUPANCY_DATA_CSV_NAME,
    RANDOM_FOREST_HYPERPARAMETERS,
    TARGET,
    TEST_DATASET_CSV_NAME,
    TESTING_DATE,
    VALIDATION_DATE_COUNT,
)


def make_test_predictions(
    features_and_targets,
    features=DEFAULT_FEATURES,
    limit_training_to_covid_data=False,
    return_features=False,
    save_predictions=True,
):
    """
    Make test predictions for each tenant meter. Use the provided features and limit to
    covid-only data if specified.
    """
    testing_dataset = pd.read_csv(TEST_DATASET_CSV_NAME, **DATE_TIME_READ_SETTINGS)

    feature_importances = []
    test_predictions = []

    for meter, meter_data in features_and_targets.groupby("meter"):
        training_dataset = meter_data[meter_data["date"] < TESTING_DATE].dropna(
            subset=features + [TARGET]
        )

        if limit_training_to_covid_data:
            training_dataset = training_dataset[
                training_dataset["date"] >= COVID_START_DATE
            ]
            assert training_dataset["date"].min() == COVID_START_DATE

        estimator = RandomForestRegressor(**RANDOM_FOREST_HYPERPARAMETERS).fit(
            training_dataset[features], training_dataset[TARGET]
        )

        test_predictions.append(
            pd.DataFrame(
                dict(
                    forecasted_consumption=estimator.predict(testing_dataset[features]),
                    meter=meter,
                ),
                index=testing_dataset.index,
            )
        )

        feature_importances.append(
            pd.DataFrame(
                dict(feature=features, importance=estimator.feature_importances_)
            )
        )

    test_predictions = pd.concat(test_predictions)

    if save_predictions:
        test_predictions[test_predictions["meter"] != "Building"].to_csv(
            FINAL_TEST_PREDICTIONS_CSV_NAME
        )
        print(f"Test predictions saved to {FINAL_TEST_PREDICTIONS_CSV_NAME}.")

    feature_importances = pd.concat(feature_importances)

    if return_features:
        return test_predictions, feature_importances

    return test_predictions


def occupancy_consumption_correlation(features_and_targets):
    """
    Find the correlations between building-wide occupancy and tenant consumption.
    Calculates the correlation by the percentage relative change and the
    correlation coefficient (Pearsons).
    """
    occupancy = pd.read_csv(OCCUPANCY_DATA_CSV_NAME, **DATE_READ_SETTINGS)

    daily_sums = (
        features_and_targets.groupby(["meter", "date"])["consumption"]
        .sum()
        .reset_index()
        .assign(day_of_week=lambda x: x["date"].dt.day_of_week)
    )

    daily_sums_before_by_day_of_week = (
        daily_sums[daily_sums["date"] < COVID_START_DATE]
        .groupby(["day_of_week", "meter"])["consumption"]
        .mean()
        .reset_index()
    )

    daily_sums_after = (
        daily_sums[daily_sums["date"] >= COVID_START_DATE]
        .copy()
        .assign()
        .assign(day_of_week=lambda x: x["date"].dt.day_of_week)
    )

    merged = daily_sums_after.merge(
        daily_sums_before_by_day_of_week,
        on=["meter", "day_of_week"],
        suffixes=["_covid", "_before"],
    )

    merged["consumption_change_percent"] = (
        100
        * (merged["consumption_covid"] - merged["consumption_before"])
        / merged["consumption_before"]
    )

    merged = merged.merge(
        features_and_targets[["entries", "baseline_change", "date"]], on=["date"]
    )

    merged["consumption_to_occupancy_change"] = (
        merged["consumption_change_percent"] / merged["baseline_change"]
    )

    consumption_to_occupancy_change_by_meter = merged.groupby("meter")[
        "consumption_to_occupancy_change"
    ].mean()

    subset = features_and_targets[
        (features_and_targets["entries"].notna())
        & (features_and_targets["entries"] < 300)
    ]

    daily_corr_by_meter = (
        (
            subset.groupby(["meter", "date"], as_index=False)["consumption"]
            .sum()
            .merge(occupancy, on=["date"])
        )
        .groupby("meter")
        .corr()
        .loc[(slice(None), "entries"), "consumption"]
    )

    print(
        f"\n\n{'#' * 12}\tCorrelations between daily occupancy and consumption\t{'#' * 12}\n\n{daily_corr_by_meter}"
    )

    print(
        f"\n\nOverall Pearson's Correlation between occupancy and consumption: {round(daily_corr_by_meter.median(), 4)}"
    )

    print(
        f"\n\n{'#' * 12}\tPercentage change in consumption to percentage change in occupancy\t{'#' * 12}{consumption_to_occupancy_change_by_meter}"
    )

    overall_change = round(consumption_to_occupancy_change_by_meter.median(), 4)

    print(f"\n\nOverall change in consumption to change in occupancy: {overall_change}")

    print(
        f"\nFor every 10% decrease in occupancy, consumption is expected to decrease by {overall_change * 10}%."
    )
    print(
        f"A 90% decrease in occupancy is expected to result in a {overall_change * 90}% decrease in consumption"
    )


def one_day_ahead_rolling_validation(
    features_and_targets, features=DEFAULT_FEATURES, limit_training_to_covid_data=False
):
    """
    Validate the accuracy of the model by predicting the next-day-ahead
    consumption. Select the most recent dates in the dataset for validation.

    The model is trained on all the data for the meter prior to the date and then makes
    predictions of the consumption on that date, which we can compare to the true value.
    """
    predictions = []

    estimator = RandomForestRegressor(**RANDOM_FOREST_HYPERPARAMETERS)

    print(f"\n\n{'#' * 12}\tRunning One Day Ahead Validation\t{'#' * 12}")

    for meter, meter_features_and_target in features_and_targets.groupby("meter"):

        meter_features_and_target = (
            meter_features_and_target.sort_values("date")
            .assign(
                entries=lambda x: x["entries"]
                .fillna(method="bfill")
                .fillna(method="ffill")
            )
            .dropna(subset=features + [TARGET])
        )

        meter_predictions = []

        validation_dates = (
            pd.Series(meter_features_and_target["date"].unique()).sort_values()
            # Select the most recent dates for validation
            .iloc[-VALIDATION_DATE_COUNT:]
        )

        for validation_date in validation_dates:

            training_dataset = meter_features_and_target[
                meter_features_and_target["date"] < validation_date
            ]

            if limit_training_to_covid_data:
                training_dataset = training_dataset[
                    training_dataset["date"] >= COVID_START_DATE
                ]

            validation_dataset = meter_features_and_target[
                meter_features_and_target["date"] == validation_date
            ]

            estimator.fit(training_dataset[features], training_dataset[TARGET])

            date_predictions = validation_dataset.assign(
                predicted=estimator.predict(validation_dataset[features]),
                training_data_count=len(training_dataset),
            ).rename(columns={TARGET: "actual"})

            meter_predictions.append(
                date_predictions[
                    ["meter", "actual", "predicted", "training_data_count"]
                ]
            )

        predictions.extend(meter_predictions)

        mae, mape = calculate_mae_and_mape(pd.concat(meter_predictions))

        print(
            f"{meter}: Mean Absolute Error: {round(mae, 4)} Median Absolute Percentage Error: {round(mape, 4)}%"
        )

    return pd.concat(predictions)


def process_validation_predictions(predictions):
    """
    Analyze the results from running next-day-ahead validation.
    Calculate the Median Absolute Percentage Error (MAPE) and
    Median Absolute Percentage Error (MAE) of the model.
    """
    predictions = predictions.copy()
    predictions["error"] = predictions["actual"] - predictions["predicted"]

    metrics_by_meter = (
        predictions.groupby("meter")
        .apply(lambda x: calculate_mae_and_mape(x))
        .reset_index()
    )

    metrics_by_meter[["mae", "mape"]] = metrics_by_meter.loc[:, 0].values.tolist()
    metrics_by_meter = metrics_by_meter.drop(columns=0)

    print(
        f"\n\n{'#'*12}\tMean Absolute and Median Percentage Errors by Meter\t{'#'*12}\n\n{metrics_by_meter.round(4)}"
    )
    print(
        f"\n\nOverall Mean Absolute Error: {round(metrics_by_meter['mae'].median(), 4)}"
    )
    print(f"Overall Percentage Error: {round(metrics_by_meter['mape'].median(), 4)}%")

    return metrics_by_meter


def calculate_mae_and_mape(predictions):
    """
    Calculate the mean absolute error (MAE) and the median
    absolute percentage error (MAPE).
    """
    predicted = predictions["predicted"]
    actual = predictions["actual"]

    absolute_errors = (predicted - actual).abs()
    mean_absolute_error = absolute_errors.mean()

    # The percentage error is undefined when the actual measurement is 0.
    percentage_error = absolute_errors / actual
    percentage_error = percentage_error[actual != 0]

    # Take the absolute value after dividing to have all positive metrics
    mean_absolute_percentage_error = 100 * percentage_error.abs().median()

    return mean_absolute_error, mean_absolute_percentage_error


def compute_efficiency_for_occupancy(
    features_and_targets, features=DEFAULT_FEATURES, make_plots=True
):
    """
    Determine the efficiency of different reductions in occupancy. Works by comparing
    the predicted consumption on the date, from a model trained on all pre-covid data,
    to the actual consumption.

    The efficiency is measured as the reduction in consumption from the predicted to the
    actual. The model trains on weather, time of year, day of week, and additional
    variables, so the predictions are normalized across these variables.
    """
    occupancy = pd.read_csv(OCCUPANCY_DATA_CSV_NAME, **DATE_READ_SETTINGS)
    estimator = RandomForestRegressor(**RANDOM_FOREST_HYPERPARAMETERS)
    predictions = []

    for meter, meter_data in features_and_targets.groupby("meter"):
        meter_data = meter_data.dropna(subset=features + [TARGET])

        training_data = meter_data[meter_data["date"] < COVID_START_DATE]
        testing_data = meter_data[meter_data["date"] >= COVID_START_DATE]

        estimator.fit(training_data[features], training_data[TARGET])

        meter_predictions = pd.DataFrame(
            dict(
                predicted=estimator.predict(testing_data[features]),
                actual=testing_data[TARGET],
                meter=meter,
                entries=testing_data["entries"],
                baseline_change=testing_data["baseline_change"],
            ),
            index=testing_data.index,
        )

        predictions.append(meter_predictions)

    predictions = pd.concat(predictions)

    predictions["difference"] = (
        predictions["actual"] - predictions["predicted"]
    ).copy()

    predictions = predictions[predictions["actual"] != 0]

    predictions["percent_difference"] = 100 * (
        predictions["difference"] / predictions["predicted"]
    )

    predictions = predictions[predictions["baseline_change"] < -80]

    predictions["occupancy_change_consumption_change"] = 100 * (
        predictions["percent_difference"] / predictions["baseline_change"]
    )

    if make_plots:
        daily_comparison = (
            predictions.groupby(predictions.index.date)[
                [
                    "baseline_change",
                    "percent_difference",
                    "occupancy_change_consumption_change",
                ]
            ]
            .median()
            .assign(
                # Convert to positive to represent a reduction in occupancy
                occupancy_reduction=lambda x: x["baseline_change"] * -1,
                consumption_reduction=lambda x: x["percent_difference"] * -1,
            )
        ).merge(occupancy, left_index=True, right_index=True)

        print(f"\n\n{'#' * 12}\tMost Efficient Reductions in Occupancy\t{'#' * 12}\n\n")
        print(daily_comparison.nlargest(8, "consumption_reduction"))

        daily_comparison.nlargest(10, "consumption_reduction")[
            "consumption_reduction"
        ].reset_index().rename(columns=dict(index="date")).to_csv(
            EFFICIENT_DAYS_CSV_NAME
        )

        daily_comparison["size"] = daily_comparison["occupancy_reduction"] ** (1 / 2)

        change_scatter_fig = px.scatter(
            daily_comparison.round(4),
            x="occupancy_reduction",
            y="consumption_reduction",
            hover_data=["entries"],
            title="Reduction in Consumption vs Reduction in Occupancy",
            opacity=0.6,
            size="size",
            size_max=40,
            template="presentation",
            labels=dict(
                occupancy_reduction="Reduction in Occupancy",
                consumption_reduction="Reduction in Consumption",
            ),
            trendline="ols",
        )

        plot(
            change_scatter_fig,
            show_link=True,
            filename="plots/consumption_change_vs_occupancy_change_scatter.html",
        )

        efficiency_scatter_fig = px.scatter(
            daily_comparison.round(4).reset_index().rename(columns=dict(index="date")),
            x="date",
            y="consumption_reduction",
            title="Energy Efficiency and Reduction in Occupancy During Covid Timeframe",
            size="size",
            template="presentation",
            size_max=40,
            color="occupancy_reduction",
            color_continuous_scale="Turbo",
            labels=dict(
                date="",
                consumption_reduction="Efficiency",
                occupancy_reduction="Reduction in Occupancy",
            ),
        )

        plot(
            efficiency_scatter_fig,
            show_link=True,
            filename="plots/energy_efficiency_over_time_scatter.html",
        )
    else:
        return predictions


def plot_consumption_before_and_during_covid(features_and_targets):
    """
    Plot the average consumption before covid with the average
    consumption during covid over a week.
    """
    features_and_targets.loc[
        features_and_targets["date"] >= COVID_START_DATE, "set"
    ] = "covid"
    features_and_targets.loc[
        features_and_targets["date"] < COVID_START_DATE, "set"
    ] = "before"

    features_and_targets["week_time"] = features_and_targets.index.strftime(
        "%A %I:%M %p"
    )

    averaged = (
        features_and_targets.groupby(
            ["week_time", "date_time_fracweek", "set", "meter"]
        )["consumption"]
        .mean()
        .reset_index()
        .sort_values(["date_time_fracweek", "meter"])
    )

    pivoted = (
        averaged.pivot_table(
            index=["week_time", "date_time_fracweek", "meter"], columns=["set"]
        )
        .reset_index()
        .sort_values("date_time_fracweek")
    )

    pivoted["change"] = (
        pivoted[("consumption", "covid")] - pivoted[("consumption", "before")]
    ) / pivoted[("consumption", "before")]

    pivoted["day_of_week"] = pivoted["week_time"].str.split(" ").str[0]
    average_change_by_weekday = pivoted.groupby("day_of_week")["change"].median()

    print(
        f"\n\n{'#' * 12}\tAverage consumption change by weekday\t{'#' * 12}\n\n{average_change_by_weekday}\n\n"
    )

    row_count = averaged["meter"].nunique()

    line_fig = (
        px.line(
            averaged,
            x="week_time",
            y="consumption",
            color="set",
            facet_row="meter",
            title="Before Covid vs During Covid Consumption",
            height=row_count * 500,
            template="presentation",
            labels=dict(
                before="Before Covid",
                covid="Covid",
                set="",
                week_time="",
                consumption="Consumption (kWh)",
            ),
        )
        .update_yaxes(matches=None)
        .update_xaxes(showticklabels=True, nticks=21)
    )

    plot(
        line_fig,
        show_link=True,
        filename="plots/consumption_before_covid_during_covid_week_comparison.html",
    )
