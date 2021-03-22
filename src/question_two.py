from src.feature_engineering import read_features_and_targets
from src.modeling import (
    occupancy_consumption_correlation,
    plot_consumption_before_and_during_covid,
)

if __name__ == "__main__":

    print(
        f"{'#'*12}\tQuestion 2. How correlated are building-wide occupancy and tenant consumption?\t{'#'*12}"
    )

    features_and_targets = read_features_and_targets()

    occupancy_consumption_correlation(features_and_targets)
    plot_consumption_before_and_during_covid(features_and_targets)
