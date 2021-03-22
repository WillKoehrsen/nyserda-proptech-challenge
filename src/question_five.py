from src.feature_engineering import read_features_and_targets
from src.modeling import compute_efficiency_for_occupancy

if __name__ == "__main__":
    print(
        f"{'#'*12}\tQuestion 5. What is the most energy-efficient occupancy level as a percentage of max occupancy provided (i.e., occupancy on 2/10/20)?\t{'#'*12}"
    )

    features_and_targets = read_features_and_targets()

    compute_efficiency_for_occupancy(features_and_targets)
