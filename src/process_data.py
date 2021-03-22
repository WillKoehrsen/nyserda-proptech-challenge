from src.feature_engineering import (
    create_dataset_from_csvs,
    create_features_and_targets,
)
from src.parsing import read_raw_excel_data_to_csv


def prepare_data():
    """
    Get the data ready for modeling and analysis. The end product is
    a csv of features and targest.
    """
    print("Reading data from Excel to csvs")
    read_raw_excel_data_to_csv()
    print("Creating dataset from csvs")
    create_dataset_from_csvs()
    print("Creating features and targets")
    create_features_and_targets()
    print("Features and targets ready for modeling!")


if __name__ == "__main__":
    prepare_data()
