import pandas as pd


def print_transformed_data_structure(transformed_data):
    for year, year_data in transformed_data.items():
        print(f"\n=== Year: {year} ===")
        for data_type, data in year_data.items():
            print(f"\n-- Data Type: {data_type} --")
            for sub_type, df in data.items():
                if isinstance(df, pd.DataFrame):
                    print(f"\nDataFrame: {sub_type}")
                    print(f"Shape: {df.shape}")
                    print("Columns:")
                    for col in df.columns:
                        print(f"  - {col}")
                    print("\nSample data:")
                    print(df.head(2))
