import pandas as pd


def load_and_merge_data():
    # Read the CSV files
    geschaeft_df = pd.read_csv("data/geschaeft.csv")
    dokument_df = pd.read_csv("data/dokument.csv")

    # Merge the dataframes using the correct column names
    merged_df = pd.merge(
        dokument_df,
        geschaeft_df,
        left_on="dokument_geschaeft_uid",
        right_on="geschaeft_uid",
        how="inner",
    )

    return merged_df


def filter_data(df):
    # Apply the required filters
    filtered_df = df[
        (df["geschaeft_typ_deutsch"] == "Motion")
        & (df["dokument_typ_deutsch"].isin(["Vorstosstext", "Vorstoss"]))
    ]

    return filtered_df


def main():
    # Load and merge the data
    merged_df = load_and_merge_data()

    # Apply filters
    filtered_df = filter_data(merged_df)

    # Display basic information about the filtered dataframe
    print("\nFiltered DataFrame Info:")
    print(f"Number of rows: {len(filtered_df)}")
    print("\nColumns in the filtered dataframe:")
    print(filtered_df.columns.tolist())


if __name__ == "__main__":
    main()
