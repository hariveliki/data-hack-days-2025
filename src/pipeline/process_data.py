import pandas as pd


def load_and_merge_data():
    geschaeft_df = pd.read_csv("data/geschaeft.csv")
    dokument_df = pd.read_csv("data/dokument.csv")

    merged_df = pd.merge(
        dokument_df,
        geschaeft_df,
        left_on="dokument_geschaeft_uid",
        right_on="geschaeft_uid",
        how="inner",
    )

    return merged_df


def filter_data(df):
    filtered_df = df[
        (df["geschaeft_typ_deutsch"] == "Motion")
        & (df["dokument_typ_deutsch"].isin(["Vorstosstext", "Vorstoss"]))
    ]

    duplicates = filtered_df.duplicated()
    if duplicates.any():
        print(f"\nFound {duplicates.sum()} duplicate rows")
        filtered_df = filtered_df.drop_duplicates()
        print(f"Removed duplicates. New row count: {len(filtered_df)}")
    else:
        print("\nNo duplicate rows found")

    return filtered_df


def main():
    merged_df = load_and_merge_data()
    filtered_df = filter_data(merged_df)

    print("\nFiltered DataFrame Info:")
    print(f"Number of rows: {len(filtered_df)}")
    print("\nColumns in the filtered dataframe:")
    print(filtered_df.columns.tolist())


if __name__ == "__main__":
    main()
