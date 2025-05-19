import pandas as pd


def merge_data():
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


def filter_and_save_data():
    df = merge_data()
    filtered_df = df[
        (df["geschaeft_typ_deutsch"] == "Motion")
        & (df["dokument_typ_deutsch"].isin(["Vorstosstext", "Vorstoss"]))
    ]

    duplicates = filtered_df.duplicated()
    if duplicates.any():
        filtered_df = filtered_df.drop_duplicates()

    filtered_df.to_csv("data/filtered_motions.csv", index=False)
    return None


def main():
    filter_and_save_data()


if __name__ == "__main__":
    main()
