import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def main(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Drop non-numeric / id columns
    df = df.drop(
        columns=["index", "track_id", "artists", "album_name", "track_name"],
        errors="ignore",
    )

    # Encode categorical
    df = pd.get_dummies(df, columns=["track_genre"], drop_first=True)

    # Boolean to int
    df["explicit"] = df["explicit"].astype(int)

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("Prepared data saved to", output_dir)


if __name__ == "__main__":
    input_csv = sys.argv[1]  # data/raw/spotify.csv
    output_dir = sys.argv[2]  # data/prepared

    main(input_csv, output_dir)
