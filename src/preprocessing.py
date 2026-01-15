import pandas as pd
import os
from pathlib import Path

def main():

    df = pd.read_csv("data/raw/dataset.csv")
    df_dropped=df.drop(df.columns[0], axis=1)

    print(df_dropped)
    df_dropped.to_csv("data/processed/clean.csv", index=False)
    print("Preprocessing completed")

if __name__ == "__main__":
    main()
