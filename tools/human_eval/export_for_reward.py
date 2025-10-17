import sqlite3
import pandas as pd
import os
import json

DB_PATH = "output/evaluations.db"
OUTPUT_CSV_PATH = "output/reward_training_data.csv"

def export_data():
    """
    Connects to the SQLite database, reads the evaluation data,
    parses the JSON ratings, and exports it to a clean CSV file.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM evaluations", conn)
        conn.close()

        # --- Parse JSON and Expand Columns ---
        if 'human_ratings' in df.columns:
            # Apply the json.loads function to each element in the series
            parsed_ratings = df['human_ratings'].apply(json.loads)
            # Convert the list of dicts into a DataFrame
            ratings_df = pd.json_normalize(parsed_ratings)

            # Drop the original JSON column and join the new parsed columns
            df = df.drop('human_ratings', axis=1)
            df = df.join(ratings_df)

            # Ensure ratings are numeric
            for col in ratings_df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
        df.to_csv(OUTPUT_CSV_PATH, index=False)

        print(f"Successfully exported and processed {len(df)} records to {OUTPUT_CSV_PATH}")
        print("Columns created:", df.columns.tolist())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    export_data()