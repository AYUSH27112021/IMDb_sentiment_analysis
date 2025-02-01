import pandas as pd
import sqlite3
import os

# Database file
DB_FILE = r"Database_File\imdb_reviews.db"
CSV_FILE = r"IMDB Dataset.csv"
BATCH_SIZE = 1000

# Create the imdb_reviews table in the SQLite database.
def create_table(DB_FILE):
    directory = os.path.dirname(DB_FILE)     
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()
    
    # Create table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review TEXT NOT NULL,
        sentiment TEXT NOT NULL
    )
    """)
    
    connection.commit()
    connection.close()
    print("Table created successfully.")

# Insert data from CSV into the imdb_reviews table.
def insert_data_from_csv(IMDB_df):
    connection = sqlite3.connect(DB_FILE)
    
    # Read the CSV in chunks
    for i in range(0, len(IMDB_df), BATCH_SIZE):
        chunk = IMDB_df.iloc[i:i + BATCH_SIZE]
        # Insert chunk into the database
        chunk.to_sql('imdb_reviews', connection, if_exists='append', index=False)
    
    connection.commit()
    connection.close()
    print("Data inserted successfully.")

if __name__ == "__main__":
    IMDB_df = pd.read_csv(filepath_or_buffer="IMDB Dataset.csv",delimiter=',',header=0,encoding="utf-8")
    if IMDB_df.isnull().all(axis=1).any():
        raise Exception("Empty rows found in the CSV file.")
    else:
        create_table(DB_FILE)
        insert_data_from_csv(IMDB_df)
        print("Data Setup")
