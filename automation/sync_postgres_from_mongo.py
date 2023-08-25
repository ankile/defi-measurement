import os

from datetime import datetime
import argparse

from pymongo import MongoClient
import psycopg2
from dotenv import load_dotenv

from tqdm import tqdm

load_dotenv(override=True)
mongo_uri = os.environ["MONGODB_CONNECTION_STRING"]
postgres_uri = os.environ["POSTGRESQL_URI"]

# Connect to MongoDB
mongo_client = MongoClient(mongo_uri, maxPoolSize=50, waitQueueTimeoutMS=100_000)
mongo_db = mongo_client["transactions"]
mongo_collection = mongo_db["mempool-hashes"]

# Connect to PostgreSQL
pg_conn = psycopg2.connect(postgres_uri)
pg_cursor = pg_conn.cursor()


def process_chunk(skip, limit):
    # Initialize a counter for the number of records inserted
    total_inserted = 0

    # Iterate through a chunk of documents in the MongoDB collection)
    progress_bar = tqdm(
        total=limit, desc=f"Processing chunk {skip} - {skip + limit - 1}"
    )

    for document in (
        mongo_collection.find({"syncedToPostgres": {"$ne": True}})
        .skip(skip)
        .limit(limit).max_time_ms(60000)
    ):
        # Insert the hashes into PostgreSQL
        batch_records = [
            (hash["hash"], datetime.utcfromtimestamp(hash["timestamp"] / 1000))
            for hash in document["hashes"]
        ]
        insert_query = """
            INSERT INTO mempool_transactions (hash, first_seen) VALUES (%s, %s)
            ON CONFLICT (hash) DO NOTHING
        """
        pg_cursor.executemany(insert_query, batch_records)
        pg_conn.commit()

        mongo_collection.update_one(
            {"_id": document["_id"]}, {"$set": {"syncedToPostgres": True}}
        )

        total_inserted += pg_cursor.rowcount

        progress_bar.update(1)
        progress_bar.set_postfix(total_inserted=total_inserted)

    return total_inserted


if __name__ == "__main__":
    # Get the count of documents in the MongoDB collection that have not been synced
    total_documents = mongo_collection.count_documents(
        {"syncedToPostgres": {"$ne": True}}, maxTimeMS=8 * 60_000
    )
    print(f"Total documents to be processed: {total_documents}")

    # Use starmap to call process_chunk with different skip and limit values
    total_inserted = process_chunk(0, total_documents)

    # Sum the results to get the total number of records inserted
    print(f"Total records inserted: {total_inserted}")

    # Close connections
    pg_cursor.close()
    pg_conn.close()
    mongo_client.close()
