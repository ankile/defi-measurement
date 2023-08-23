import os
import sys

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")


import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm
from collections import deque

from pool_state import v3Pool


import pickle
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError


from datetime import datetime


load_dotenv(override=True)

from decimal import getcontext


def load_pool_from_blob(
    pool_address: str,
    postgres_uri: str,
    azure_connection_string: str,
    container_name: str,
    verbose: bool = True,
    invalidate_before_date: datetime | None = None,  # Add the optional invalidate date
    pbar: tqdm | None = None,
) -> v3Pool:
    
    # Initialize the BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)
    
    # Get the container client
    container_client = blob_service_client.get_container_client(container_name)
    
    # Define the blob name
    blob_name = f"{pool_address}.pickle"
    
    # Check if the blob exists in Azure Blob Storage
    blob_exists = False
    blob_client = container_client.get_blob_client(blob=blob_name)
    try:
        blob_exists = blob_client.exists()
    except ResourceNotFoundError:
        # Blob does not exist
        pass
    except Exception as e:
        # Handle other exceptions, perhaps logging them for debugging
        print(f"An error occurred: {e}")

    if blob_exists:
        # Check if blob needs to be invalidated
        blob_properties = blob_client.get_blob_properties()
        last_modified = blob_properties['last_modified']
        
        if invalidate_before_date and last_modified < invalidate_before_date:
            blob_exists = False
            if verbose:
                print("Blob is invalidated due to old modification date")
            if pbar:
                pbar.set_postfix_str("Blob is invalidated due to old modification date")

    if blob_exists:
        if verbose:
            print("Loading pool from Azure blob storage cache")
        if pbar:
            pbar.set_postfix_str("Loading pool from Azure blob storage cache")

        blob_data = blob_client.download_blob()
        pool = pickle.loads(blob_data.readall())
        return pool

    # If it's not in the cache, load it and add it to the cache
    if verbose:
        print("Loading pool from database")
    if pbar:
        pbar.set_postfix_str("Loading pool from database")

    pool = v3Pool(
        poolAdd=pool_address,
        connStr=postgres_uri,
        initialize=False,
        delete_conn=True,
        verbose=verbose,
    )

    # Save the cache to Azure Blob Storage
    blob_client.upload_blob(pickle.dumps(pool), overwrite=True)

    return pool


if __name__ == "__main__":
    print("Starting...")
    getcontext().prec = 100  # Set the precision high enough for our purposes


    # Read in the environment variables
    postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
    postgres_uri_us = os.environ["POSTGRESQL_URI_US"]
    azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]


    engine = create_engine(postgres_uri_mp)

    print("Getting the data...")

    query = """
    SELECT *
    FROM SWAP_LIMIT_PRICE AS LIM
    INNER JOIN MEMPOOL_TRANSACTIONS AS MEM ON LIM.CALL_TX_HASH = MEM.HASH
    """

    col_rename = dict(
        call_block_number='block_number',
        contract_address='pool',
    )

    df = pd.read_sql_query(query, engine).rename(columns=col_rename).sort_values(by=['pool', 'block_number'])

    print("Wrangling the data...")

    swap_counts = df.groupby(by=['pool', 'block_number'])[['call_success']].count().sort_values('call_success', ascending=False)

    single_swap_blocks = swap_counts[swap_counts == 1].sort_index()

    # Make into a dictionary of the form {block: [block_number]}
    single_swap_blocks = single_swap_blocks.reset_index()
    single_swap_blocks = single_swap_blocks.groupby('pool')['block_number'].apply(list).to_dict()

    single_swap_blocks = dict(sorted(single_swap_blocks.items(), key=lambda item: len(item[1]), reverse=True))


    failed_pools = []
    pbar = tqdm(single_swap_blocks.items())
    for pool_addr, blocks in pbar:
        try:
            load_pool_from_blob(
                pool_addr,
                postgres_uri_us,
                azure_connection_string=azure_storage_uri,
                container_name="uniswap-v3-pool-cache",
                verbose=False,
                pbar=pbar,
            )
        except AssertionError:
            failed_pools.append(pool_addr)
            pbar.set_description_str(f"Loading pools ({len(failed_pools)} failed)")
            continue


    print("The following pools failed to load:")
    print(failed_pools)
