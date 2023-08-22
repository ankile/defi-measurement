import os
import sys
import math

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from datetime import datetime, timezone

import argparse


# Set display options
pd.set_option("display.max_colwidth", None)  # Display entire cell content
pd.set_option("display.max_rows", 50)  # Display all rows
pd.set_option("display.max_columns", None)  # Display all columns


from dotenv import load_dotenv
from pool_state import v3Pool
from sqlalchemy import create_engine
from tqdm import tqdm

from moralis import evm_api


load_dotenv(override=True)
# from experiments.random_permutations import load_pool

from experiments.preload_pool_cache import load_pool_from_blob
from experiments.calculate_simple_sandwich import (
    get_data,
    SwapData,
    single_sandwich_mev,
    max_frontrun,
    auto_sandwich_mev,
    SimpleSandwich,
)


from sqlalchemy import create_engine, Column, String, Float, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker




# Read in the environment variables
postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
postgres_uri_us = os.environ["POSTGRESQL_URI_US"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
moralis_api_key = os.environ["MORALIS_API_KEY"]

engine = create_engine(postgres_uri_mp)

SessionLocal = sessionmaker(bind=engine)

def get_token_info():
    engine_us = create_engine(postgres_uri_us)

    token_info = pd.read_sql_table('token_info', engine_us)

    token_info0 = token_info[["token0", "decimals0"]].rename(columns={"token0": "token", "decimals0": "decimals"})
    token_info1 = token_info[["token1", "decimals1"]].rename(columns={"token1": "token", "decimals1": "decimals"})

    token_info = pd.concat([token_info0, token_info1], ignore_index=True).drop_duplicates().reset_index(drop=True).set_index("token")

    token_info = token_info.dropna()

    # Make a dictionary of the form {token: decimals}
    token_info = token_info.to_dict()["decimals"]

    return token_info


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    args = parser.parse_args()

    token_info = get_token_info()
    session = SessionLocal()

    rows = session.query(SimpleSandwich).filter(SimpleSandwich.profit_usd == None).all()

    errors = 0
    it = tqdm(rows)

    for row in it:
        it.set_postfix(errors=errors)

        try:
            params = {
                "chain": "eth",
                "to_block": row.block_number,
                "address": row.token_in,
            }
            result = evm_api.token.get_token_price(
                api_key=moralis_api_key,
                params=params, # type: ignore
            )

            usd_price = result["usdPrice"]

            if result is None or "usdPrice" not in result:
                raise Exception('result is None or "usdPrice" not in result')

            if (decimals := token_info.get(row.token_in, None)) is None:
                params = {
                    "chain": "eth",
                    "addresses": [
                        row.token_in,
                    ]
                }

                result = evm_api.token.get_token_metadata(
                    api_key=moralis_api_key,
                    params=params, # type: ignore
                )
                decimals = int(result[0]["decimals"])

            profit_usd_value = (row.profit_float / (10**decimals)) * usd_price
        except Exception as e:
            if args.debug:
                print(e)
            errors += 1
            profit_usd_value = -1
            continue

        
        # Update the table
        row.profit_usd = profit_usd_value
        session.commit()

        time.sleep(0.01)

    session.close()