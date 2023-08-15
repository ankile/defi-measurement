

import os
import sys
import math

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from datetime import datetime, timezone


from dotenv import load_dotenv
from pool_state import v3Pool
from sqlalchemy import create_engine
from tqdm import tqdm

load_dotenv(override=True)
# from experiments.random_permutations import load_pool

from experiments.preload_pool_cache import load_pool_from_blob


from decimal import getcontext

getcontext().prec = 100  # Set the precision high enough for our purposes


# Read in the environment variables
postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
postgres_uri_us = os.environ["POSTGRESQL_URI_US"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]


# ## Get the Data


engine = create_engine(postgres_uri_mp)

query = """
SELECT *
FROM SWAP_LIMIT_PRICE AS LIM
INNER JOIN MEMPOOL_TRANSACTIONS AS MEM ON LIM.transaction_hash = MEM.HASH
"""

# col_rename = dict(
#     call_block_number='block_number',
#     contract_address='pool',
# )

# df = pd.read_sql_query(query, engine).rename(columns=col_rename).sort_values(by=['pool', 'block_number'])
df = pd.read_sql_query(query, engine)

# ### Populate the swap data we have with the block number that the swap appeared in


engine = create_engine(postgres_uri_us)

block_numbers = pd.read_sql_query(
    """
    SELECT block_number, tx_hash, block_ts
    FROM swaps
    WHERE block_number >= 17400000
    ORDER BY block_ts ASC
    """,
    engine
).set_index('tx_hash')



block_number_dict = block_numbers[~block_numbers.index.duplicated(keep='first')].to_dict(orient="index")


dataset = df.assign(block_number=df.transaction_hash.map(lambda x: block_number_dict[x]['block_number'] if x in block_number_dict else None))
dataset = dataset[~dataset.block_number.isna()]



swap_counts = dataset.groupby(by=['pool', 'block_number'])[['transaction_hash']].count().sort_values('transaction_hash', ascending=False)

swap_counts[swap_counts == 1].transaction_hash.sum(), swap_counts[swap_counts > 1].transaction_hash.sum(), swap_counts.transaction_hash.sum()

# ## Create Sandwich Attacks on Single Swaps
# 
# Start with this to validate the approach.


single_swap_blocks = swap_counts[swap_counts == 1].sort_index()

df_single = dataset.set_index(['pool', 'block_number']).loc[single_swap_blocks.index]

# Group by level 0 and count unique values in level 1
grouped_counts = df_single.groupby(level=0).apply(lambda x: x.index.get_level_values(1).nunique())

# Sort the indices based on the counts
sorted_indices = grouped_counts.sort_values(ascending=False).index

# Reindex the DataFrame based on this sorted order
df_single_sorted = df_single.loc[sorted_indices]


# Keep only the swap in subset for now
df_single_sorted = df_single_sorted[df_single_sorted.transaction_type == 'V3_SWAP_EXACT_IN'].drop(columns=['transaction_type', 'recipient', 'amountOut', 'amountInMax', 'payerIsUser', 'transaction_hash'])

# ## Get a pool and do some initial testing


def to_price(inv_sqrt_price):
    return 1/(inv_sqrt_price)**2 * 1e12


pool: v3Pool = load_pool_from_blob(
    "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
    postgres_uri_us,
    azure_storage_uri,
    "uniswap-v3-pool-cache",
    verbose=True,
    invalidate_before_date=datetime(2023, 8, 15, tzinfo=timezone.utc),
)


from dataclasses import dataclass
from datetime import datetime
from typing import Union

@dataclass
class SwapData:
    pool: str
    block_number: float
    amountIn: str
    amountOutMin: str
    token0: str
    fee: str
    token1: str
    hash: str
    first_seen: Union[datetime, None]
    # eth_in_float: float



def valid_frontrun(pool: v3Pool, swap: SwapData, frontrun_input) -> bool:
    try:
        curr_price = pool.getPriceAt(swap.block_number)

        swap_params = {
            "input": frontrun_input,
            "tokenIn": swap.token0,
            "as_of": swap.block_number,
            "gasFee": True,
            "givenPrice": curr_price,
        }

        _, heur = pool.swapIn(swap_params)

        swap_params = {
            "input": int(swap.amountIn),
            "tokenIn": swap.token0,
            "as_of": swap.block_number,
            "gasFee": True,
            "givenPrice": heur.sqrtP_next,
        }

        output, heur = pool.swapIn(swap_params)

        return output >= int(swap.amountOutMin)
    
    except AssertionError:
        print("Assertion error in [valid_frontrun], return False")
        return False


def exponential_search(pool: v3Pool, swap: SwapData, start=1e6, max_tries=100, factor=8) -> (int, int):
    lower = 0
    upper = start
    
    for i in range(max_tries):
        # print(f"{i}: Trying {upper / 1e18}")
        if not valid_frontrun(pool, swap, upper):
            return lower, upper
        else:
            lower = upper
            upper *= factor

    else:
        raise Exception("Exponential search exceeded max tries")
    
def binary_search(pool: v3Pool, swap: SwapData, lower, upper, max_tries=100) -> int:
    for i in range(max_tries):
        # print(f"{i}: Trying {lower / 1e18} - {upper / 1e18}")
        if math.isclose(lower, upper, abs_tol=1e18):
            return lower
        
        mid = (lower + upper) // 2
        if valid_frontrun(pool, swap, mid):
            lower = mid
        else:
            upper = mid

    else:
        raise Exception("Binary search exceeded max tries")
    

def max_frontrun(pool: v3Pool, swap: SwapData, start=1e6, factor=2) -> int:
    lower, upper = exponential_search(pool, swap, start=start, factor=factor)
    return binary_search(pool, swap, lower, upper)
    


def single_sandwich_mev(pool: v3Pool, swap: SwapData, frontrun_input: int, pool_fee=True) -> (int, float):
    total_gas = 0
    if not pool_fee:
        orig_fee = pool.fee
        pool.fee = 0

    curr_price = pool.getPriceAt(swap.block_number)

    swap_params = {
        "input": frontrun_input,
        "tokenIn": swap.token0,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": curr_price,
    }

    output_frontrun, heur = pool.swapIn(swap_params)
    total_gas += heur.gas_fee

    swap_params = {
        "input": int(swap.amountIn),
        "tokenIn": swap.token0,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": heur.sqrtP_next,
    }

    output, heur = pool.swapIn(swap_params)
    swap_params = {
        "input": output_frontrun,
        "tokenIn": swap.token1,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": heur.sqrtP_next,
    }

    output, heur = pool.swapIn(swap_params)
    total_gas += heur.gas_fee

    if not pool_fee:
        pool.fee = orig_fee

    return output - frontrun_input, total_gas


def auto_sandwich_mev(pool: v3Pool, swap: SwapData, start=1e18, factor=2, pool_fee=True) -> (int, float, int):
    frontrun_input = max_frontrun(pool, swap, start=start, factor=factor)
    profit, gas_fee = single_sandwich_mev(pool, swap, frontrun_input, pool_fee=pool_fee)
    return profit, gas_fee, frontrun_input


profits = {}

curr_pool = None

for swap_dict in tqdm(df_single_sorted.reset_index().to_dict(orient='records')):
    swap = SwapData(**swap_dict)

    if not curr_pool or curr_pool.pool != swap.pool:
        curr_pool: v3Pool = load_pool_from_blob(
            swap.pool,
            postgres_uri_us,
            azure_storage_uri,
            "uniswap-v3-pool-cache",
            verbose=True,
            invalidate_before_date=datetime(2023, 8, 16, tzinfo=timezone.utc),
        )

    profit, gas_fee, frontrun_input = auto_sandwich_mev(pool, swap, start=1e20, factor=4, pool_fee=True)
    profit_nofee, _ = single_sandwich_mev(pool, swap, frontrun_input, pool_fee=False)

    profits[swap.hash] = dict(
        block_number=swap.block_number,
        pool=swap.pool,
        profit=profit,
        profit_nofee=profit_nofee,
        gas_fee_eth=gas_fee,
        frontrun_input=frontrun_input,
    )

profits = pd.DataFrame.from_dict(profits, orient='index')

profits.to_csv("single_swap_profits.csv")
