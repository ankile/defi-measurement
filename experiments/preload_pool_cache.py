# %% [markdown]
# # Run Sandwich Attacks on Swaps from the Public Mempool

# %%
print("Starting...")

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

load_dotenv(override=True)
from random_permutations import load_pool

from decimal import getcontext

getcontext().prec = 100  # Set the precision high enough for our purposes


# Read in the environment variables
postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
postgres_uri_us = os.environ["POSTGRESQL_URI_US"]

engine = create_engine(postgres_uri_mp)

# %% [markdown]
# ## Get the Data

print("Getting the data...")

# %%
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

# %%
swap_counts = df.groupby(by=['pool', 'block_number'])[['call_success']].count().sort_values('call_success', ascending=False)

swap_counts[swap_counts == 1].call_success.sum(), swap_counts[swap_counts > 1].call_success.sum(), swap_counts.call_success.sum()

# %% [markdown]
# ## Create Sandwich Attacks on Single Swaps
# 
# Start with this to validate the approach.

# %%
single_swap_blocks = swap_counts[swap_counts == 1].sort_index()

# Make into a dictionary of the form {block: [block_number]}
single_swap_blocks = single_swap_blocks.reset_index()
single_swap_blocks = single_swap_blocks.groupby('pool')['block_number'].apply(list).to_dict()

single_swap_blocks = dict(sorted(single_swap_blocks.items(), key=lambda item: len(item[1]), reverse=True))


# %%
# Just starting to load some pools into the cache to speed up the process down the line
for pool_addr, blocks in tqdm(single_swap_blocks.items()):
    try:
        pool = load_pool(pool_addr, postgres_uri_us, verbose=False)
    except AssertionError:
        print(f"Pool {pool_addr} not found in the database")
        continue
