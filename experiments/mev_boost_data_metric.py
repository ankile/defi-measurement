import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np


import os
import sys
import math

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")
from tqdm import tqdm

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Double,
    DateTime,
    Boolean,
    BigInteger,
    Float,
    ARRAY,
    CHAR,
)
from sqlalchemy.orm import sessionmaker, declarative_base

from typing import cast


load_dotenv()

from datetime import datetime, timezone
from preload_pool_cache import load_pool_from_blob

postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
postgres_uri_us = os.environ["POSTGRESQL_URI_US"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

## Get data
minmax_block = pd.read_sql_query(
    """
    SELECT
        MIN(block_number) AS min_block,
        MAX(block_number) AS max_block
    FROM
        mev_boost
    """,
    postgres_uri_mp,
)

min_block = minmax_block["min_block"][0]
max_block = minmax_block["max_block"][0]

print(f"Min block: {min_block}, Max block: {max_block}")


def get_swaps_for_address(address, min_block, max_block):
    return pd.read_sql_query(
        f"""
        SELECT * FROM swaps
        WHERE block_number >= {min_block}
        AND block_number <= {max_block}
        AND address = '{address}'
        """,
        postgres_uri_us,
    )


token_info = pd.read_sql_query(
    f"""
    SELECT * FROM token_info
    WHERE decimals0 IS NOT NULL
    AND decimals1 IS NOT NULL
    """,
    postgres_uri_us,
).set_index("pool")[["token0", "token1", "decimals0", "decimals1"]]

token_info = token_info.to_dict(orient="index")


def get_mev_boost_values() -> dict[int, float]:
    res = pd.read_sql_query(
        """
        SELECT block_number, mevboost_value
        FROM
            mev_boost
        """,
        postgres_uri_mp,
    )
    return dict(zip(res.block_number, res.mevboost_value))


def get_pool_block_pairs(limit, offset) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT sc.address, sc.block_number FROM swap_counts AS sc
        INNER JOIN token_info AS ti ON sc.address = ti.pool
            AND ti.decimals0 IS NOT NULL AND ti.decimals1 IS NOT NULL
        WHERE sc.block_number >= 15537940 AND sc.block_number <= 17959956
        ORDER BY sc.address ASC, sc.block_number ASC
        LIMIT {limit} OFFSET {offset}
        """,
        postgres_uri_us,
    )


mev_boost_values = get_mev_boost_values()


def get_price(sqrt_price, pool_addr):
    return (
        1
        / (sqrt_price**2)
        / 10
        ** (token_info[pool_addr]["decimals0"] - token_info[pool_addr]["decimals1"])
    )


def get_pool(address, it):
    return load_pool_from_blob(
        address,
        postgres_uri_us,
        azure_storage_uri,
        "uniswap-v3-pool-cache",
        verbose=False,
        invalidate_before_date=datetime(2023, 8, 20, tzinfo=timezone.utc),
        pbar=it,
    )


from sqlalchemy import Boolean


engine_mp = create_engine(postgres_uri_mp)

SessionLocalMP = sessionmaker(bind=engine_mp)

program_start = datetime.now()
Base = declarative_base()


class BlockMetrics(Base):
    __tablename__ = "block_metrics"

    # Meta Data
    block_number = Column(Integer, primary_key=True)
    pool_address = Column(String, primary_key=True)
    num_transactions = Column(Integer)
    n_buys = Column(Integer)
    n_sells = Column(Integer)
    baseline_price = Column(Double)

    # MEV Data
    mev_boost = Column(Boolean)
    mev_boost_amount = Column(Double)

    # Realized Data
    realized_order = Column(ARRAY(CHAR(7)))
    realized_prices = Column(ARRAY(Double))
    realized_l1 = Column(Double)
    realized_l2 = Column(Double)
    realized_linf = Column(Double)

    # Volume Heuristic Data
    volume_heur_order = Column(ARRAY(CHAR(7)))
    volume_heur_prices = Column(ARRAY(Double))
    volume_heur_l1 = Column(Double)
    volume_heur_l2 = Column(Double)
    volume_heur_linf = Column(Double)


Base.metadata.create_all(engine_mp)


def norm(prices, norm):
    if norm == 1:
        return float(np.sum(np.abs(prices)))
    elif norm == 2:
        return float(np.sqrt(np.sum(prices**2)))
    elif norm == np.inf:
        return float(np.max(np.abs(prices)))
    else:
        raise ValueError("Invalid norm")


def do_swap(swap, curr_price, pool):
    token_in = (
        token_info[swap.address]["token0"]
        if int(swap.amount0) > 0
        else token_info[swap.address]["token1"]
    )
    input_amount = int(swap.amount0) if int(swap.amount0) > 0 else int(swap.amount1)

    _, heur = pool.swapIn(
        {
            "tokenIn": token_in,
            "input": input_amount,
            "as_of": swap.block_number,
            "gas": True,
            "givenPrice": curr_price,
        }
    )

    return heur


def get_pool_block_count() -> int:
    n_pool_block_pairs = pd.read_sql_query(
        """
        SELECT COUNT(*)
        FROM swap_counts AS sc
        INNER JOIN token_info AS ti ON sc.address = ti.pool
            AND ti.decimals0 IS NOT NULL AND ti.decimals1 IS NOT NULL
        WHERE sc.block_number >= 15537940 AND sc.block_number <= 17959956;
        """,
        postgres_uri_us,
    ).iloc[0, 0]

    return int(n_pool_block_pairs)  # type: ignore


def run_metrics(limit, offset, process_id):
    pool_block_pairs = get_pool_block_pairs(limit, offset)

    it = tqdm(total=pool_block_pairs.shape[0], position=process_id, desc=f"[{process_id}] ({offset}-{offset+limit})")
    pool = get_pool(pool_block_pairs.address[0], it)

    program_start = datetime.now()

    errors = 0
    successes = 0

    for pool_addr, df in pool_block_pairs.groupby("address"):
        it.set_description(
            f"[{process_id}] ({offset}-{offset+limit}) Processing pool {pool_addr}"
        )

        if pool_addr not in token_info:
            print(f"Skipping pool {pool_addr} because it is not in token_info")
            continue

        if pool_addr != pool.pool:
            pool = get_pool(pool_addr, it)

        swaps_for_pool = get_swaps_for_address(
            pool_addr, df.block_number.min(), df.block_number.max()
        )

        block_numbers = df.block_number.unique()

        for block_number in df.block_number.unique():
            it.set_postfix(errors=errors, successes=successes)
            it.update(1)

            try:
                swaps = swaps_for_pool[
                    swaps_for_pool.block_number == block_number
                ].sort_values("transaction_index")

                if swaps.shape[0] == 0:
                    continue

                curr_price = pool.getPriceAt(block_number)

                swap_metric = BlockMetrics(
                    block_number=int(block_number),
                    pool_address=pool_addr,
                    num_transactions=swaps.shape[0],
                    n_buys=swaps[~swaps.amount0.str.startswith("-")].shape[0],
                    n_sells=swaps[swaps.amount0.str.startswith("-")].shape[0],
                    mev_boost=block_number in mev_boost_values,
                    mev_boost_amount=mev_boost_values.get(block_number, 0),
                    baseline_price=get_price(curr_price, pool_addr),
                )

                # Run the baseline measurement
                prices = []
                ordering = []
                for i, (_, swap) in enumerate(swaps.iterrows()):
                    heur = do_swap(swap, curr_price, pool)

                    prices.append(get_price(heur.sqrtP_next, swap.address))
                    ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")
                    curr_price = heur.sqrtP_next

                swap_metric.realized_prices = prices  # type: ignore
                swap_metric.realized_order = ordering  # type: ignore
                prices_np = np.array(prices) - swap_metric.baseline_price
                swap_metric.realized_l1 = norm(prices_np, 1)  # type: ignore
                swap_metric.realized_l2 = norm(prices_np, 2)  # type: ignore
                swap_metric.realized_linf = norm(prices_np, np.inf)  # type: ignore

                if swaps.shape[0] == 1:
                    swap_metric.volume_heur_prices = prices  # type: ignore
                    swap_metric.volume_heur_order = ordering  # type: ignore
                    swap_metric.volume_heur_l1 = swap_metric.realized_l1  # type: ignore
                    swap_metric.volume_heur_l2 = swap_metric.realized_l2  # type: ignore
                    swap_metric.volume_heur_linf = swap_metric.realized_linf  # type: ignore

                else:
                    # Run the volume heuristic measurement
                    curr_price_sqrt = cast(float, pool.getPriceAt(block_number))
                    curr_price = get_price(curr_price_sqrt, pool_addr)
                    # baseline_price = curr_price_sqrt
                    prices = []
                    ordering = []

                    # Split the swaps into the set of buys and sells and order by volume ascending
                    swaps = swaps.assign(
                        amount0_float=swaps.amount0.astype(float),
                        amount1_float=swaps.amount1.astype(float),
                    )
                    buy_df = swaps[~swaps.amount0.str.startswith("-")]
                    sell_df = swaps[swaps.amount0.str.startswith("-")]
                    buys = (
                        [
                            row
                            for _, row in buy_df.sort_values(
                                "amount0_float", ascending=False
                            ).iterrows()
                        ]
                        if buy_df.shape[0] > 0
                        else []
                    )
                    sells = (
                        [
                            row
                            for _, row in sell_df.sort_values(
                                "amount1_float", ascending=False
                            ).iterrows()
                        ]
                        if sell_df.shape[0] > 0
                        else []
                    )

                    # While wer're still in the core
                    while len(buys) > 0 and len(sells) > 0:
                        if curr_price <= swap_metric.baseline_price:
                            swap = buys.pop(-1)
                        else:
                            swap = sells.pop(-1)

                        heur = do_swap(swap, curr_price_sqrt, pool)

                        curr_price_sqrt = heur.sqrtP_next
                        curr_price = get_price(curr_price_sqrt, swap.address)
                        prices.append(curr_price)
                        ordering.append(
                            f"{swap.transaction_index:03}_{swap.log_index:03}"
                        )

                    # Process whatever is left in the tail
                    for swap in (buys + sells)[::-1]:
                        heur = do_swap(swap, curr_price_sqrt, pool)

                        curr_price_sqrt = heur.sqrtP_next
                        prices.append(get_price(curr_price_sqrt, swap.address))
                        ordering.append(
                            f"{swap.transaction_index:03}_{swap.log_index:03}"
                        )

                    swap_metric.volume_heur_prices = prices  # type: ignore
                    swap_metric.volume_heur_order = ordering  # type: ignore
                    prices_np = np.array(prices) - swap_metric.baseline_price

                    swap_metric.volume_heur_l1 = norm(prices_np, 1)  # type: ignore
                    swap_metric.volume_heur_l2 = norm(prices_np, 2)  # type: ignore
                    swap_metric.volume_heur_linf = norm(prices_np, np.inf)  # type: ignore

                with SessionLocalMP() as session:
                    session.add(swap_metric)
                    session.commit()
                    session.close()

                successes += 1

            except Exception as e:
                errors += 1
                with open(f"error-{program_start}.log", "a") as f:
                    f.write(
                        f"Error processing block {block_number} for pool {pool_addr}: {e}\n"
                    )
                continue


from multiprocessing import Pool

n_pool_block_pairs = get_pool_block_count()
n_processes = 14

# Calculate the chunk size
chunk_size = n_pool_block_pairs // n_processes


# Define a function to be mapped
def run_chunk(i):
    offset = i * chunk_size
    run_metrics(chunk_size, offset, i)


# Create a pool of workers and map the function across the input values
with Pool(n_processes) as pool:
    pool.map(run_chunk, range(n_processes))

print("All processes completed.")
