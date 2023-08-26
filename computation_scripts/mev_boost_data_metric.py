import math
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

current_path = sys.path[0]
sys.path.append(current_path[: current_path.find("defi-measurement")] + "liquidity-distribution-history")

sys.path.append("..")

from datetime import datetime, timezone
from itertools import permutations
from multiprocessing import Pool
from typing import Iterable, cast

from pool_state import v3Pool
from preload_pool_cache import load_pool_from_blob
from sqlalchemy import ARRAY, CHAR, Boolean, Column, DateTime, Double, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from tqdm import tqdm

import argparse

# If the `output` directory doesn't exist, create it
if not os.path.exists("output"):
    os.mkdir("output")

load_dotenv()

postgres_uri = os.environ["POSTGRESQL_URI"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

Base = declarative_base()
engine = create_engine(postgres_uri)
SessionLocalMP = sessionmaker(bind=engine)


class BlockPoolMetrics(Base):
    __tablename__ = "block_pool_metrics"

    # Meta Data
    block_number = Column(Integer, primary_key=True)
    pool_address = Column(String, primary_key=True)
    num_transactions = Column(Integer)
    n_buys = Column(Integer)
    n_sells = Column(Integer)
    baseline_price = Column(Double)
    created_at = Column(DateTime, default=datetime.now)

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

    # T* Data
    tstar_l1 = Column(Double, nullable=True)
    tstar_l2 = Column(Double, nullable=True)
    tstar_linf = Column(Double, nullable=True)


Base.metadata.create_all(engine)


def get_swaps_for_address(address, min_block, max_block):
    return pd.read_sql_query(
        f"""
        SELECT * FROM swaps
        WHERE block_number >= {min_block}
        AND block_number <= {max_block}
        AND address = '{address}'
        """,
        postgres_uri,
    )


def get_token_info():
    token_info = pd.read_sql_query(
        f"""
        SELECT * FROM token_info
        WHERE decimals0 IS NOT NULL
        AND decimals1 IS NOT NULL
        """,
        postgres_uri,
    ).set_index("pool")[["token0", "token1", "decimals0", "decimals1"]]

    return token_info.to_dict(orient="index")


def get_mev_boost_values() -> dict[int, float]:
    res = pd.read_sql_query(
        """
        SELECT block_number, mevboost_value
        FROM
            mev_boost
        """,
        postgres_uri,
    )
    return dict(zip(res.block_number, res.mevboost_value))


def get_pool_block_pairs(*, limit, offset, only_unprocessed) -> pd.DataFrame:
    return pd.read_sql_query(
        f"""
        SELECT sc.address, sc.block_number FROM swap_counts AS sc
        INNER JOIN token_info AS ti ON sc.address = ti.pool
            AND ti.decimals0 IS NOT NULL AND ti.decimals1 IS NOT NULL
        {"LEFT JOIN block_pool_metrics AS bpm ON sc.address = bpm.pool_address AND sc.block_number = bpm.block_number" if only_unprocessed else ""}
        WHERE sc.block_number >= 15537940 AND sc.block_number <= 17959956
            {"AND bpm.pool_address IS NULL" if only_unprocessed else ""}
        ORDER BY sc.address ASC, sc.block_number ASC
        LIMIT {limit} OFFSET {offset}
        """,
        postgres_uri,
    )


def get_price(sqrt_price, pool_addr, token_info):
    return 1 / (sqrt_price**2) / 10 ** (token_info[pool_addr]["decimals0"] - token_info[pool_addr]["decimals1"])


def get_pool(address, it):
    return load_pool_from_blob(
        address,
        postgres_uri,
        azure_storage_uri,
        "uniswap-v3-pool-cache",
        verbose=False,
        invalidate_before_date=datetime(2023, 8, 20, tzinfo=timezone.utc),
        pbar=it,
    )


def norm(prices, norm):
    if norm == 1:
        return float(np.sum(np.abs(prices)))
    elif norm == 2:
        return float(np.sqrt(np.sum(prices**2)))
    elif norm == np.inf:
        return float(np.max(np.abs(prices)))
    else:
        raise ValueError("Invalid norm")


def do_swap(swap, curr_price, pool, token_info):
    token_in = token_info[swap.address]["token0"] if int(swap.amount0) > 0 else token_info[swap.address]["token1"]
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


def get_pool_block_count(*, only_unprocessed) -> int:
    n_pool_block_pairs = pd.read_sql_query(
        f"""
        SELECT COUNT(*)
        FROM swap_counts AS sc
        INNER JOIN token_info AS ti ON sc.address = ti.pool
            AND ti.decimals0 IS NOT NULL AND ti.decimals1 IS NOT NULL
        {"LEFT JOIN block_pool_metrics AS bpm ON sc.address = bpm.pool_address AND sc.block_number = bpm.block_number" if only_unprocessed else ""}
        WHERE sc.block_number >= 15537940 AND sc.block_number <= 17959956
            {"AND bpm.pool_address IS NULL" if only_unprocessed else ""}
        """,
        postgres_uri,
    ).iloc[0, 0]

    return int(n_pool_block_pairs)  # type: ignore


def set_metrics(blockpool_metric: BlockPoolMetrics, field: str, prices: list, ordering: list):
    assert field in ["realized", "volume_heur"]
    setattr(blockpool_metric, f"{field}_prices", prices)  # type: ignore
    setattr(blockpool_metric, f"{field}_order", ordering)  # type: ignore

    prices_np = np.array(prices) - blockpool_metric.baseline_price
    setattr(blockpool_metric, f"{field}_l1", norm(prices_np, 1))  # type: ignore
    setattr(blockpool_metric, f"{field}_l2", norm(prices_np, 2))  # type: ignore
    setattr(blockpool_metric, f"{field}_linf", norm(prices_np, np.inf))  # type: ignore


def run_swap_order(pool: v3Pool, swaps: Iterable, block_number: int, token_info):
    prices = []
    ordering = []
    curr_price_sqrt = pool.getPriceAt(block_number)

    for swap in swaps:
        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        prices.append(get_price(heur.sqrtP_next, swap.address, token_info))
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")
        curr_price_sqrt = heur.sqrtP_next

    return prices, ordering


def realized_measurement(
    pool: v3Pool, swaps: pd.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: dict
):
    # Run the realized measurement
    prices, ordering = run_swap_order(pool, swaps.itertuples(index=False, name="Swap"), block_number, token_info)

    set_metrics(blockpool_metric, "realized", prices, ordering)


def volume_heuristic(
    pool: v3Pool, swaps: pd.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: dict
):
    pool_addr = blockpool_metric.pool_address
    baseline_price = blockpool_metric.baseline_price

    # Run the volume heuristic measurement
    curr_price_sqrt = cast(float, pool.getPriceAt(block_number))
    curr_price = get_price(curr_price_sqrt, pool_addr, token_info)

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
        [row for _, row in buy_df.sort_values("amount0_float", ascending=False).iterrows()]
        if buy_df.shape[0] > 0
        else []
    )
    sells = (
        [row for _, row in sell_df.sort_values("amount1_float", ascending=False).iterrows()]
        if sell_df.shape[0] > 0
        else []
    )

    # While we're still in the core
    while len(buys) > 0 and len(sells) > 0:
        if curr_price == baseline_price:
            # If we're at the baseline price, we can swap in either direction
            # Choose the one that moves the price the least
            buy_diff = (
                get_price(do_swap(buys[-1], curr_price_sqrt, pool, token_info).sqrtP_next, pool_addr, token_info)
                - baseline_price
            )
            sell_diff = (
                get_price(do_swap(sells[-1], curr_price_sqrt, pool, token_info).sqrtP_next, pool_addr, token_info)
                - baseline_price
            )

            if abs(buy_diff) < abs(sell_diff):
                swap = buys.pop(-1)
            else:
                swap = sells.pop(-1)
        elif curr_price <= baseline_price:
            swap = buys.pop(-1)
        else:
            swap = sells.pop(-1)

        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        curr_price_sqrt = heur.sqrtP_next
        curr_price = get_price(curr_price_sqrt, swap.address, token_info)
        prices.append(curr_price)
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")

    # Process whatever is left in the tail
    for swap in (buys + sells)[::-1]:
        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        curr_price_sqrt = heur.sqrtP_next
        prices.append(get_price(curr_price_sqrt, swap.address, token_info))
        ordering.append(f"{swap.transaction_index:03}_{swap.log_index:03}")

    blockpool_metric.volume_heur_prices = prices  # type: ignore
    blockpool_metric.volume_heur_order = ordering  # type: ignore
    prices_np = np.array(prices) - baseline_price

    blockpool_metric.volume_heur_l1 = norm(prices_np, 1)  # type: ignore
    blockpool_metric.volume_heur_l2 = norm(prices_np, 2)  # type: ignore
    blockpool_metric.volume_heur_linf = norm(prices_np, np.inf)  # type: ignore


def tstar(pool: v3Pool, swaps: pd.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: dict):
    # Run the t* measurement if not more than 7 swaps
    if swaps.shape[0] > 7:
        return

    best_scores = {
        "l1": math.inf,
        "l2": math.inf,
        "linf": math.inf,
    }
    for swap_perm in permutations(swaps.itertuples(index=False, name="Swap")):
        prices, _ = run_swap_order(pool, swap_perm, block_number, token_info)
        best_scores["l1"] = min(best_scores["l1"], norm(np.array(prices) - blockpool_metric.baseline_price, 1))
        best_scores["l2"] = min(best_scores["l2"], norm(np.array(prices) - blockpool_metric.baseline_price, 2))
        best_scores["linf"] = min(best_scores["linf"], norm(np.array(prices) - blockpool_metric.baseline_price, np.inf))

    blockpool_metric.tstar_l1 = best_scores["l1"]  # type: ignore
    blockpool_metric.tstar_l2 = best_scores["l2"]  # type: ignore
    blockpool_metric.tstar_linf = best_scores["linf"]  # type: ignore


def copy_over(blockpool_metric: BlockPoolMetrics, to: list[str]):
    for field in to:
        setattr(blockpool_metric, f"{field}_prices", blockpool_metric.realized_prices)
        setattr(blockpool_metric, f"{field}_order", blockpool_metric.realized_order)
        setattr(blockpool_metric, f"{field}_l1", blockpool_metric.realized_l1)
        setattr(blockpool_metric, f"{field}_l2", blockpool_metric.realized_l2)
        setattr(blockpool_metric, f"{field}_linf", blockpool_metric.realized_linf)


def run_metrics(limit, offset, process_id, token_info, mev_boost_values, only_unprocessed):
    pool_block_pairs = get_pool_block_pairs(limit=limit, offset=offset, only_unprocessed=only_unprocessed)

    it = tqdm(total=pool_block_pairs.shape[0], position=process_id, desc=f"[{process_id}] ({offset}-{offset+limit})")
    pool = get_pool(pool_block_pairs.address[0], it)

    program_start = datetime.now()

    errors = 0
    successes = 0

    for pool_addr, df in pool_block_pairs.groupby("address"):
        it.set_description(f"[{process_id}] ({offset}-{offset+limit}) Processing pool {pool_addr}")

        if pool_addr not in token_info:
            continue

        if pool_addr != pool.pool:
            pool = get_pool(pool_addr, it)

        swaps_for_pool = get_swaps_for_address(pool_addr, df.block_number.min(), df.block_number.max())

        for block_number in df.block_number.unique():
            block_number = int(block_number)
            it.set_postfix(errors=errors, successes=successes)
            it.update(1)

            try:
                swaps = swaps_for_pool[swaps_for_pool.block_number == block_number].sort_values("transaction_index")

                if swaps.shape[0] == 0:
                    continue

                curr_price_sqrt = pool.getPriceAt(block_number)

                blockpool_metric = BlockPoolMetrics(
                    block_number=block_number,
                    pool_address=pool_addr,
                    num_transactions=swaps.shape[0],
                    n_buys=swaps[~swaps.amount0.str.startswith("-")].shape[0],
                    n_sells=swaps[swaps.amount0.str.startswith("-")].shape[0],
                    mev_boost=block_number in mev_boost_values,
                    mev_boost_amount=mev_boost_values.get(block_number, 0),
                    baseline_price=get_price(curr_price_sqrt, pool_addr, token_info),
                )

                # Run the baseline measurement
                realized_measurement(pool, swaps, block_number, blockpool_metric, token_info)

                if swaps.shape[0] > 1:
                    volume_heuristic(pool, swaps, block_number, blockpool_metric, token_info)
                    tstar(pool, swaps, block_number, blockpool_metric, token_info)
                else:
                    copy_over(blockpool_metric, to=["volume_heur", "tstar"])

                with SessionLocalMP() as session:
                    session.add(blockpool_metric)
                    session.commit()
                    session.close()

                successes += 1

            except Exception as e:
                errors += 1
                with open(f"outout/error-{program_start}.log", "a") as f:
                    f.write(f"Error processing block {block_number} for pool {pool_addr}: {e}\n")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MEV Boost Data Metrics")
    parser.add_argument("--n-cpus", type=int, default=1, help="Number of CPUs to use")
    args = parser.parse_args()

    only_unprocessed = True

    print(f"Starting MEV Boost Data Metric Calculations with {args.n_cpus} CPUs")

    n_pool_block_pairs = get_pool_block_count(only_unprocessed=only_unprocessed)
    print(f"Processing {n_pool_block_pairs:,} pool-block pairs")

    mev_boost_values = get_mev_boost_values()
    token_info = get_token_info()

    n_processes = args.n_cpus

    # Calculate the chunk size
    chunk_size = n_pool_block_pairs // n_processes

    # Define a function to be mapped
    def run_chunk(i):
        offset = i * chunk_size
        run_metrics(
            limit=chunk_size,
            offset=offset,
            process_id=i,
            token_info=token_info,
            mev_boost_values=mev_boost_values,
            only_unprocessed=only_unprocessed,
        )

    # Create a pool of workers and map the function across the input values
    with Pool(n_processes) as pool:
        pool.map(run_chunk, range(n_processes))

    print("All processes completed.")
