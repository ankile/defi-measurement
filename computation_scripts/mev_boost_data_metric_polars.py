import math
import os
from pathlib import Path
import polars as pl

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from decimal import Decimal

from ipdb import set_trace as bp

from typing import Dict, Any


from datetime import datetime
from itertools import permutations
from multiprocessing import Pool, Lock
from typing import Iterable, cast

from v3.state import v3Pool
from tqdm import tqdm

from numpy.linalg import norm

import argparse

# If the `output` directory doesn't exist, create it
if not os.path.exists("output"):
    os.mkdir("output")

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import polars as pl


load_dotenv()


DATA_PATH = Path(os.environ["DATA_PATH"])


@dataclass
class BlockPoolMetrics:
    block_number: int
    pool_address: str
    num_transactions: int = 0
    n_buys: int = 0
    n_sells: int = 0
    baseline_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    mev_boost: bool = False
    mev_boost_amount: float = 0.0

    realized_order: List[str] = field(default_factory=list)
    realized_prices: List[float] = field(default_factory=list)
    realized_l1: float = 0.0
    realized_l2: float = 0.0
    realized_linf: float = 0.0

    volume_heur_order: List[str] = field(default_factory=list)
    volume_heur_prices: List[float] = field(default_factory=list)
    volume_heur_l1: float = 0.0
    volume_heur_l2: float = 0.0
    volume_heur_linf: float = 0.0

    tstar_l1: float = math.inf
    tstar_l2: float = math.inf
    tstar_linf: float = math.inf

    def increment_transactions(self, is_buy: bool = True):
        self.num_transactions += 1
        if is_buy:
            self.n_buys += 1
        else:
            self.n_sells += 1

    def add_realized_price(self, order: str, price: float):
        self.realized_order.append(order)
        self.realized_prices.append(price)

    def add_volume_heur_price(self, order: str, price: float):
        self.volume_heur_order.append(order)
        self.volume_heur_prices.append(price)

    def to_dict(self):
        return {
            "block_number": self.block_number,
            "pool_address": self.pool_address,
            "num_transactions": self.num_transactions,
            "n_buys": self.n_buys,
            "n_sells": self.n_sells,
            "baseline_price": self.baseline_price,
            "created_at": self.created_at,
            "mev_boost": self.mev_boost,
            "mev_boost_amount": self.mev_boost_amount,
            "realized_order": self.realized_order,
            "realized_prices": self.realized_prices,
            "realized_l1": self.realized_l1,
            "realized_l2": self.realized_l2,
            "realized_linf": self.realized_linf,
            "volume_heur_order": self.volume_heur_order,
            "volume_heur_prices": self.volume_heur_prices,
            "volume_heur_l1": self.volume_heur_l1,
            "volume_heur_l2": self.volume_heur_l2,
            "volume_heur_linf": self.volume_heur_linf,
            "tstar_l1": self.tstar_l1,
            "tstar_l2": self.tstar_l2,
            "tstar_linf": self.tstar_linf,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_polars_dataframe(self):
        data = self.to_dict()
        return pl.DataFrame(
            {
                **{k: [v] for k, v in data.items() if not isinstance(v, list)},
                "realized_order": [pl.Series(self.realized_order, dtype=pl.List(pl.Utf8))],
                "realized_prices": [pl.Series(self.realized_prices, dtype=pl.List(pl.Float64))],
                "volume_heur_order": [pl.Series(self.volume_heur_order, dtype=pl.List(pl.Utf8))],
                "volume_heur_prices": [pl.Series(self.volume_heur_prices, dtype=pl.List(pl.Float64))],
            }
        )


def get_swaps_for_address(address: str, min_block: int, max_block: int) -> pl.DataFrame:
    # Create a lazy frame from all parquet files
    df = pl.scan_parquet(DATA_PATH / "pool_swap_events" / "*.parquet")

    # Apply filters
    filtered_df = df.filter(
        (pl.col("block_number") >= min_block) & (pl.col("block_number") <= max_block) & (pl.col("address") == address)
    )

    # Collect and return the result
    return filtered_df.collect()


def get_token_info() -> Dict[str, Dict[str, Any]]:
    # Load the data from the Parquet file
    df = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Select the required columns
    token_info = df.select(
        [pl.col("pool"), pl.col("token0"), pl.col("token1"), pl.col("decimals0"), pl.col("decimals1")]
    )

    # Use struct to group the columns we want as a nested dictionary
    result = token_info.select(
        [pl.col("pool"), pl.struct(["token0", "token1", "decimals0", "decimals1"]).alias("info")]
    )

    result = result.collect().to_dict(as_series=False)

    # Convert to dictionary efficiently
    return dict(zip(result["pool"], result["info"]))


def get_mev_boost_values() -> dict[int, float]:
    """
    Get the MEV Boost values from the database

    Dictionary with block number as key and MEV Boost value as value
    """
    # Load the data from the Parquet file
    df = pl.read_parquet(DATA_PATH / "mev-boost" / "openethdata_eth_data_clean.parquet")

    # Select the relevant columns
    df = df.select(["block_number", "mevboost_value"])

    # Convert to dictionary
    mev_boost_values = dict(zip(df["block_number"].to_list(), df["mevboost_value"].to_list()))

    return mev_boost_values


def get_pool_block_pairs(*, limit: int, offset: int, only_unprocessed: bool) -> pl.DataFrame:
    # Load DataFrames lazily
    swap_counts = pl.scan_parquet(DATA_PATH / "swap_counts.parquet")
    token_info = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Check if block_pool_metrics.parquet exists
    block_pool_metrics_path = DATA_PATH / "block_pool_metrics/*.parquet"
    block_pool_metrics: Optional[pl.LazyFrame] = None
    if block_pool_metrics_path.exists():
        block_pool_metrics = pl.scan_parquet(block_pool_metrics_path)
    elif only_unprocessed:
        print(
            "Warning: only_unprocessed is True, but block_pool_metrics.parquet doesn't exist. Returning all pool-block pairs."
        )

    # Start building the query
    query = (
        swap_counts.filter((pl.col("block_number") >= 15537940) & (pl.col("block_number") <= 17959956))
        .join(
            token_info.filter(pl.col("decimals0").is_not_null() & pl.col("decimals1").is_not_null()),
            left_on="address",
            right_on="pool",
        )
        .select(["address", "block_number"])
    )

    if only_unprocessed and block_pool_metrics is not None:
        query = (
            query.join(
                block_pool_metrics,
                left_on=["address", "block_number"],
                right_on=["pool_address", "block_number"],
                how="left",
            )
            .filter(pl.col("pool_address").is_null())
            .select(["address", "block_number"])
        )

    # Apply ordering, limit, and offset
    query = query.sort(["address", "block_number"]).slice(offset, limit)

    # Collect the results
    return query.collect()


def get_price(sqrt_price, pool_addr, token_info):
    return 1 / (sqrt_price**2) / 10 ** (token_info[pool_addr]["decimals0"] - token_info[pool_addr]["decimals1"])


def get_pool(address, update=False):
    return v3Pool(
        address,
        "ethereum",
        update=update,
        update_from="allium",
    )


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


def get_pool_block_count(*, only_unprocessed: bool) -> int:
    # Load DataFrames lazily
    swap_counts = pl.scan_parquet(DATA_PATH / "swap_counts.parquet")
    token_info = pl.scan_parquet(DATA_PATH / "pool_token_info.parquet")

    # Check if block_pool_metrics.parquet exists
    block_pool_metrics_path = DATA_PATH / "block_pool_metrics/*.parquet"
    block_pool_metrics: Optional[pl.LazyFrame] = None
    if block_pool_metrics_path.exists():
        block_pool_metrics = pl.scan_parquet(block_pool_metrics_path)
    else:
        print("block_pool_metrics.parquet not found. Proceeding without it.")

    # Start building the query
    query = swap_counts.filter((pl.col("block_number") >= 15537940) & (pl.col("block_number") <= 17959956)).join(
        token_info.filter(pl.col("decimals0").is_not_null() & pl.col("decimals1").is_not_null()),
        left_on="address",
        right_on="pool",
    )

    if only_unprocessed and block_pool_metrics is not None:
        query = query.join(
            block_pool_metrics,
            left_on=["address", "block_number"],
            right_on=["pool_address", "block_number"],
            how="left",
        ).filter(pl.col("pool_address").is_null())
    elif only_unprocessed and block_pool_metrics is None:
        print(
            "Warning: only_unprocessed is True, but block_pool_metrics.parquet doesn't exist. Returning all pool-block pairs."
        )

    # Count the rows
    result = query.select(pl.len()).collect()

    return result[0, 0]


def set_metrics(blockpool_metric, field: str, prices: list, ordering: list):
    assert field in ["realized", "volume_heur"]
    setattr(blockpool_metric, f"{field}_prices", prices)  # type: ignore
    setattr(blockpool_metric, f"{field}_order", ordering)  # type: ignore

    prices_np = np.array(prices) - blockpool_metric.baseline_price
    setattr(blockpool_metric, f"{field}_l1", norm(prices_np, ord=1))  # type: ignore
    setattr(blockpool_metric, f"{field}_l2", norm(prices_np, ord=2))  # type: ignore
    setattr(blockpool_metric, f"{field}_linf", norm(prices_np, ord=np.inf))  # type: ignore


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
    pool: v3Pool, swaps: pl.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: Dict[str, Any]
):
    # Convert Polars DataFrame to an iterator of named tuples
    swap_iterator = swaps.iter_rows(named=True)

    # Run the realized measurement
    prices, ordering = run_swap_order(pool, swap_iterator, block_number, token_info)

    set_metrics(blockpool_metric, "realized", prices, ordering)


def volume_heuristic(
    pool: v3Pool, swaps: pl.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: Dict[str, Any]
):
    pool_addr = blockpool_metric.pool_address
    baseline_price = blockpool_metric.baseline_price

    # Run the volume heuristic measurement
    curr_price_sqrt = cast(float, pool.getPriceAt(block_number))
    curr_price = get_price(curr_price_sqrt, pool_addr, token_info)

    prices = []
    ordering = []

    # Convert amount0 and amount1 to float
    swaps = swaps.with_columns(
        [
            pl.col("amount0").cast(pl.Float64).alias("amount0_float"),
            pl.col("amount1").cast(pl.Float64).alias("amount1_float"),
        ]
    )

    # Split the swaps into the set of buys and sells
    buy_df = swaps.filter(~pl.col("amount0").str.starts_with("-"))
    sell_df = swaps.filter(pl.col("amount0").str.starts_with("-"))

    # Order buys by volume descending
    buys = buy_df.sort("amount0_float", descending=True).to_dicts() if buy_df.height > 0 else []

    # Order sells by volume descending
    sells = sell_df.sort("amount1_float", descending=True).to_dicts() if sell_df.height > 0 else []

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
        curr_price = get_price(curr_price_sqrt, swap["address"], token_info)
        prices.append(curr_price)
        ordering.append(f"{swap['transaction_index']:03}_{swap['log_index']:03}")

    # Process whatever is left in the tail
    for swap in (buys + sells)[::-1]:
        heur = do_swap(swap, curr_price_sqrt, pool, token_info)

        curr_price_sqrt = heur.sqrtP_next
        prices.append(get_price(curr_price_sqrt, swap["address"], token_info))
        ordering.append(f"{swap['transaction_index']:03}_{swap['log_index']:03}")

    blockpool_metric.volume_heur_prices = prices
    blockpool_metric.volume_heur_order = ordering
    prices_np = np.array(prices) - baseline_price

    blockpool_metric.volume_heur_l1 = float(norm(prices_np, ord=1))
    blockpool_metric.volume_heur_l2 = float(norm(prices_np, ord=2))
    blockpool_metric.volume_heur_linf = float(norm(prices_np, ord=np.inf))


def tstar(
    pool: v3Pool, swaps: pl.DataFrame, block_number: int, blockpool_metric: BlockPoolMetrics, token_info: Dict[str, Any]
):
    # Run the t* measurement if not more than 7 swaps
    if swaps.height > 7:
        return

    # Convert Polars DataFrame to a list of dictionaries
    swaps_list = swaps.to_dicts()

    for swap_perm in permutations(swaps_list):
        prices, _ = run_swap_order(pool, swap_perm, block_number, token_info)
        prices_array = np.array(prices) - blockpool_metric.baseline_price

        blockpool_metric.tstar_l1 = min(blockpool_metric.tstar_l1, norm(prices_array, ord=1))
        blockpool_metric.tstar_l2 = min(blockpool_metric.tstar_l2, norm(prices_array, ord=2))
        blockpool_metric.tstar_linf = min(blockpool_metric.tstar_linf, norm(prices_array, ord=np.inf))


def copy_over(blockpool_metric: BlockPoolMetrics, to: list[str]):
    for field in to:
        setattr(blockpool_metric, f"{field}_prices", blockpool_metric.realized_prices)
        setattr(blockpool_metric, f"{field}_order", blockpool_metric.realized_order)
        setattr(blockpool_metric, f"{field}_l1", blockpool_metric.realized_l1)
        setattr(blockpool_metric, f"{field}_l2", blockpool_metric.realized_l2)
        setattr(blockpool_metric, f"{field}_linf", blockpool_metric.realized_linf)


def get_min_max_block(group: pl.DataFrame) -> tuple[int, int]:
    min_block = group["block_number"].min()
    max_block = group["block_number"].max()

    if min_block is None or max_block is None:
        raise ValueError("Empty group encountered")

    # Convert Decimal to int if necessary
    min_block = int(min_block) if isinstance(min_block, Decimal) else min_block
    max_block = int(max_block) if isinstance(max_block, Decimal) else max_block

    return min_block, max_block  # type: ignore


def run_metrics(
    limit: int,
    offset: int,
    process_id: int,
    token_info: Dict[str, Any],
    mev_boost_values: Dict[int, float],
    only_unprocessed: bool,
):

    def write_buffer():
        nonlocal buffer
        if buffer:
            df = pl.DataFrame(buffer)
            if os.path.exists(output_file):
                existing_df = pl.read_parquet(output_file)
                df = pl.concat([existing_df, df])
            df.write_parquet(output_file)
            buffer = []

    output_file = DATA_PATH / "block_pool_metrics" / f"block_pool_metrics_{offset}-{offset+limit}.parquet"
    pool_block_pairs: pl.DataFrame = get_pool_block_pairs(limit=limit, offset=offset, only_unprocessed=only_unprocessed)

    it = tqdm(total=pool_block_pairs.height, position=process_id, desc=f"[{process_id}] ({offset}-{offset+limit})")
    pool = get_pool(pool_block_pairs.select("address").row(0)[0])

    program_start = datetime.now()

    errors = 0
    successes = 0
    buffer = []

    for pool_addr, group in pool_block_pairs.group_by("address"):
        it.set_description(f"[{process_id}] ({offset}-{offset+limit}) Processing pool {pool_addr}")

        if pool_addr not in token_info:
            continue

        if pool_addr != pool.pool:
            pool = get_pool(pool_addr)

        min_block, max_block = get_min_max_block(group)
        swaps_for_pool = get_swaps_for_address(pool_addr, min_block=min_block, max_block=max_block)

        for block_number in group["block_number"].unique():
            block_number = int(block_number)
            it.set_postfix(errors=errors, successes=successes)
            it.update(1)

            try:
                swaps = swaps_for_pool.filter(pl.col("block_number") == block_number).sort("transaction_index")

                if swaps.height == 0:
                    continue

                curr_price_sqrt = pool.getPriceAt(block_number)

                blockpool_metric = BlockPoolMetrics(
                    block_number=block_number,
                    pool_address=pool_addr,
                    num_transactions=swaps.height,
                    n_buys=swaps.filter(~pl.col("amount0").str.starts_with("-")).height,
                    n_sells=swaps.filter(pl.col("amount0").str.starts_with("-")).height,
                    mev_boost=block_number in mev_boost_values,
                    mev_boost_amount=mev_boost_values.get(block_number, 0),
                    baseline_price=get_price(curr_price_sqrt, pool_addr, token_info),
                )

                realized_measurement(pool, swaps, block_number, blockpool_metric, token_info)

                if swaps.height > 1:
                    volume_heuristic(pool, swaps, block_number, blockpool_metric, token_info)
                    tstar(pool, swaps, block_number, blockpool_metric, token_info)
                else:
                    copy_over(blockpool_metric, to=["volume_heur", "tstar"])

                buffer.append(blockpool_metric.to_dict())

                # Write to Parquet file every 100 rows or at the end
                if len(buffer) >= 100:
                    write_buffer()

                successes += 1

            except Exception as e:
                errors += 1
                with open(f"output/error-{program_start}.log", "a") as f:
                    f.write(f"Error processing block {block_number} for pool {pool_addr}: {e}\n")
                continue

    # Write any remaining rows in the buffer
    write_buffer()
    it.close()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Calculate MEV Boost Data Metrics")
    parser.add_argument("--n-cpus", type=int, default=1, help="Number of CPUs to use")
    args = parser.parse_args()

    only_unprocessed = True
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting MEV Boost Data Metric Calculations with {args.n_cpus} CPUs")

    n_pool_block_pairs = get_pool_block_count(only_unprocessed=only_unprocessed)
    print(f"Processing {n_pool_block_pairs:,} pool-block pairs")

    mev_boost_values = get_mev_boost_values()

    token_info = get_token_info()
    print(f"Loaded {len(token_info):,} token info entries")

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
