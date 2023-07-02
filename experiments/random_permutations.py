from functools import partial
import os
import pickle
import random
import sys

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

from datetime import datetime
from typing import List, Tuple, cast


from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator
from pool_state import v3Pool
from sqlalchemy import create_engine
from tqdm import tqdm, trange
from multiprocessing import Pool


load_dotenv(override=True)


from decimal import ROUND_DOWN, Decimal, getcontext

getcontext().prec = 100  # Set the precision high enough for our purposes

import pandera as pa
from pandera.typing import DataFrame, Series


class SwapSchema(pa.DataFrameModel):
    block_number: Series[int]
    transaction_index: Series[int]
    log_index: Series[int]
    amount0: Series[str]
    amount1: Series[str]
    sqrtpricex96: Series[str]
    tick: Series[int]
    tx_hash: Series[str]
    block_ts: Series[datetime]


def get_swap_params(pool: v3Pool, swaps: DataFrame[SwapSchema]) -> list:
    # Convert the dataframe into a list of swap parameter dicts
    swaps_parameters = []
    for row in swaps.to_dict(orient="records"):
        # Calculate what is tokenIn and what is tokenOut
        token_in = pool.token0
        amount0 = str(row["amount0"])
        if str(row["amount0"])[0].startswith("-"):
            token_in = pool.token1
            amount0 = str(row["amount1"])

        swapParams = {
            "tokenIn": token_in,
            "input": int(amount0),
            "gasFee": True,
            "as_of": row["block_number"] + 0 / 1e4,
        }

        swaps_parameters.append(swapParams)

    return swaps_parameters


class PoolInfo:
    def __init__(self, pool_address: str, token0_dec: int, token1_dec: int):
        self.pool_address = pool_address
        self.token0_dec = token0_dec
        self.token1_dec = token1_dec

    @staticmethod
    def convert_sqrtPriceX96_to_price(sqrtPriceX96: str, decimals=2) -> float:
        # Assume sqrtPriceX96 is the value you got from the swap event
        # Square it and shift it right by 96 places (rescaling for fixed-point format)
        sqrtPrice = Decimal(sqrtPriceX96) / Decimal(2**96)

        # The result is the price of token1 in terms of token0 (after rounding down)
        raw_price = sqrtPrice ** Decimal(2)

        adj_price = float((raw_price * Decimal(1e6) / Decimal(1e18)))

        price = 1 / adj_price

        return price if decimals is None else round(price, decimals)

    def sqrt_to_price(self, sp: float) -> float:
        return 1 / (sp**2 * self.token0_dec / self.token1_dec)


usdceth30 = PoolInfo(
    pool_address="0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
    token0_dec=6,
    token1_dec=18,
)


# Read in the environment variables
postgres_uri = os.environ["POSTGRESQL_URI_US"]

if postgres_uri is None:
    raise ValueError("Connection string to Postgres is not set")


def swaps_from_pool(pool_address: str, after: int = 0):
    engine = create_engine(postgres_uri)

    df = pd.read_sql(
        f"""
            SELECT block_number, transaction_index,
                log_index, amount0, amount1,
                sqrtpricex96,tick, tx_hash, block_ts
            FROM swaps
            WHERE address = '{pool_address}'
            AND block_number >= {after};
        """,
        engine,
    )
    return df


def plot_simulation(
    results: np.ndarray,
    pool: v3Pool,
    block_num: int,
    swaps_parameters: List[dict],
    save: bool = False,
    show: bool = True,
) -> None:
    # Plot every line on the same plot to get the density
    _, ax = plt.subplots(figsize=(12, 8))

    for i in range(len(results)):
        ax.plot(results[i], color="black", alpha=0.05)

    # Plot the original order in red
    prices = run_simulation(pool, swaps_parameters, pbar=False)
    ax.plot(prices, color="red", label="Original order")

    # Calculate the number of buys and sells
    n_buys = len([s for s in swaps_parameters if s["tokenIn"] == pool.token0])
    n_sells = len([s for s in swaps_parameters if s["tokenIn"] == pool.token1])

    # Make the x-axis tick labels integers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Make the plot nice
    ax.set_title(
        f"Price of ETH in USDC for block {block_num:_} (buys: {n_buys} / sells: {n_sells})",
        fontsize=16,
    )
    ax.set_xlabel("Swap number")
    ax.set_ylabel("Price of ETH in USDC")
    ax.set_xlim(0, len(swaps_parameters))
    # ax.set_ylim(0.9 * results.min(), 1.1 * results.max())
    ax.grid(True)

    # Show the legend containing the number of simulations
    ax.legend([f"{len(results):_} simulations"])

    # Show the plot
    if save:
        plt.savefig(f"output/simulation_{block_num}_{results.shape[0]}.png", dpi=300)

    if show:
        plt.show()


def swap_count_per_block(df: pd.DataFrame, more_than: int = 0) -> List[Tuple[int, int]]:
    swap_counts = (
        df.groupby(by=["block_number"])
        .count()
        .sort_values(by=["transaction_index"], ascending=False)
        .block_ts
    )

    swap_counts = swap_counts[swap_counts > more_than]
    swap_count_list = list(swap_counts.reset_index().values.tolist())

    return cast(List[Tuple[int, int]], swap_count_list)


def get_swap_df_from_block(df, block_number) -> DataFrame[SwapSchema]:
    # Get the swaps for this block
    swaps = df[df.block_number == block_number]
    swaps = swaps.sort_values(by=["transaction_index"])

    # Convert sqrtPriceX96 to price
    swaps["price"] = swaps.sqrtpricex96.apply(usdceth30.convert_sqrtPriceX96_to_price)

    # Add column indicating sell or buy
    swaps["direction"] = swaps.apply(
        lambda x: "sell" if int(x.amount0) < 0 else "buy", axis=1
    )

    return swaps


def run_simulation(pool: v3Pool, swaps_parameters: list, pbar=True) -> np.ndarray:
    # Get the sqrtPriceX96 at the start of the block
    sqrtPrice_next = pool.getPriceAt(swaps_parameters[0]["as_of"])

    prices = np.zeros(len(swaps_parameters), dtype=np.float64)

    for i, s in tqdm(enumerate(swaps_parameters), disable=not pbar):
        s["givenPrice"] = sqrtPrice_next
        _, heur = pool.swapIn(s, fees=True)
        sqrtPrice_next = heur.sqrtP_next
        prices[i] = 1 / (heur.sqrt_P**2 / 1e12)

    return prices


def get_swap_params_random(swaps_parameters: list) -> list:
    swaps_parameters = swaps_parameters.copy()
    random.shuffle(swaps_parameters)
    return swaps_parameters


def run_simulation_batch(i, *, pool, swaps_parameters, batch_size):
    results = []
    for _ in trange(batch_size, desc=f"Batch {i}", leave=True, position=i):
        swaps_parameters_random = get_swap_params_random(swaps_parameters)
        prices_random = run_simulation(pool, swaps_parameters_random, pbar=False)
        results.append(prices_random)
    return results


def n_random_permutation(
    pool: v3Pool,
    swaps_parameters: list,
    n_simulations: int = 5,
    cores: int = -1,
) -> np.ndarray:
    # Initialize liquidity in the pool
    pool.createLiq()

    print(f"Running {n_simulations:_} simulations with {cores} cores")

    # Calculate batch size
    batch_size = ceil(n_simulations / cores)

    sim = partial(
        run_simulation_batch,
        pool=pool,
        swaps_parameters=swaps_parameters,
        batch_size=batch_size,
    )

    with Pool(processes=cores) as p:
        results = list(
            p.map(sim, range(cores)),
        )

    # Flatten the results list
    results = [price for batch in results for price in batch]

    return np.array(results)


def load_pool(
    pool_address: str,
    postgres_uri: str,
) -> v3Pool:
    # Check if pool_cache.pickle exists
    filename = "cache/pool_cache.pickle"
    if not os.path.exists(filename):
        os.mkdir("cache")
        with open(filename, "wb") as f:
            pickle.dump({}, f)

    # Check if we already have this pool in the cache
    with open(filename, "rb") as f:
        pool_cache = pickle.load(f)
        if pool_address in pool_cache:
            # If it is, load the pool from the cache
            print("Loading pool from cache")
            return pool_cache[pool_address]

    # If it's not in the cache, load it and add it to the cache
    pool = v3Pool(
        poolAdd=pool_address,
        connStr=postgres_uri,
        initialize=False,
        delete_conn=True,
    )

    # Add the pool to the cache
    pool_cache[pool_address] = pool

    # Save the cache
    with open(filename, "wb") as f:
        pickle.dump(pool_cache, f)

    return pool


def main(
    n_blocks: int = 1,
    n_simulations: int = 1000,
    cores: int = -1,
    save: bool = True,
    show: bool = True,
):
    # Create the pool
    print("Loading pool")
    pool = load_pool(
        pool_address=usdceth30.pool_address,
        postgres_uri=postgres_uri,
    )

    # Get all swaps for this pool
    print("Loading swaps")
    df = swaps_from_pool(pool.pool, after=int(15e6))

    # Get the swap counts
    print("Calculating swap counts")
    swap_counts = swap_count_per_block(df, more_than=4)

    # Get the swaps for the block with the most swaps
    print("Getting swaps for block with most swaps")

    for i in trange(n_blocks):
        block_num = swap_counts[i][0]

        swap_df = get_swap_df_from_block(df, block_num)

        # Get the swap parameters
        print("Getting swap parameters")
        swaps_parameters = get_swap_params(pool, swap_df)

        # Run the simulation
        print("Running simulation")
        if cores == -1:
            cores = os.cpu_count() or 1

        results = n_random_permutation(
            pool, swaps_parameters, n_simulations=n_simulations, cores=cores
        )

        # Plot the simulation
        print("Plotting simulation")
        plot_simulation(
            results, pool, block_num, swaps_parameters, save=save, show=show
        )


if __name__ == "__main__":
    main(
        n_blocks=10,
        n_simulations=1000,
        cores=-1,
        save=True,
        show=False,
    )
