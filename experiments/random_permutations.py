import argparse
from functools import partial
import json
import os
import pickle
import random
import sys

from pydantic import BaseModel

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

from datetime import datetime
from typing import Tuple, cast
from prisma import Client
import asyncio


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
from azure.storage.blob import BlobServiceClient


load_dotenv(override=True)


from decimal import ROUND_DOWN, Decimal, getcontext

getcontext().prec = 100  # Set the precision high enough for our purposes

import pandera as pa
from pandera.typing import DataFrame, Series

# Check if the cache folder exists
if not os.path.exists("cache"):
    os.mkdir("cache")

# Check if the output folder exists
if not os.path.exists("output"):
    os.mkdir("output")

# Check that the errors.csv file exists
if not os.path.exists("output/errors.csv"):
    with open("output/errors.csv", "w") as f:
        f.write("block_number,pool_address,error\n")


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


# Read in the environment variables
postgres_uri = os.environ["POSTGRESQL_URI_US"]
blobstorage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]


def swaps_df(after: int = 0):
    # Check if the swaps_df is cached
    filename = "cache/swaps_df.pickle"
    if not os.path.exists(filename):
        print("Loading swaps from database")
        engine = create_engine(postgres_uri)

        df = pd.read_sql(
            f"""
                SELECT block_number, address, transaction_index,
                    log_index, amount0, amount1,
                    sqrtpricex96, tick, tx_hash, block_ts
                FROM swaps
                WHERE block_number > {after};
            """,
            engine,
        )

        engine.dispose()

        with open(filename, "wb") as f:
            pickle.dump(df, f)

    else:
        print("Loading swaps from cache")
        with open(filename, "rb") as f:
            df = pickle.load(f)

    return df


def get_pool_info_df():
    engine = create_engine(postgres_uri)

    df = pd.read_sql(
        f"""
            SELECT pool, decimals0, decimals1, token0, token1, token0symbol, token1symbol, fee
            FROM token_info
            WHERE token0symbol IS NOT NULL
            AND token1symbol IS NOT NULL
            AND decimals0 IS NOT NULL
            AND decimals1 IS NOT NULL;
        """,
        engine,
    )

    engine.dispose()
    return df


def plot_simulation(
    permutation_prices: np.ndarray,
    orignal_prices: np.ndarray,
    pool: v3Pool,
    block_num: int,
    swaps_parameters: list[dict],
    filename: str,
    save: bool = False,
    show: bool = True,
):
    # Plot every line on the same plot to get the density
    _, ax = plt.subplots(figsize=(12, 8))

    for i in range(len(permutation_prices)):
        ax.plot(permutation_prices[i], color="black", alpha=0.05)

    # Plot the original order in red
    ax.plot(orignal_prices, color="red", label="Original order")

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
    # ax.set_ylim(0.9 * permutation_prices.min(), 1.1 * permutation_prices.max())
    ax.grid(True)

    # Show the legend containing the number of simulations
    ax.legend([f"{len(permutation_prices):_} simulations"])

    # Show the plot
    if save:
        plt.savefig(filename, dpi=300)

    if show:
        print(f"Showing plot for block {block_num:_}")
        plt.show()

    plt.close()


def swap_count_per_block(
    df: pd.DataFrame, more_than: int = 0
) -> list[Tuple[int, str, int]]:
    swap_counts = (
        df.groupby(by=["block_number", "address"])
        .count()
        .sort_values(by=["transaction_index"], ascending=False)
        .block_ts
    )

    swap_counts = swap_counts[swap_counts > more_than]
    swap_count_list = list(swap_counts.reset_index().values.tolist())

    return cast(list[Tuple[int, str, int]], swap_count_list)


def get_swap_df_from_block(
    df, block_number, pool_info: PoolInfo
) -> DataFrame[SwapSchema]:
    # Get the swaps for this block
    swaps = df[df.block_number == block_number]
    swaps = swaps.sort_values(by=["transaction_index"])

    # Convert sqrtPriceX96 to price
    swaps["price"] = swaps.sqrtpricex96.apply(pool_info.convert_sqrtPriceX96_to_price)

    # Add column indicating sell or buy
    swaps["direction"] = swaps.apply(
        lambda x: "sell" if int(x.amount0) < 0 else "buy", axis=1
    )

    return swaps


def run_simulation(pool: v3Pool, swaps_parameters: list, pbar=True) -> np.ndarray:
    # Get the sqrtPriceX96 at the start of the block
    sqrtPrice_next = pool.getPriceAt(swaps_parameters[0]["as_of"])

    prices = np.zeros(len(swaps_parameters) + 1, dtype=np.float64)

    for i, s in tqdm(enumerate(swaps_parameters, start=0), disable=not pbar):
        s["givenPrice"] = sqrtPrice_next
        _, heur = pool.swapIn(s, fees=True)
        sqrtPrice_next = heur.sqrtP_next
        prices[i] = 1 / (heur.sqrt_P**2 / 1e12)

    # Calculate the price at the end of the block
    prices[-1] = 1 / (sqrtPrice_next**2 / 1e12)

    return prices


def run_simulation_batch(i, *, pool, swaps_parameters, batch_size):
    results = []
    swaps_parameters = swaps_parameters.copy()
    for _ in trange(batch_size, desc=f"Batch {i}", leave=False, position=i):
        random.shuffle(swaps_parameters)
        prices_random = run_simulation(pool, swaps_parameters, pbar=False)
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
    verbose: bool = True,
) -> v3Pool:
    os.makedirs('cache/pool_cache/', exist_ok=True)

    # Check if we already have this pool in the cache
    filename = f"cache/pool_cache/{pool_address}.pickle"
    if os.path.exists(filename):
        if verbose:
            print("Loading pool from cache")
        with open(filename, "rb") as f:
            pool = pickle.load(f)
            return pool

    # If it's not in the cache, load it and add it to the cache
    if verbose:
        print("Loading pool from database")
    pool = v3Pool(
        poolAdd=pool_address,
        connStr=postgres_uri,
        initialize=False,
        delete_conn=True,
    )

    # Save the cache
    with open(filename, "wb") as f:
        pickle.dump(pool, f)

    return pool


class Stats(BaseModel):
    baseline: float
    original_std: float
    permutation_stds: list
    mean_permutation_std: float
    permutation_abs_deviations: list
    max_abs_permutation_deviations: list
    mean_max_abs_permutation_deviation: float
    abs_original_deviation: list
    max_abs_original_deviation: float
    permutation_areas: list
    mean_permutation_area: float
    original_area: float


def calculate_stats(
    original_prices: np.ndarray, permutation_prices: np.ndarray
) -> Stats:
    baseline = original_prices[0]
    permutation_abs_deviations = np.abs(permutation_prices - baseline)
    abs_original_deviation = np.abs(original_prices - baseline)

    stats = Stats(
        baseline=baseline.item(),
        original_std=np.std(original_prices).item(),
        permutation_stds=list(np.std(permutation_prices, axis=1)),
        mean_permutation_std=np.mean(np.std(permutation_prices, axis=1)).item(),
        permutation_abs_deviations=list(permutation_abs_deviations.tolist()),
        max_abs_permutation_deviations=list(np.max(permutation_abs_deviations, axis=1)),
        mean_max_abs_permutation_deviation=np.mean(
            np.max(permutation_abs_deviations, axis=1)
        ).item(),
        abs_original_deviation=list(abs_original_deviation),
        max_abs_original_deviation=np.max(abs_original_deviation).item(),
        permutation_areas=list(np.trapz(permutation_abs_deviations, axis=1)),  # type: ignore
        mean_permutation_area=np.mean(np.trapz(permutation_abs_deviations, axis=1)).item(),  # type: ignore
        original_area=np.trapz(abs_original_deviation).item(),  # type: ignore
    )

    return stats


def save_to_storage(data_filename, figure_filename) -> Tuple[str, str]:
    # Set up the Azure blob service client
    blob_service_client = BlobServiceClient.from_connection_string(blobstorage_uri)

    # Create a new container (if it does not exist)
    container_name = "uniswap-pool-pickles"
    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
        print(f"Container '{container_name}' created.")
    except:
        print(f"Container '{container_name}' already exists.")

    folder_name = data_filename.split("/")[-1].split(".")[0]

    # Set up the blob client for the parquet file
    data_client = blob_service_client.get_blob_client(
        container_name, f"permutation-simulation/{folder_name}/data.json"
    )

    # Upload the parquet file to the blob
    with open(data_filename, "rb") as data:
        data_client.upload_blob(data, overwrite=True)

    # Set up the blob client for the figure
    figure_client = blob_service_client.get_blob_client(
        container_name, f"permutation-simulation/{folder_name}/figure.png"
    )

    # Upload the figure to the blob
    with open(figure_filename, "rb") as data:
        figure_client.upload_blob(data, overwrite=True)

    print(f"Uploaded {data_filename} and {figure_filename} to blob storage")

    return data_client.url, figure_client.url


def save_data(data: dict[str, list | dict], filename: str) -> str:
    with open(filename, "w") as f:
        json.dump(data, f)

    return filename


async def simulation_exists(
    prisma: Client, block_num: int, pool_address: str, n_simulations: int
) -> bool:
    record = await prisma.permutationsimulation.find_first(
        where={
            "block_number": block_num,
            "pool_address": pool_address,
            "n_permutations": n_simulations,
        }
    )
    if record:
        reason = f"Skipping block {block_num} for pool {pool_address} and n_permutations {n_simulations} as it already exists"
        print(reason)
        return True

    return False


async def main(
    n_blocks: int = 1,
    offset: int = 0,
    n_simulations: int = 1000,
    more_than: int = 4,
    random_order: bool = False,
    cores: int = -1,
    save: bool = True,
    show: bool = True,
):
    print(f"Running with argumens: {locals()}")

    # Create Prisma client if we're saving to the database
    prisma = Client()
    await prisma.connect()

    # Get the pool info
    print("Loading pool info")
    pools_info_df = get_pool_info_df()

    # Get the swap data
    print("Loading swap data")
    df = swaps_df(after=int(15e6))

    # Get the swap counts
    print("Calculating swap counts")
    swap_counts = swap_count_per_block(df, more_than=more_than)

    print(f"Found {len(swap_counts)} blocks with more than {more_than} swaps")

    # Shuffle the swap counts if we want to run them in random order
    if random_order:
        print("Shuffling swap counts")
        random.shuffle(swap_counts)

    n_successful = 0

    it = trange(offset, offset + n_blocks, leave=True, unit="block")
    for i in it:
        block_num, pool_address, _ = swap_counts[i]
        it.set_description(f"Block {block_num}, pool {pool_address[-6:]}")

        # Check if the simulation already exists
        if await simulation_exists(prisma, block_num, pool_address, n_simulations):
            continue

        # Create the pool
        print(f"[{datetime.now()}] Loading pool {pool_address}")
        try:
            pool = load_pool(
                pool_address=pool_address,
                postgres_uri=postgres_uri,
            )
        except AssertionError as e:
            print(f"Skipping block {block_num} for pool {pool_address} as {e}")
            with open("errors.csv", "a") as f:
                f.write(f"{block_num},{pool_address},{e}\n")

            continue

        pool_info_df = pools_info_df[pools_info_df.pool == pool.pool]

        if pool_info_df.empty:
            reason = f"Skipping block {block_num} for pool {pool.pool} and n_permutations {n_simulations} as we don't have the pool info"
            print(reason)
            with open("errors.csv", "a") as f:
                f.write(f"{block_num},{pool_address},{reason}\n")
            continue

        pool_info = PoolInfo(
            pool_address=pool.pool,
            token0_dec=pool_info_df.decimals0.iloc[0].item(),
            token1_dec=pool_info_df.decimals1.iloc[0].item(),
        )

        # Get all swaps for this pool
        print("Filtering swaps for this pool")
        pool_df = df[df.address == pool.pool]

        # Get the swap parameters
        swap_df = get_swap_df_from_block(pool_df, block_num, pool_info)
        print("Getting swap parameters")
        swaps_parameters = get_swap_params(pool, swap_df)

        print(
            f"Running block {block_num} for pool {pool.pool} with {len(swaps_parameters)} swaps and n_permutations {n_simulations}"
        )

        # Run the simulation
        print("Running simulation")
        if cores == -1:
            cores = os.cpu_count() or 1

        permutation_prices = n_random_permutation(
            pool, swaps_parameters, n_simulations=n_simulations, cores=cores
        )
        original_prices = run_simulation(pool, swaps_parameters, pbar=False)

        # Create filename
        filename_stem = f"output/simulation_{block_num}_{permutation_prices.shape[0]}"
        data_filename = filename_stem + ".json"
        figure_filename = filename_stem + ".png"

        # Plot the simulation
        print("Plotting simulation")
        plot_simulation(
            permutation_prices,
            original_prices,
            pool,
            block_num,
            swaps_parameters,
            figure_filename,
            save=save,
            show=show,
        )

        stats = calculate_stats(original_prices, permutation_prices)

        data = {
            "original_prices": list(original_prices.tolist()),
            "permutation_prices": list(permutation_prices.tolist()),
            "stats": stats.dict(),
        }

        # Save the parquet file and figure to blob storage
        if save:
            # Save the parquet file to file
            print("Saving data to local file")
            save_data(data, data_filename)

            print("Saving data to blob storage")
            data_url, figure_url = save_to_storage(data_filename, figure_filename)

            # Save the simulation to the database
            print("Saving simulation to database")
            await prisma.permutationsimulation.create(
                data={
                    "block_number": block_num,
                    "pool_address": pool.pool,
                    "data_location": data_url,
                    "figure_location": figure_url,
                    "ts": datetime.now(),
                    "n_permutations": permutation_prices.shape[0],
                    "n_swaps": permutation_prices.shape[1],
                    "original_std": stats.original_std,
                    "mean_permutation_std": stats.mean_permutation_std,
                    "original_area": stats.original_area,
                    "mean_permutation_area": stats.mean_permutation_area,
                    "max_abs_original_deviation": stats.max_abs_original_deviation,
                    "mean_max_abs_permutation_deviation": stats.mean_max_abs_permutation_deviation,
                }
            )

        n_successful += 1
        it.set_postfix(successful=n_successful)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_blocks",
        "-b",
        type=int,
        default=1,
        help="Number of blocks to simulate",
    )
    parser.add_argument(
        "--offset",
        "-o",
        type=int,
        default=0,
        help="Offset from the block with the most swaps",
    )
    parser.add_argument(
        "--n_simulations",
        "-s",
        type=int,
        default=1000,
        help="Number of simulations to run",
    )
    parser.add_argument(
        "--more-than",
        "-m",
        type=int,
        default=4,
        help="Minimum number of swaps per block",
    )
    parser.add_argument(
        "--random-order",
        "-r",
        type=str2bool,
        default=False,
        help="Run the simulations in random order",
    )
    parser.add_argument(
        "--cores",
        "-c",
        type=int,
        default=-1,
        help="Number of cores to use",
    )
    parser.add_argument(
        "--save",
        type=str2bool,
        default=True,
        help="Save the plot",
    )
    parser.add_argument(
        "--show",
        type=str2bool,
        default=False,
        help="Show the plot",
    )
    parser.add_argument(
        "--clear-caches",
        nargs="+",
        help="Caches to clear",
    )

    args = parser.parse_args()

    if args.clear_caches:
        resp = input(f"Are you sure you want to clear {args.clear_caches}? (y/n) ")
        if resp.lower() != "y":
            sys.exit(0)
        for cache in args.clear_caches:
            filename = f"cache/{cache}.pickle"
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed {filename}")

    asyncio.run(
        main(
            n_blocks=args.n_blocks,
            offset=args.offset,
            more_than=args.more_than,
            random_order=args.random_order,
            n_simulations=args.n_simulations,
            cores=args.cores,
            save=args.save,
            show=args.show,
        )
    )
