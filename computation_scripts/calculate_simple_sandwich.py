import argparse
import sys

import numpy as np

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")


import os
import math

from multiprocessing import Pool, cpu_count

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union, cast
from collections import namedtuple

import pandas as pd

from datetime import datetime, timezone

from pool_state import v3Pool
from tqdm import tqdm

from sqlalchemy import create_engine, Column, Integer, String, Double, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base


from computation_scripts.preload_pool_cache import load_pool_from_blob

# Create the `output` directory if it doesn't exist
if not os.path.exists("output"):
    os.makedirs("output")


# Read in the environment variables
postgres_uri = os.environ["POSTGRESQL_URI"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

Base = declarative_base()

engine = create_engine(postgres_uri)
SessionLocal = sessionmaker(bind=engine)

program_start = datetime.now()


class SimpleSandwich(Base):
    __tablename__ = "simple_sandwiches"

    user_hash = Column(String, primary_key=True, nullable=False)
    block_number = Column(Integer, nullable=False)
    pool = Column(String, nullable=False)
    token_in = Column(String, nullable=False)
    token_out = Column(String, nullable=False)
    profit = Column(String, nullable=False)
    profit_nofee = Column(String, nullable=False)
    profit_float = Column(Double, nullable=False)
    profit_nofee_float = Column(Double, nullable=False)
    gas_fee_eth = Column(Double, nullable=False)
    frontrun_input = Column(String, nullable=False)
    price_baseline = Column(Double, nullable=False)
    price_frontrun = Column(Double, nullable=False)
    price_user = Column(Double, nullable=False)
    price_backrun = Column(Double, nullable=False)
    profit_percent = Column(Double)
    frontrun_input_float = Column(Double, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    user_input_float = Column(Double)
    profit_per_user_input = Column(Double)
    profit_usd = Column(Double)


Base.metadata.create_all(engine)


def get_data(unprocessed_only=False):
    # ## Get the Data
    query = """
        SELECT *
        FROM SWAP_LIMIT_PRICE AS LIM
        INNER JOIN MEMPOOL_TRANSACTIONS AS MEM ON LIM.transaction_hash = MEM.HASH
        WHERE LIM.transaction_type = 'V3_SWAP_EXACT_IN'
    """

    if unprocessed_only:
        query += """
            AND NOT EXISTS (
                SELECT 1
                FROM simple_sandwiches AS SS
                WHERE SS.user_hash = MEM.HASH
            )
        """

    df = pd.read_sql_query(query, engine)

    block_numbers = pd.read_sql_query(
        """
        SELECT block_number, tx_hash, block_ts
        FROM swaps
        WHERE block_number >= 17400000
        ORDER BY block_ts ASC
        """,
        engine,
    ).set_index("tx_hash")

    block_number_dict = block_numbers[
        ~block_numbers.index.duplicated(keep="first")
    ].to_dict(orient="index")

    dataset = df.assign(
        block_number=df.transaction_hash.map(
            lambda x: block_number_dict[x]["block_number"]
            if x in block_number_dict
            else None
        )
    )
    dataset = dataset[~dataset.block_number.isna()]

    swap_counts = (
        dataset.groupby(by=["pool", "block_number"])[["transaction_hash"]]
        .count()
        .sort_values("transaction_hash", ascending=False)
    )

    single_swap_blocks = swap_counts[swap_counts.transaction_hash == 1].sort_index()

    df_single = dataset.set_index(["pool", "block_number"]).loc[
        single_swap_blocks.index
    ]

    # Group by level 0 and count unique values in level 1
    grouped_counts = df_single.groupby(level=0).apply(
        lambda x: x.index.get_level_values(1).nunique()
    )

    # Sort the indices based on the counts
    sorted_indices = grouped_counts.sort_values(ascending=False).index

    # Reindex the DataFrame based on this sorted order
    df_single_sorted = df_single.loc[sorted_indices]

    # Keep only the swap in subset for now
    df_single_sorted = df_single_sorted.drop(
        columns=[
            "transaction_type",
            "recipient",
            "amountOut",
            "amountInMax",
            "payerIsUser",
            "transaction_hash",
        ]
    )

    return df_single_sorted


def to_price(inv_sqrt_price):
    return 1 / (inv_sqrt_price) ** 2 * 1e12


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
        return False


def exponential_search(
    pool: v3Pool, swap: SwapData, start=1e6, max_tries=100, factor=8, verbose=False
) -> tuple[float, float]:
    lower: float = 0.0
    upper = start

    for i in range(max_tries):
        if verbose:
            print(f"{i}: Trying {upper}")

        if not valid_frontrun(pool, swap, upper):
            return lower, upper
        else:
            lower = upper
            upper *= factor

    else:
        raise Exception("Exponential search exceeded max tries")


def binary_search(pool: v3Pool, swap: SwapData, lower, upper, max_tries=100, verbose=False) -> int:
    for i in range(max_tries):
        if verbose:
            print(f"{i}: Trying {lower} - {upper}")

        if math.isclose(lower, upper, rel_tol=1e-5):
            return lower

        mid = (lower + upper) // 2
        if valid_frontrun(pool, swap, mid):
            lower = mid
        else:
            upper = mid
    else:
        raise Exception("Binary search exceeded max tries")


def max_frontrun(
    pool: v3Pool, swap: SwapData, start=1e6, factor=2, verbose=False
) -> int:
    lower, upper = exponential_search(pool, swap, start=start, factor=factor, verbose=verbose)
    return binary_search(pool, swap, lower, upper, verbose=verbose)


def single_sandwich_mev(
    pool: v3Pool, swap: SwapData, frontrun_input: int, pool_fee=True
) -> tuple[int, float, tuple]:
    total_gas = 0
    orig_fee = pool.fee
    if not pool_fee:
        pool.fee = 0

    price_baseline = pool.getPriceAt(swap.block_number)

    # Frontrun swap
    swap_params = {
        "input": frontrun_input,
        "tokenIn": swap.token0,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": price_baseline,
    }

    output_frontrun, heur = pool.swapIn(swap_params)
    total_gas += heur.gas_fee
    price_frontrun = heur.sqrtP_next

    # User swap
    swap_params = {
        "input": int(swap.amountIn),
        "tokenIn": swap.token0,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": price_frontrun,
    }
    output, heur = pool.swapIn(swap_params)
    price_user = heur.sqrtP_next

    # Backrun swap
    swap_params = {
        "input": output_frontrun,
        "tokenIn": swap.token1,
        "as_of": swap.block_number,
        "gasFee": True,
        "givenPrice": heur.sqrtP_next,
    }
    output, heur = pool.swapIn(swap_params)
    total_gas += heur.gas_fee
    price_backrun = heur.sqrtP_next

    if not pool_fee:
        pool.fee = orig_fee

    return (
        output - frontrun_input,
        total_gas,
        (price_baseline, price_frontrun, price_user, price_backrun),
    )


AutoSandwichResult = namedtuple(
    "AutoSandwichResult",
    [
        "profit",
        "gas_fee",
        "frontrun_input",
        "price_baseline",
        "price_frontrun",
        "price_user",
        "price_backrun",
    ],
)


def auto_sandwich_mev(
    pool: v3Pool, swap: SwapData, start=1e18, factor=2, pool_fee=True, verbose=False
) -> AutoSandwichResult:
    frontrun_input = max_frontrun(pool, swap, start=start, factor=factor, verbose=verbose)
    profit, gas_fee, prices = single_sandwich_mev(
        pool, swap, frontrun_input, pool_fee=pool_fee
    )
    price_baseline, price_frontrun, price_user, price_backrun = prices

    return AutoSandwichResult(
        profit=profit,
        gas_fee=gas_fee,
        frontrun_input=frontrun_input,
        price_baseline=price_baseline,
        price_frontrun=price_frontrun,
        price_user=price_user,
        price_backrun=price_backrun,
    )


def persist_sandwich(sandwich: SimpleSandwich):
    with SessionLocal() as session:
        session.add(sandwich)
        session.commit()


def is_processed(user_hash: str) -> bool:
    with SessionLocal() as session:
        existing_sandwich = (
            session.query(SimpleSandwich).filter_by(user_hash=user_hash).first()
        )
        return existing_sandwich is not None


def run_sandwiches(swaps: pd.DataFrame, position=0):
    curr_pool: v3Pool | None = None
    swap_dicts: Any = swaps.reset_index().to_dict(orient="records")
    it = tqdm(swap_dicts, position=position)
    errors = 0
    skipped = 0

    for swap_dict in it:
        it.set_postfix(errors=errors, skipped=skipped)

        try:
            swap = SwapData(**swap_dict)
            if is_processed(swap.hash):
                skipped += 1
                continue

            it.set_description(f"Pool: {swap.pool}, block: {swap.block_number}")

            if not curr_pool or curr_pool.pool != swap.pool:
                curr_pool = load_pool_from_blob(
                    swap.pool,
                    postgres_uri,
                    azure_storage_uri,
                    "uniswap-v3-pool-cache",
                    verbose=False,
                    invalidate_before_date=datetime(2023, 8, 18, tzinfo=timezone.utc),
                    pbar=it,
                )
            if not curr_pool:
                raise Exception("Pool undefined")

            sandwich_result = auto_sandwich_mev(
                curr_pool,
                swap,
                start=1e16,
                factor=8,
                pool_fee=True,
            )
            profit_nofee, _, _ = single_sandwich_mev(
                curr_pool,
                swap,
                sandwich_result.frontrun_input,
                pool_fee=False,
            )

            sandwich = SimpleSandwich(
                # Basic info about the swap
                user_hash=swap.hash,
                block_number=swap.block_number,
                pool=swap.pool,
                token_in=swap.token0,
                token_out=swap.token1,
                # Profit measures
                profit=sandwich_result.profit,
                profit_nofee=profit_nofee,
                profit_float=sandwich_result.profit,
                profit_nofee_float=profit_nofee,
                gas_fee_eth=sandwich_result.gas_fee,
                # Price measures for price trajectory plotting
                price_baseline=sandwich_result.price_baseline,
                price_frontrun=sandwich_result.price_frontrun,
                price_user=sandwich_result.price_user,
                price_backrun=sandwich_result.price_backrun,
                # Input volume measures
                frontrun_input=sandwich_result.frontrun_input,
                frontrun_input_float=float(sandwich_result.frontrun_input),
                user_input_float=float(swap.amountIn),
                # Relative profit measures
                profit_percent=         sandwich_result.profit / float(sandwich_result.frontrun_input) if float(sandwich_result.frontrun_input) > 0 else 0,
                profit_per_user_input=  sandwich_result.profit / float(swap.amountIn) if float(swap.amountIn) > 0 else 0,
            )

            persist_sandwich(sandwich)

        except Exception as e:
            errors += 1
            with open(f"output/errors-{program_start}.txt", "a") as f:
                f.write(f"{swap_dict}\n{e}\n\n\n")
            continue

def run_sandwich_parallel(args):
    df, position = args
    run_sandwiches(df, position=position)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--offset', type=int, default=0, help='Offset for the dataframe')
    parser.add_argument('--limit', type=int, default=None, help='Limit for the dataframe')
    parser.add_argument('--n-cpus', type=int, default=None, help='Get help partitioning the calculations')

    args = parser.parse_args()
    
    df = get_data(unprocessed_only=True)
    print(f"Loaded a total of {len(df)} swaps")

    if args.n_cpus is not None:
        # Run the script in parallel
        n_cpus = args.n_cpus if args.n_cpus > 0 else cpu_count()
        print(f"Running in parallel with {n_cpus} cpus")
        chunks = np.array_split(df, n_cpus) # type: ignore
        args_list = [(chunk, i) for i, chunk in enumerate(chunks)]
    
        with Pool(n_cpus) as p:
            p.map(run_sandwich_parallel, args_list) # type: ignore
        
    else:
        # Run the script in serial
        # Apply offset and limit to the dataframe
        df_sliced: pd.DataFrame = df.iloc[args.offset:args.offset+args.limit if args.limit is not None else None, :] # type: ignore

        run_sandwiches(df_sliced)
