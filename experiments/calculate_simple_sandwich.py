import sys

current_path = sys.path[0]
sys.path.append(
    current_path[: current_path.find("defi-measurement")]
    + "liquidity-distribution-history"
)

sys.path.append("..")


import os
import math


from dataclasses import dataclass
from datetime import datetime
from typing import Any, Union, cast
from collections import namedtuple

import pandas as pd

from datetime import datetime, timezone

from pool_state import v3Pool
from tqdm import tqdm

from sqlalchemy import create_engine, Column, Integer, String, Double
from sqlalchemy.orm import sessionmaker, declarative_base


from experiments.preload_pool_cache import load_pool_from_blob


# Read in the environment variables
postgres_uri_mp = os.environ["POSTGRESQL_URI_MP"]
postgres_uri_us = os.environ["POSTGRESQL_URI_US"]
azure_storage_uri = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

Base = declarative_base()

engine_mp = create_engine(postgres_uri_mp)
engine_us = create_engine(postgres_uri_us)
SessionLocalMP = sessionmaker(bind=engine_mp)

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


Base.metadata.create_all(engine_mp)


def get_data():
    # ## Get the Data
    query = """
        SELECT *
        FROM SWAP_LIMIT_PRICE AS LIM
        INNER JOIN MEMPOOL_TRANSACTIONS AS MEM ON LIM.transaction_hash = MEM.HASH
    """

    df = pd.read_sql_query(query, engine_mp)

    block_numbers = pd.read_sql_query(
        """
        SELECT block_number, tx_hash, block_ts
        FROM swaps
        WHERE block_number >= 17400000
        ORDER BY block_ts ASC
        """,
        engine_us,
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

    single_swap_blocks = swap_counts[swap_counts == 1].sort_index()

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
    df_single_sorted = df_single_sorted[
        df_single_sorted.transaction_type == "V3_SWAP_EXACT_IN"
    ].drop(
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
    with SessionLocalMP() as session:
        session.add(sandwich)
        session.commit()


def is_processed(user_hash: str) -> bool:
    with SessionLocalMP() as session:
        existing_sandwich = (
            session.query(SimpleSandwich).filter_by(user_hash=user_hash).first()
        )
        return existing_sandwich is not None


def run_sandwiches(swaps: pd.DataFrame):
    curr_pool: v3Pool | None = None
    swap_dicts: Any = swaps.reset_index().to_dict(orient="records")
    it = tqdm(swap_dicts)
    errors = 0

    for swap_dict in it:
        it.set_postfix(errors=errors)

        try:
            swap = SwapData(**swap_dict)
            if is_processed(swap.hash):
                continue

            it.set_description(f"Pool: {swap.pool}, block: {swap.block_number}")

            if not curr_pool or curr_pool.pool != swap.pool:
                curr_pool = load_pool_from_blob(
                    swap.pool,
                    postgres_uri_us,
                    azure_storage_uri,
                    "uniswap-v3-pool-cache",
                    verbose=False,
                    invalidate_before_date=datetime(2023, 8, 16, tzinfo=timezone.utc),
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
                user_hash=swap.hash,
                block_number=swap.block_number,
                pool=swap.pool,
                token_in=swap.token0,
                token_out=swap.token1,
                profit=sandwich_result.profit,
                profit_nofee=profit_nofee,
                profit_float=sandwich_result.profit,
                profit_nofee_float=profit_nofee,
                gas_fee_eth=sandwich_result.gas_fee,
                frontrun_input=sandwich_result.frontrun_input,
                price_baseline=sandwich_result.price_baseline,
                price_frontrun=sandwich_result.price_frontrun,
                price_user=sandwich_result.price_user,
                price_backrun=sandwich_result.price_backrun,
            )

            persist_sandwich(sandwich)

        except Exception as e:
            errors += 1
            with open(f"output/errors-{program_start}.txt", "a") as f:
                f.write(f"{swap_dict}\n{e}\n\n\n")
            continue


if __name__ == "__main__":
    df_single_sorted = get_data()
    run_sandwiches(df_single_sorted)
