import numpy as np

import sys

import os
from pool_state import v3Pool
import numpy as np
import matplotlib.pyplot as plt
import random

from typing import TypeVar
from datetime import datetime

import json

import pandas as pd
from tqdm import tqdm, trange

from sqlalchemy import create_engine

from dotenv import load_dotenv

load_dotenv(override=True)

from enum import Enum

from decimal import Decimal, getcontext, ROUND_DOWN

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
postgres_uri = os.getenv("POSTGRESQL_URI_US")

if postgres_uri is None:
    raise ValueError("Connection string to Postgres is not set")


def add_path():
    current_path = sys.path[0]
    sys.path.append(
        current_path[: current_path.find("defi-measurement")]
        + "liquidity-distribution-history"
    )


def swaps_from_pool(pool_address: str, after: int = 0):
    engine = create_engine(postgres_uri)

    df = pd.read_sql(
        f"""
            SELECT block_number, transaction_index,
                log_index, amount0, amount1,
                sqrtpricex96,tick, tx_hash, block_ts,
            FROM swaps
            WHERE address = '{pool_address}'
            AND block_number >= {after};
        """,
        engine,
    )
    return df


def swap_count_per_block(df: pd.DataFrame, more_than: int = 0) -> pd.Series[int]:
    swap_counts = (
        df.groupby(by=["block_number"])
        .count()
        .sort_values(by=["transaction_index"], ascending=False)
        .block_ts
    )

    return swap_counts[swap_counts > more_than]


def get_swap_df_from_block(df, block_number):
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
