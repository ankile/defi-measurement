import json

from web3 import Web3
import pandas as pd

from datetime import datetime

import matplotlib.pyplot as plt
import bisect

# Import python dotenv
from dotenv import load_dotenv
import numpy as np
import os

from operator import itemgetter

from tqdm import tqdm, trange

# Import sqlalchemy
from sqlalchemy import create_engine

from sqlalchemy import (
    MetaData,
    Table,
    String,
    Column,
    Text,
    DateTime,
    Boolean,
    Integer,
    BigInteger,
    Float,
    ForeignKey,
    Numeric,
)
from sqlalchemy.orm.mapper import Mapper
from datetime import datetime

from pymongo import MongoClient, UpdateOne, DESCENDING, ASCENDING, InsertOne, DeleteOne


load_dotenv(override=True)

w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

postgres_uri = os.getenv("POSTGRESQL_URI")

assert postgres_uri is not None, "POSTGRESQL_URI is not set in .env file"

engine = create_engine(postgres_uri)

# Connect to mongodb
client = MongoClient(os.getenv("MONGO_URI"))

# Get database
mempool = client.transactions.mempool

mempool.estimated_document_count()

metadata = MetaData()

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Swap(Base):
    __tablename__ = "swaps"

    # id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    transaction_hash = Column(String, index=True)
    block_timestamp = Column(DateTime, nullable=False)
    block_number = Column(Integer, nullable=False, index=True, primary_key=True)
    transaction_index = Column(Integer, nullable=False, primary_key=True)
    log_index = Column(Integer, nullable=False, primary_key=True)
    sender = Column(String, nullable=False, index=True)
    recipient = Column(String, nullable=False, index=True)
    amount0 = Column(String, nullable=False)
    amount1 = Column(String, nullable=False)
    sqrtPriceX96 = Column(String, nullable=False)
    liquidity = Column(String, nullable=False)
    tick = Column(String, nullable=False)
    address = Column(String, nullable=False, index=True)
    from_address = Column(String, nullable=False, index=True)
    to_address = Column(String, nullable=False, index=True)
    from_mempool = Column(Boolean)


# Create a Session class bound to this engine
Session = sessionmaker(bind=engine)

with open("abi/UniswapV3Pool.json", "r") as f:
    uniswap_v3_pool_abi = json.load(f)


from datetime import datetime
from pydantic import BaseModel


class SwapArgs(BaseModel):
    sender: str
    recipient: str
    amount0: int
    amount1: int
    sqrtPriceX96: int
    liquidity: int
    tick: int


class SwapData(BaseModel):
    args: SwapArgs
    blockNumber: int
    event: str
    logIndex: int
    transactionIndex: int
    address: str
    blockNumber: int


# Get start and stop block from the command line
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--start", type=int, help="start block")
parser.add_argument("--steps", type=int, help="steps to take")

args = parser.parse_args()

swaps_to_insert = []
swaps_inserted = 0


def v3_swaps(tx_hash):
    # Get transaction receipt
    tx_receipt = w3.eth.get_transaction_receipt(tx_hash)

    liquidity_events = []

    for log in tx_receipt["logs"]:
        contract = w3.eth.contract(abi=uniswap_v3_pool_abi, address=log["address"])

        # Parse the logs for Swap, Mint, and Burn events
        try:
            event_data = contract.events.Swap().process_log(log)
        except:
            continue

        liquidity_events.append(event_data)

    return liquidity_events


it = trange(args.start, args.start + args.steps, 1 if args.steps > 0 else -1)
for block_number in it:
    it.set_description(f"Processing block {block_number:,}")

    block = w3.eth.get_block(block_number)

    if "transactions" not in block:
        with open(f"./errors.txt", "a") as f:
            f.writelines(f"Block {block_number} has no transactions\n")
        continue

    block_timestamp = datetime.fromtimestamp(
        block["timestamp"] if "timestamp" in block else 0
    )

    for transaction in block["transactions"]:
        # Get transaction hash
        tx_hash = w3.to_hex(transaction)  # type: ignore

        # Get swaps from transaction
        swaps = v3_swaps(tx_hash)

        for swap in swaps:
            swap = SwapData(**swap)

            # Check if this transaction is in the mempool database from mongo
            from_mempool = bool(mempool.find_one({"hash": tx_hash}))

            swap_to_insert = Swap(
                transaction_hash=tx_hash,
                block_timestamp=block_timestamp,
                block_number=block_number,
                log_index=swap.logIndex,
                sender=swap.args.sender,
                recipient=swap.args.recipient,
                amount0=str(swap.args.amount0),
                amount1=str(swap.args.amount1),
                sqrtPriceX96=str(swap.args.sqrtPriceX96),
                liquidity=str(swap.args.liquidity),
                tick=str(swap.args.tick),
                address=swap.address,
                from_address=swap.args.sender,
                to_address=swap.args.recipient,
                transaction_index=swap.transactionIndex,
                from_mempool=from_mempool,
            )

            swaps_to_insert.append(swap_to_insert)
            it.set_postfix({"swaps_to_insert": len(swaps_to_insert)})

    # Checkpoint if we get more than 100 swaps
    if len(swaps_to_insert) > 100:
        # Insert the swaps into the database
        with Session() as session:
            for swap in swaps_to_insert:
                session.merge(swap)

            session.commit()

        swaps_to_insert = []

# Insert the swaps into the database
with Session() as session:
    for swap in swaps_to_insert:
        session.merge(swap)

    session.commit()
