import time
from web3 import Web3

print("This script will find the first block mined after Uniswap V3's release.")

# Connect to Ethereum node
web3 = Web3(
    Web3.HTTPProvider("https://mainnet.infura.io/v3/db0355f9beed4af58889a043d591db91")
)

print(f"Connected to Ethereum node {web3.client_version}")

# Get the latest block number
latest_block = web3.eth.block_number


# List of Uniswap V3 contract addresses
uniswap_v3_addresses = [
    # "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
    # "0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b",
    # "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    # Add more addresses here
]

# Start from the latest block and go backwards for 100 blocks
for block_num in range(latest_block - 1000, latest_block, 1):
    block = web3.eth.get_block(block_num, full_transactions=True)

    # Loop through each transaction in the block
    for tx in block["transactions"]:
        # Check if the "to" address is a Uniswap V3 contract
        if tx["to"] in uniswap_v3_addresses:
            print(f"Found a Uniswap V3 transaction in block {block_num}: {tx['hash']}")
