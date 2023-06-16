# MEV Measurement in DEX


## Data

### Swaps Table


| Tx Hash | Block | Block # | Pool Addr | Token0 | Token1 | Token0In | Token1In | sqrtPriceX96Limit |
|:-------:|:-----:|:-------:|:---------:|:------:|:------:|:--------:|:--------:|:-----------------:|
|   ✅    |  ✅   |   ✅    |    ✅     |   ✅   |   ✅   |    ✅    |    ✅    |        ✅         |


### Blocks Table


| Block # | Initial Liquidity | Initial sqrtPriceX96 | Liquidity Positions |
|:-------:|:-----------------:|:--------------------:|:-------------------:|
|   ✅    |        ✅         |          ✅          |         ❌ (WIP)         |


### Potential Data Sources

- Running a Ethereum full node using geth and interacting with it using web3.py
- Using Infura.io's full node and API and interact with it using web3.py
- Use thegraph.com's subgraphs and GrapQL API for Uniswap data, preferably hosted


### Data Storage and Querying



## Simulating Counterfactual Transaction Sequence

