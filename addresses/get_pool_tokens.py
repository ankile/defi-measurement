import requests
import json


query = """
query topPools {
  pools(
    first: 500
    orderBy: totalValueLockedUSD
    orderDirection: desc
    subgraphError: allow
  ) {
    id
    __typename
    token0 {symbol}
    token1 {symbol}
  }
}
"""


def get_pool_tokens():
    response = requests.post(
        "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        json={"query": query},
    )

    if response.status_code != 200:
        raise Exception(
            f"The Graph query failed: {response.status_code}", response.text
        )

    # Map the response to a list of dictionaries pool address -> token symbols
    pools = response.json()["data"]["pools"]
    pool_tokens = {}
    for pool in pools:
        pool_tokens[pool["id"]] = [pool["token0"]["symbol"], pool["token1"]["symbol"]]

    return pool_tokens


def write_list_to_json_file(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    pool_tokens = get_pool_tokens()
    write_list_to_json_file(pool_tokens, "addresses/pool_tokens.json")
