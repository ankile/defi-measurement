import requests
import json
from tqdm import trange


query = """
query topPools($skip: Int) {
  pools(
    first: 1000
    skip: $skip
    orderBy: totalValueLockedUSD
    orderDirection: desc
    subgraphError: allow
  ) {
    id
    __typename
    token0 {symbol}
    token1 {symbol}
    totalValueLockedUSD
  }
}
"""


def send_query(query, skip=0) -> requests.Response:
    response = requests.post(
        "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
        json={
            "query": query,
            "variables": {"skip": skip},
        },
    )
    if response.status_code != 200:
        raise Exception(
            f"The Graph query failed: {response.status_code}", response.text
        )

    return response


def get_pool_tokens():
    pool_tokens = {}

    for page in trange(0, 6):
        response = send_query(query, skip=page * 1000)
        pools = response.json()["data"]["pools"]
        for pool in pools:
            pool_tokens[pool["id"]] = [
                pool["token0"]["symbol"],
                pool["token1"]["symbol"],
            ]

    return pool_tokens


def write_list_to_json_file(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    pool_tokens = get_pool_tokens()
    write_list_to_json_file(pool_tokens, "addresses/pool_tokens.json")
