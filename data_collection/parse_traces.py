import pandas as pd
from eth_abi import abi
import eth_utils
import pickle as pkl
import os
import time

search_commands = ['00', '01']
command_mapping = {"00": "V3_SWAP_EXACT_IN", 
                   "01": "V3_SWAP_EXACT_OUT"}
'''
select *
from uniswap_ethereum.UniversalRouter_call_execute
limit 1000
'''
df = pd.read_csv("data.csv")

if not os.path.exists("clean"):
    os.mkdir("clean")

# vectorized clean the data
df['split_command'] = df['commands'].str.rstrip().str.split("").apply(lambda x: [_x for _x in x if _x != ""])
df['clean_commands'] = df['split_command'].apply(lambda x: [x[i] + x[i+1] for i in range(0, len(x), 2)])
df['clean_commands'] = df['clean_commands'].apply(lambda x: [_x for _x in x if _x != "0x"]).copy()
df['command_location'] = df['clean_commands'].apply(lambda x: [count for (count, row) in enumerate(x) 
                                                               if row in search_commands])

df['clean_inputs'] = df['inputs'].apply(lambda x: [_x.replace("[", "").replace("]", "") for _x in x.split(" ")])

v3_swaps = df[df['command_location'].apply(lambda x: len(x) != 0)].copy()


cleaned_data = []
for i in range(v3_swaps.shape[0]):
    swap = v3_swaps.iloc[i]
    swap_payload = []

    for command_location in swap['command_location']:
        
        command = swap['clean_commands'][command_location]
        input_str = swap['clean_inputs'][command_location]
        input_str = eth_utils.decode_hex(input_str)

        decodedABI = abi.decode(['address', 'uint256', 'uint256', 'bytes', 'bool'], input_str)

        path = decodedABI[3]

        payload = []
        offset = 43

        addr1 = eth_utils.to_hex(path[0:20])
        fee = eth_utils.to_int(path[20:23])
        addr2 = eth_utils.to_hex(path[23:43])
        payload.append([addr1, fee, addr2])

        while offset < len(path):
            # consume the path
            addr1 = addr2
            fee = eth_utils.to_int(path[offset: offset+3])
            addr2 = eth_utils.to_hex(path[offset+3: offset+23])

            payload.append([addr1, fee, addr2])

            offset+=23

        if len(path) < 43:
            # this can only happen if bug (i think)
            raise ValueError("Incorrect Path Specification")
        
        command_str = command_mapping[command]
        cleaned_input = {"recipient": decodedABI[0],
                    f"{'amountIn' if command == '00' else 'amountOut'}": decodedABI[1],
                    f"{'amountOutMin' if command == '00' else 'amountInMax'}": decodedABI[2],
                    "path": payload,
                    "payerIsUser": decodedABI[4]}
        
        swap_payload.append([command_str, cleaned_input, swap['call_tx_hash']])
    cleaned_data.append(swap_payload)

with open(f"clean/data_{int(round(time.time()), 0)}", 'wb') as f:
    pkl.dump(cleaned_data, f)