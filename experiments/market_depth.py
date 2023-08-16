import pandas as pd
import numpy as np
import sys

import sys
sys.path.append("/Users/austin/code/liquidity-distribution-history/")
from pool_state import v3Pool
from swap_utils import *

def pct_change_function(huer):
    return (huer[4]**2 - huer[3]**2) / huer[3]**2

def verbose_print(string, verbose):
    if verbose:
        print(string)
        
def swapDF_to_max(token):
    assert token in [pool.token0, pool.token1], 'Token not part of pool'

    if pool.token == token:
        guessHigh = pool.swapDF['xInTick'].cumsum().max()
    else:
        guessHigh = pool.swapDF['yInTick'].cumsum().max()

    return guessHigh

def binary_search(guessPrev, guessHigh, token, block, 
                  target_change, heurisitic_function, 
                  top_depth = 50, verbose = True):
    depth = 0
    search = True
    
    while search:
        guessMid = np.floor((guessPrev + guessHigh) / 2)
        
        swapParams = {'tokenIn': token,
                    'input': guessMid,
                    'as_of': block}
        
        out, huer = pool.swapIn(swapParams)
        pct_change = heurisitic_function(huer)
        
        if np.isclose(abs(pct_change), abs(target_change), atol = .00015, rtol = 0):
            search = False
            verbose_print(f"Found price at {guessMid} with target {target_change}", verbose)
            return guessMid
        elif abs(pct_change) < abs(target_change):
            verbose_print(f"Search low - New grid of {guessMid} to {guessHigh}", verbose)
            if guessPrev == guessMid:
                return guessMid
            guessPrev, guessHigh = guessMid, guessHigh

        elif abs(pct_change) > abs(target_change):
            verbose_print(f"Search high - New grid of {guessPrev} to {guessMid}", verbose)
            if guessHigh == guessMid:
                return guessMid
            guessPrev, guessHigh = guessPrev, guessMid
        
        depth+=1
        if depth > top_depth:
            raise ValueError("Reached maximum depth")
           
        
    
def find_guess(guess, at_block, token, top_depth, heurisitic_function, guess_growth = 1.05, verbose = True):
    end = True
    guessPrev = 0
    guessHigh = guess
    
    while end:
        swapIn = guessHigh
        swapParams = {'tokenIn': token,
                    'input': swapIn,
                    'as_of': at_block, }

        try:
            out, huer = pool.swapIn(swapParams)
        except AssertionError as e:
            max_allowable = pool.swapIn(swapParams, find_max = True)
            verbose_print(f"Pool is illiquid - returning max of {max_allowable}", True)
            return 0, max_allowable
        
        pct_change = heurisitic_function(huer)

        if abs(pct_change) <= abs(top_depth):
            guessPrev = guessHigh
            guessHigh*=guess_growth
            verbose_print(f'Doubling guess from {guessPrev} to {guessHigh}', verbose)
        else:
            verbose_print(f'Found {pct_change} with {guessPrev} to {guessHigh}', verbose)
            end = False
            
    return guessPrev, guessHigh

# ----------

addr = '0x95dbb3c7546f22bce375900abfdd64a4e5bd73d6'
pool = v3Pool(addr)#, update = True)
swaps = pool.swaps

# ----------

guess_growth = 1.25
top_depth = .02


depths = [.001, .005, .01, .02, .05]
max_depth = max(depths)
data = []
start_of_sample = pd.to_datetime('09-01-2022 00:00:00')
date_range = pd.date_range(start_of_sample, 
                           pool.mb['block_ts'].max().floor("1d"), freq = '15t')

for block_ts in date_range:
    at_block = pool.getBlockAtTS(block_ts)
    guessLow = 1 * 1e6
    try:
        ####
        # --- Search High
        ####
        guessPrev, guessHigh = find_guess(guessLow, at_block, pool.token1, max_depth, pct_change_function, 
                                          guess_growth=guess_growth, verbose = False)

        for depth in depths:
            if guessHigh != 0:
                amt1 = binary_search(0, guessHigh, pool.token1, 
                          at_block, depth, pct_change_function, verbose = False)
            else:
                amt1 = 0
                
            data.append([pool.token1, depth, amt1, at_block, block_ts, 1])

        ####
        # --- Search Low
        ####
        guessLow = 1 * 1e6
        guessPrev, guessHigh = find_guess(guessLow, at_block, pool.token0, max_depth, pct_change_function,
                                          guess_growth=guess_growth, verbose = False)
        for depth in depths:
            if guessHigh != 0:
                amt0 = binary_search(0, guessHigh, pool.token0, 
                      at_block, depth, pct_change_function, verbose = False)
            else:
                amt0 = 0
                
            data.append([pool.token0, depth, amt0, at_block, block_ts, pool.getPriceAt(at_block) ** 2])
    
    except AssertionError as e:
        print(f"Dead pool at {block_ts}")
        continue


df = pd.DataFrame(data, columns = ['tokens', 'depth', 'amount',
                              'block', 'timestamp', 'usd_value'])
df[df['timestamp'] <= swaps['block_ts'].max()].to_csv("euroc_md.csv")