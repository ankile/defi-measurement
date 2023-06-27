import { ethers } from "ethers";
import { Provider } from "@ethersproject/providers";
import fs from "fs";
import { Interface, LogDescription } from "@ethersproject/abi";

const provider = new ethers.providers.JsonRpcProvider("http://localhost:8545");

async function v3Swaps(
  txHash: string,
  uniswapV3PoolAbi: any,
): Promise<LogDescription[]> {
  const txReceipt = await provider.getTransactionReceipt(txHash);

  let liquidityEvents: LogDescription[] = [];

  for (let log of txReceipt.logs) {
    const contractInterface = new Interface(uniswapV3PoolAbi);

    try {
      const eventData = contractInterface.parseLog(log);
      if (eventData.name === "Swap") {
        liquidityEvents.push(eventData);
      }
    } catch {
      continue;
    }
  }

  return liquidityEvents;
}

let start = args.start;
let end = args.start + args.steps;
let step = args.steps > 0 ? 1 : -1;

let swapsToInsert: Swap[] = [];

for (let blockNumber = start; blockNumber != end; blockNumber += step) {
  console.log(`Processing block ${blockNumber}`);

  let block = await provider.getBlock(blockNumber);

  if (block.transactions.length === 0) {
    fs.appendFileSync(
      "./errors.txt",
      `Block ${blockNumber} has no transactions\n`,
    );
    continue;
  }

  let blockTimestamp = new Date(block.timestamp * 1000);

  for (let transactionHash of block.transactions) {
    let swaps = await v3Swaps(transactionHash, uniswapV3PoolAbi);

    for (let swap of swaps) {
      // Check if this transaction is in the mempool database from mongo
      let fromMempool = await mempool.findOne({ hash: transactionHash });

      let swapToInsert = new Swap(
        transactionHash,
        blockTimestamp,
        blockNumber,
        swap.logIndex,
        swap.args.sender,
        swap.args.recipient,
        swap.args.amount0.toString(),
        swap.args.amount1.toString(),
        swap.args.sqrtPriceX96.toString(),
        swap.args.liquidity.toString(),
        swap.args.tick.toString(),
        swap.address,
        swap.args.sender,
        swap.args.recipient,
        swap.transactionIndex,
        fromMempool,
      );

      swapsToInsert.push(swapToInsert);
      console.log({ swaps_to_insert: swapsToInsert.length });
    }

    if (swapsToInsert.length > 100) {
      // Insert the swaps into the database
      const session = new Session();
      for (let swap of swapsToInsert) {
        await session.merge(swap);
      }

      await session.commit();
      swapsToInsert = [];
    }
  }
}

// Insert the remaining swaps into the database
const session = new Session();
for (let swap of swapsToInsert) {
  await session.merge(swap);
}

await session.commit();
