require("dotenv").config();

const ethers = require("ethers");
const { MongoClient } = require("mongodb");

const readline = require("readline");

let rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function printTransactionCounts() {
  readline.cursorTo(rl, 0, 0);
  readline.clearScreenDown(rl);
  rl.write(
    `Total Transactions: ${totalTransactions}\n` +
      `Total Uniswap Transactions: ${uniswapTransactionCount} (${(
        (uniswapTransactionCount / totalTransactions) *
        100
      ).toFixed(2)}%)\n` +
      `${Object.entries(transactionCounts)
        .map(
          ([routerContract, count]) =>
            `${routerContract}: ${count} (${(
              (count / uniswapTransactionCount) *
              100
            ).toFixed(2)}%)`
        )
        .join("\n")}`
  );
}

const nodeWSConnectionString = process.env.NODE_WS_CONNECTION_STRING;
const mongodbConnectionString = process.env.MONGODB_CONNECTION_STRING;

const BATCH_SIZE = 10;
const DB = "transactions";
const COLLECTION = "mempool";

// Read the map of contract addresses to contract names from the router_addresses.json file, looks like this:
/*
{
  '0xf164fC0Ec4E93095b804a4795bBe1e041497b92a': 'UniswapV2Router01',
  '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D': 'UniswapV2Router02',
  '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'UniswapV3Router01',
  '0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45': 'UniswapV3Router02',
  '0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B': 'UniversalRouter01',
  '0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD': 'UniversalRouter02'
}
*/
const uniswapRouterAddresses = require("./router_addresses.json");

// Make a copy of the map where all the keys are lowercase
const uniswapRouterAddressesLowercase = Object.fromEntries(
  Object.entries(uniswapRouterAddresses).map(([k, v]) => [k.toLowerCase(), v])
);

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(mongodbConnectionString);

function convertHexWeiToGwei(hex) {
  // Convert hex to a BigNumber instance
  let bn = ethers.BigNumber.from(hex._hex);

  // Convert the BigNumber to a float
  let floatValue = parseFloat(ethers.utils.formatUnits(bn, "gwei"));

  // Return the float
  return floatValue;
}

let totalTransactions = 0;
let uniswapTransactionCount = 0;

// Create object to track transaction count for each of the Uniswap router versions in the map
let transactionCounts = Object.fromEntries(
  Object.values(uniswapRouterAddresses).map((val) => [val, 0])
);

const init = async function () {
  await connectDb();

  const mempool = client.db(DB).collection(COLLECTION);

  const customWsProvider = new ethers.providers.WebSocketProvider(
    nodeWSConnectionString
  );

  connectErrorHandlers();

  transactionBatch = [];

  customWsProvider.on("pending", (tx) => {
    customWsProvider.getTransaction(tx).then(function (transaction) {
      // Increment total transactions
      totalTransactions++;
      if (!transaction || !transaction.to) {
        return;
      }

      // Determine collection based on contract address
      let routerContract =
        uniswapRouterAddressesLowercase[transaction.to.toLowerCase()];

      // If the transaction is not a Uniswap transaction, return
      if (!routerContract) {
        return;
      }

      // Increment transactions for the appropriate collection
      transactionCounts[routerContract]++;
      uniswapTransactionCount++;

      // Set id to transaction hash
      transaction._id = transaction.hash;

      // Add Uniswap version to transaction
      transaction.routerContract = routerContract;

      // Add timestamp to transaction
      transaction.timestamp = Date.now();

      let date = new Date(transaction.timestamp);

      transaction.formattedDate = date.toISOString();

      // Convert gasPrice, maxPriorityFeePerGas, and maxFeePerGas to Gwei
      transaction.gasPriceRaw = transaction.gasPrice;
      transaction.gasPrice = convertHexWeiToGwei(transaction.gasPrice);

      // Convert gasLimit to a number
      transaction.gasLimitRaw = transaction.gasLimit;
      transaction.gasLimit = parseInt(transaction.gasLimit);

      // Add boolean to indicate if to address was checksummed
      transaction.toChecksummed =
        transaction.to === transaction.to.toLowerCase();

      transaction.totalTransactionsSeen = totalTransactions;
      transaction.uniswapTransactionCount = uniswapTransactionCount;
      transaction.uniswapShareOfTotal =
        uniswapTransactionCount / totalTransactions;

      transactionBatch.push(transaction);

      // If the batch size has been reached, insert the batch into the database
      if (transactionBatch.length >= BATCH_SIZE) {
        // Upsert the batch into the database
        let operations = transactionBatch.map((transaction) => ({
          updateOne: {
            filter: { _id: transaction._id },
            update: { $set: transaction },
            upsert: true,
          },
        }));

        mempool.bulkWrite(operations);

        // Clear the transaction batch
        transactionBatch = [];

        // Print transaction counts
        printTransactionCounts();
      }
    });
  });

  async function connectDb() {
    readline.cursorTo(rl, 0, 0);
    readline.clearScreenDown(rl);
    rl.write("Connecting to MongoDB...");
    await client.connect();
    rl.write(" Connected to MongoDB!\n");
  }

  function connectErrorHandlers() {
    customWsProvider._websocket.on("error", async (error) => {
      console.log(
        `Unable to connect to node with error ${error}! Attempting reconnect in 3s...`
      );
      setTimeout(init, 3000);
    });
    customWsProvider._websocket.on("close", async (code) => {
      console.log(
        `Connection lost with code ${code}! Attempting reconnect in 3s...`
      );
      customWsProvider._websocket.terminate();
      setTimeout(init, 3000);
    });
  }
};

init();
