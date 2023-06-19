require("dotenv").config();

const ethers = require("ethers");
const { MongoClient } = require("mongodb");

const nodeWSConnectionString = process.env.NODE_WS_CONNECTION_STRING;
const mongodbConnectionString = process.env.MONGODB_CONNECTION_STRING;

let uniswapRouterAddresses = new Map([
  ["0xf164fC0Ec4E93095b804a4795bBe1e041497b92a", "v2"], // UniswapV2Router01
  ["0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D", "v2"], // UniswapV2Router02
  ["0xE592427A0AEce92De3Edee1F18E0157C05861564", "v3"], // SwapRouter (UniswapV3)
  ["0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45", "v3"], // SwapRouter02 (UniswapV3)
  ["0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B", "v3"], // UniversalRouter (UniswapV3)
  ["0x3fC91A3afd70395Cd496C647d5a6CC9D4B2b7FAD", "v3"], // UniversalRouterV1_2 (UniswapV3)
]);

// Make all strings in the map lowercase
uniswapRouterAddresses = new Map(
  [...uniswapRouterAddresses].map(([k, v]) => [k.toLowerCase(), v])
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
let uniswapV2Transactions = 0;
let uniswapV3Transactions = 0;

const init = async function () {
  await connectDb();

  const mempool = client.db("transactions").collection("mempool");

  // Print initial message
  process.stdout.write(
    `Total Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
  );

  const customWsProvider = new ethers.providers.WebSocketProvider(
    nodeWSConnectionString
  );

  connectErrorHandlers();

  customWsProvider.on("pending", (tx) => {
    customWsProvider.getTransaction(tx).then(function (transaction) {
      // Increment total transactions
      totalTransactions++;
      if (!transaction || !transaction.to) {
        return;
      }

      // Determine collection based on contract address
      let uniswapVersion = uniswapRouterAddresses.get(
        transaction.to.toLowerCase()
      );

      // If the transaction is not a Uniswap transaction, return
      if (!uniswapVersion) {
        return;
      }

      // Increment transactions for the appropriate collection
      if (uniswapVersion === "v2") {
        uniswapV2Transactions++;
      } else {
        uniswapV3Transactions++;
      }

      // Add Uniswap version to transaction
      transaction.uniswapVersion = uniswapVersion;

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

      // Insert transaction into the appropriate collection
      mempool.insertOne(transaction, function (err, res) {
        if (err) throw err;
      });
    });
    // Log total transactions and Uniswap transactions

    // Then, when a new transaction comes in, update the line
    process.stdout.write(
      `\rTotal Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
    );
  });

  async function connectDb() {
    process.stdout.write("Connecting to MongoDB...");
    await client.connect();
    process.stdout.write("\rConnected to MongoDB!");
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
