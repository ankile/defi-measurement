require("dotenv").config();

var ethers = require("ethers");
const { MongoClient } = require("mongodb");

const quiknodeApiKey = process.env.QUIKNODE_API_KEY;
const mongodbConnectionString = process.env.MONGODB_CONNECTION_STRING;

var url = `wss://powerful-quiet-tree.discover.quiknode.pro/${quiknodeApiKey}/`;
const uri = mongodbConnectionString;

const uniswapV2RouterAddress = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D";
const uniswapV3RouterAddress = "0xE592427A0AEce92De3Edee1F18E0157C05861564";

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri);

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
  console.log("Connecting to MongoDB...");

  await client.connect();

  console.log("Connected to MongoDB!");

  const mempool = client.db("transactions").collection("mempool");

  // Print initial message
  process.stdout.write(
    `Total Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
  );

  var customWsProvider = new ethers.providers.WebSocketProvider(url);

  customWsProvider._websocket.on("error", async () => {
    console.log(`Unable to connect to ${ep.subdomain} retrying in 3s...`);
    setTimeout(init, 3000);
  });
  customWsProvider._websocket.on("close", async (code) => {
    console.log(
      `Connection lost with code ${code}! Attempting reconnect in 3s...`
    );
    customWsProvider._websocket.terminate();
    setTimeout(init, 3000);
  });

  customWsProvider.on("pending", (tx) => {
    customWsProvider.getTransaction(tx).then(function (transaction) {
      // Increment total transactions
      totalTransactions++;
      if (
        transaction &&
        transaction.to &&
        (transaction.to.toLowerCase() ===
          uniswapV2RouterAddress.toLowerCase() ||
          transaction.to.toLowerCase() === uniswapV3RouterAddress.toLowerCase())
      ) {
        // Determine collection based on contract address
        let uniswapVersion =
          transaction.to.toLowerCase() === uniswapV2RouterAddress.toLowerCase()
            ? "v2"
            : "v3";

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
      }
    });
    // Log total transactions and Uniswap transactions

    // Then, when a new transaction comes in, update the line
    process.stdout.write(
      `\rTotal Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
    );
  });
};

init();
