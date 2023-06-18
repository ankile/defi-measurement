require("dotenv").config();

const ethers = require("ethers");
const { MongoClient } = require("mongodb");

const nodeWSConnectionString = process.env.NODE_WS_CONNECTION_STRING;
const mongodbConnectionString = process.env.MONGODB_CONNECTION_STRING;

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(mongodbConnectionString);

let totalTransactions = 0;
let transactionBuffer = [];

const init = async function () {
  console.log("Connecting to MongoDB...");

  await client.connect();

  console.log("Connected to MongoDB!");

  const mempoolHashes = client.db("transactions").collection("mempool-hashes");

  // Print initial message
  process.stdout.write(`Total Transactions: ${totalTransactions}`);

  const customWsProvider = new ethers.providers.WebSocketProvider(
    nodeWSConnectionString
  );

  customWsProvider._websocket.on("error", async (error) => {
    console.log(`Unable to connect to node! Attempting reconnect in 3s...`);
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
    // Increment the totalTransactions counter
    totalTransactions++;

    // Add the transaction hash to the transactionBuffer with timestamp
    transactionBuffer.push({
      hash: tx,
      timestamp: Date.now(),
    });

    // Insert all transaction hashes in the buffer into one document in the mempoolHashes collection
    if (transactionBuffer.length === 1000) {
      mempoolHashes.insertOne({ hashes: transactionBuffer });
      transactionBuffer = [];
    }

    // Then, when a new transaction comes in, update the line
    process.stdout.write(`\rTotal Transactions: ${totalTransactions}`);
  });
};

init();
