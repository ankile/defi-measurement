require("dotenv").config();

const ethers = require("ethers");
const { MongoClient } = require("mongodb");

const readline = require("readline");
const nodemailer = require("nodemailer");

const { printTransactionCounts } = require("./printTransactionCounts");

// Read the map of contract addresses to contract names from the router_addresses.json file, looks like this:
const uniswapRouterAddresses = require("./router_addresses.json");

const nodeWSConnectionString = process.env.NODE_WS_CONNECTION_STRING;
const mongodbConnectionString = process.env.MONGODB_CONNECTION_STRING;

const BATCH_SIZE = 10;
const DB = "transactions";
const COLLECTION = "mempool";

console.log("Setting up email transporter...");
const transporter = nodemailer.createTransport({
  service: "gmail", // or another email service that supports SMTP
  host: "smtp.gmail.com",
  port: 465,
  secure: true,
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASS,
  },
});

console.l;

async function sendEmail(subject, text) {
  let info = await transporter.sendMail({
    from: process.env.EMAIL_USER,
    to: process.env.EMAIL_TO,
    subject: subject,
    text: text,
  });

  return info;
}

let rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});
exports.rl = rl;

// Make a copy of the map where all the keys are lowercase
const uniswapRouterAddressesLowercase = Object.fromEntries(
  Object.entries(uniswapRouterAddresses).map(([k, v]) => [k.toLowerCase(), v]),
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
let transactionBatch = [];
let lastDatabaseWrite = new Date();

// Create object to track transaction count for each of the Uniswap router versions in the map
let transactionCounts = Object.fromEntries(
  Object.values(uniswapRouterAddresses).map((val) => [val, 0]),
);

const init = async function () {
  await sendEmail("Mempool started", "Mempool started");
  await connectDb();

  const mempool = client.db(DB).collection(COLLECTION);

  const customWsProvider = new ethers.providers.WebSocketProvider(
    nodeWSConnectionString,
  );

  connectErrorHandlers();

  customWsProvider.on("pending", (tx) => {
    customWsProvider.getTransaction(tx).then(async function (transaction) {
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
      const now = new Date();
      transaction.ts = now;
      transaction.timestamp = now.getTime();

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

      // Save to database every second
      if (lastDatabaseWrite.getTime() + 1000 < now.getTime()) {
        // Upsert the batch into the database
        let operations = transactionBatch.map((transaction) => ({
          updateOne: {
            filter: { _id: transaction._id },
            update: { $set: transaction },
            upsert: true,
          },
        }));

        try {
          await mempool.bulkWrite(operations);
        } catch (err) {
          const errMsg = `Failed to write to MongoDB with error: ${err}. Retrying connection...`;
          console.log(errMsg);
          await sendEmail("MongoDB write error", errMsg);
          await client.close(); // Close the possibly broken connection
          await connectDb(); // Try to reconnect
        }

        // Clear the transaction batch
        transactionBatch = [];

        // Print transaction counts
        printTransactionCounts(
          rl,
          totalTransactions,
          uniswapTransactionCount,
          transactionCounts,
        );
      }
    });
  });

  async function connectDb() {
    readline.cursorTo(rl, 0, 0);
    readline.clearScreenDown(rl);
    rl.write("Connecting to MongoDB...");
    try {
      await client.connect();
      rl.write(" Connected to MongoDB!\n");
    } catch (err) {
      const errMsg = `Failed to connect to MongoDB with error: ${err}. Retrying in 3s...`;
      console.log(errMsg);
      await sendEmail("MongoDB connection error", errMsg);
      setTimeout(connectDb, 1000);
    }
  }

  function connectErrorHandlers() {
    customWsProvider._websocket.on("error", async (error) => {
      console.log(
        `Unable to connect to node with error ${error}! Attempting reconnect in 3s...`,
      );
      setTimeout(init, 1000);
    });
    customWsProvider._websocket.on("close", async (code) => {
      console.log(
        `Connection lost with code ${code}! Attempting reconnect in 3s...`,
      );
      customWsProvider._websocket.terminate();
      setTimeout(init, 1000);
    });
  }
};

// Have a catch all email alert for any uncaught errors
process.on("uncaughtException", async (err) => {
  const errMsg = `Uncaught exception: ${err}. Restarting...`;
  console.log(errMsg);
  await sendEmail("Uncaught exception", errMsg);

  // Restart the process
  process.exit(1);
});

init();
