var ethers = require("ethers");
const { MongoClient, ServerApiVersion } = require("mongodb");

var url =
  "wss://powerful-quiet-tree.discover.quiknode.pro/a57622ac783ac8de95c780597f3e6b5cc9d609ef/";
const uri =
  "mongodb+srv://mempool:3yJqr9L1OJfus8MJ@uniswap-mempool.glkzum9.mongodb.net/?retryWrites=true&w=majority";

var uniswapV2RouterAddress = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D";
var uniswapV3RouterAddress = "0xE592427A0AEce92De3Edee1F18E0157C05861564";

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri, {
  serverApi: {
    version: ServerApiVersion.v1,
    strict: true,
    deprecationErrors: true,
  },
});

var totalTransactions = 0;
var uniswapV2Transactions = 0;
var uniswapV3Transactions = 0;

var init = async function () {
  console.log("Connecting to MongoDB...");

  await client.connect();

  await client.db("admin").command({ ping: 1 });
  console.log("Pinged your deployment. You successfully connected to MongoDB!");

  // Print initial message
  process.stdout.write(
    `Total Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
  );

  var customWsProvider = new ethers.providers.WebSocketProvider(url);

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
        var collection =
          transaction.to.toLowerCase() === uniswapV2RouterAddress.toLowerCase()
            ? "v2"
            : "v3";

        // Increment transactions for the appropriate collection
        if (collection === "v2") {
          uniswapV2Transactions++;
        } else {
          uniswapV3Transactions++;
        }

        // Add timestamp to transaction
        transaction.timestamp = Date.now();

        // Insert transaction into the appropriate collection
        client
          .db()
          .collection(collection)
          .insertOne(transaction, function (err, res) {
            if (err) throw err;
          });
      }
    });
    // Log total transactions and Uniswap transactions

    // Then, when a new transaction comes in, update the line
    process.stdout.write(
      `\rTotal Transactions: ${totalTransactions} | Uniswap V2 Transactions: ${uniswapV2Transactions} | Uniswap V3 Transactions: ${uniswapV3Transactions}`
    );

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
  });
};

init();
