import { PrismaClient, MempoolTransaction } from "@prisma/client";
import { ethers } from "ethers";
import { config } from "dotenv";

config();

const ethereum_node_uri = process.env.NODE_WS_CONNECTION_STRING;

if (!ethereum_node_uri) {
  throw new Error("Missing ethereum_node_uri");
}

const prisma = new PrismaClient();
const web3 = new ethers.providers.WebSocketProvider(ethereum_node_uri);

const transactionBatchSize = 100;
let transactionBatch: MempoolTransaction[] = [];

web3.on("pending", async (txHash: string) => {
  // Get current time
  const now = new Date();
  transactionBatch.push({
    hash: txHash,
    firstSeen: now,
  });

  if (transactionBatch.length >= transactionBatchSize) {
    await prisma.mempoolTransaction.createMany({
      data: transactionBatch,
      skipDuplicates: true,
    });
    transactionBatch = [];
    console.log("Added to the table", now);
  }
});
