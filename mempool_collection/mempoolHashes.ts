import { PrismaClient, MempoolTransaction } from "@prisma/client";
import { ethers } from "ethers";
import { config } from "dotenv";
import nodemailer from "nodemailer";


config();

const ethereum_node_uri = process.env.NODE_WS_CONNECTION_STRING;

if (!ethereum_node_uri) {
  throw new Error("Missing ethereum_node_uri");
}

const prisma = new PrismaClient();
const web3 = new ethers.providers.WebSocketProvider(ethereum_node_uri);

const transactionBatchSize = 1000;
let transactionBatch: MempoolTransaction[] = [];
const transactionQueue: MempoolTransaction[] = [];
let totalTransactions = 0;


async function writeToDb() {
  try {
    await prisma.mempoolTransaction.createMany({
      data: transactionBatch,
      skipDuplicates: true,
    });
    totalTransactions += transactionBatch.length;
    transactionBatch = [];
    process.stdout.write(`\rTotal Transactions: ${totalTransactions}, last addition ${new Date()}`);
  } catch (err: any) {
    console.error(`[${new Date()}] Failed to write to the database, retrying in 5 seconds...`, err);
    await sendEmail("Database connection failure", "Failed to write to the database. Error: " + err.message);
    setTimeout(writeToDb, 5000);
  }
}


web3.on("pending", (txHash: string) => {
  // Get current time
  const now = new Date();
  transactionQueue.push({
    hash: txHash,
    firstSeen: now,
  });
});

// Periodically process the queue
setInterval(async () => {
  while (transactionQueue.length > 0 && transactionBatch.length < transactionBatchSize) {
    transactionBatch.push(transactionQueue.shift()!);
  }

  if (transactionBatch.length >= transactionBatchSize) {
    await writeToDb();
  }
}, 1000); // Adjust the interval as needed


process.on("unhandledRejection", async (reason, promise) => {
  console.log(`[${new Date()}] Unhandled Rejection at:`, promise, 'reason:', reason);
  await sendEmail("Memmpool hash collection failure (other)", "Unhandled Rejection at: " + promise + " reason: " + reason);
});


async function sendEmail(subject: string, text: string) {
  const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER,
      pass: process.env.EMAIL_PASS
    }
  });

  const mailOptions = {
    from: process.env.EMAIL_USER, 
    to: process.env.EMAIL_TO,
    subject: subject,
    text: text
  };

  try {
    let info = await transporter.sendMail(mailOptions);
    console.log(`[${new Date()}] Message sent: ${info.response}`);
  } catch (error: any ) {
    console.log(`[${new Date()}] Error occurred while sending email: ${error.message}`);
  }
}
