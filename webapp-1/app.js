const express = require("express");
const app = express();
const server = require("http").createServer(app);
const io = require("socket.io")(server);
const dotenv = require("dotenv");
const { MongoClient } = require("mongodb");

dotenv.config();

app.set("view engine", "ejs");

app.get("/", (req, res) => {
  console.log("Client connected");
  res.render("index", { count: 0 });
});

// Replace the placeholder with your Atlas connection string
const uri = process.env.mongodbConnStr;

// Create a MongoClient with a MongoClientOptions object to set the Stable API version
const client = new MongoClient(uri);

let db;

console.log("Connecting to MongoDB...");
client.connect().then(() => {
  console.log("Connected to MongoDB");
  db = client.db("transactions");

  setInterval(getDocumentCount, 1000); // updates every second

  io.on("connection", (socket) => {
    console.log("New client connected");
    getDocumentCount();

    socket.on("disconnect", () => {
      console.log("Client disconnected");
    });
  });
});

async function getDocumentCount() {
  if (!db) {
    console.log("DB not connected");
    return;
  }
  const count = await db.collection("mempool").estimatedDocumentCount({});
  const formattedNumber = count.toLocaleString("fr-FR"); // outputs "123 456 789"

  io.emit("document count", formattedNumber);
}

server.listen(3000, () => {
  console.log("Listening on port 3000");
});
