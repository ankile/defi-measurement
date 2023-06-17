const ethers = require("ethers");

const provider = new ethers.providers.WebSocketProvider("ws://127.0.0.1:8546");

const web3 = new ethers.providers.Web3(provider);

console.log(web3.eth.syncing);
