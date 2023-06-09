const EthDater = require("ethereum-block-by-date");
const Web3 = require("web3");

const web3 = new Web3(new Web3.providers.HttpProvider(process.env.PROVIDER));
const dater = new EthDater(web3);

let block = await dater.getDate(
  "2022-06-05T00:00:00Z", // Date, required. Any valid moment.js value: string, milliseconds, Date() object, moment() object.
  true, // Block after, optional. Search for the nearest block before or after the given date. By default true.
  false // Refresh boundaries, optional. Recheck the latest block before request. By default false.
);

console.log(block.block); // prints the block number
