const ethers = require("ethers");

// Make an HTTP Provider
const provider = new ethers.providers.JsonRpcProvider("http://localhost:8545");

const signer = provider.getSigner();

async function getSyncStatus() {
  const syncInfo = await signer.send("eth_syncing");
  return syncInfo;
}

async function main() {
  const syncStatus = await getSyncStatus();
  console.log(syncStatus);
}

main();
