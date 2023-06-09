package main

import (
    "fmt"
    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/ethclient"
)

func main() {
    // Connect to the Ethereum network.
    client, err := ethclient.Dial("ws://localhost:8545")
    if err != nil {
        fmt.Println(err)
        return
    }

    // Get the latest block number.
    latestBlockNumber, err := client.BlockNumber()
    if err != nil {
        fmt.Println(err)
        return
    }

    // Get all transactions from the latest block.
    transactions := client.GetTransactions(latestBlockNumber)

    // Filter out only the transactions that originate from Uniswap.
    uniswapTransactions := []common.Hash{}
    for _, transaction := range transactions {
        if transaction.To() == common.HexToAddress("0x2654f910a080711b84782112c977c§814694a0e38") {
            uniswapTransactions = append(uniswapTransactions, transaction.Hash())
        }
    }

    // Print the list of Uniswap transactions.
    for _, transactionHash := range uniswapTransactions {
        fmt.Println(transactionHash)
    }
}
