const readline = require("readline");

function printTransactionCounts(
  rl,
  totalTransactions,
  uniswapTransactionCount,
  transactionCounts
) {
  readline.cursorTo(rl, 0, 0);
  readline.clearScreenDown(rl);
  rl.write(
    `Total Transactions: ${totalTransactions}\n` +
      `Total Uniswap Transactions: ${uniswapTransactionCount} (${(
        (uniswapTransactionCount / totalTransactions) *
        100
      ).toFixed(2)}%)\n` +
      `${Object.entries(transactionCounts)
        .map(
          ([routerContract, count]) =>
            `${routerContract}: ${count} (${(
              (count / uniswapTransactionCount) *
              100
            ).toFixed(2)}%)`
        )
        .join("\n")}`
  );
}
exports.printTransactionCounts = printTransactionCounts;
