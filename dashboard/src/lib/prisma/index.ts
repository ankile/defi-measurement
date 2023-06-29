import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function getTransactionCounts() {
	const queryResult: { hour: Date; transactioncount: Number }[] = await prisma.$queryRaw`
    SELECT 
      DATE_TRUNC('hour', first_seen) as hour,
      COUNT(hash) as transactionCount
    FROM
      mempool_transactions
    WHERE
      DATE_TRUNC('hour', first_seen) < DATE_TRUNC('hour', NOW())
    GROUP BY
      DATE_TRUNC('hour', first_seen)
    ORDER BY
      DATE_TRUNC('hour', first_seen);
  `;

	return queryResult.map((row) => ({
		hour: row.hour,
		transactionCount: row.transactioncount,
	}));
}

export async function getMempoolBlockDelayHistogram() {
	const mempoolBlockDelayHistogram: { diffInSeconds: Number; frequency: Number }[] =
		await prisma.$queryRaw`
    SELECT 
        EXTRACT(EPOCH FROM (swaps.block_timestamp - date_trunc('second', mempool_transactions.first_seen))) AS diffInSeconds,
        COUNT(*) AS frequency
    FROM
        swaps
    INNER JOIN 
        mempool_transactions 
        ON swaps.transaction_hash = mempool_transactions.hash
    GROUP BY
      diffInSeconds
    HAVING 
        EXTRACT(EPOCH FROM (swaps.block_timestamp - date_trunc('second', mempool_transactions.first_seen))) <= 35
    ORDER BY
      diffInSeconds ASC;
  `;

	// Convert from bigints to numbers before returning

	return mempoolBlockDelayHistogram.map((row) => ({
		diffInSeconds: Number(row.diffinseconds),
		frequency: Number(row.frequency),
	}));
}

export async function getTableCounts() {
	const [uniswapV2]: Array<{ count: number; latesttimestamp: Date }> = await prisma.$queryRaw`
		SELECT COUNT(*) as count, MAX(block_timestamp) as latestTimestamp, MIN(block_timestamp) as earliestTimestamp
		FROM swaps_v2;
	`;

	const [uniswapV3]: Array<{ count: number; latesttimestamp: Date }> = await prisma.$queryRaw`
		SELECT COUNT(*) as count, MAX(block_timestamp) as latestTimestamp, MIN(block_timestamp) as earliestTimestamp
		FROM swaps;
	`;

	const [mempool]: Array<{ count: number; latesttimestamp: Date }> = await prisma.$queryRaw`
		SELECT COUNT(*) as count, MAX(first_seen) as latestTimestamp, MIN(first_seen) as earliestTimestamp
		FROM mempool_transactions;
	`;

	return {
		uniswapV2: {
			count: Number(uniswapV2.count),
			latestTimestamp: new Date(uniswapV2.latesttimestamp),
			earliestTimestamp: new Date(uniswapV2.earliesttimestamp),
		},
		uniswapV3: {
			count: Number(uniswapV3.count),
			latestTimestamp: new Date(uniswapV3.latesttimestamp),
			earliestTimestamp: new Date(uniswapV3.earliesttimestamp),
		},
		mempool: {
			count: Number(mempool.count),
			latestTimestamp: new Date(mempool.latesttimestamp),
			earliestTimestamp: new Date(mempool.earliesttimestamp),
		},
	};
}
