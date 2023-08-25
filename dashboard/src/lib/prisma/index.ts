import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function getTransactionCounts() {
	const queryResult: { hour: Date; transaction_count: number }[] = await prisma.$queryRaw`
    SELECT * FROM mempool_rate_per_hour
    ORDER BY hour;
  `;

	return queryResult.map((row) => ({
		hour: row.hour,
		transactionCount: Number(row.transaction_count),
	}));
}

export async function getMempoolBlockDelayHistogram() {
	const mempoolBlockDelayHistogram: { diffinseconds: Number; frequency: Number }[] =
		await prisma.$queryRaw`
    SELECT 
        EXTRACT(EPOCH FROM (swaps.block_ts - date_trunc('second', mempool_transactions.first_seen))) AS diffInSeconds,
        COUNT(*) AS frequency
    FROM
        swaps
    INNER JOIN 
        mempool_transactions 
        ON swaps.tx_hash = mempool_transactions.hash
    GROUP BY
      diffInSeconds 
    HAVING 
        EXTRACT(EPOCH FROM (swaps.block_ts - date_trunc('second', mempool_transactions.first_seen))) <= 20
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

	const [uniswapV3]: Array<{ count: number; latesttimestamp: Date, earliesttimestamp: Date }> = await prisma.$queryRaw`
		SELECT COUNT(*) as count, MAX(block_ts) as latestTimestamp, MIN(block_ts) as earliestTimestamp
		FROM swaps;
	`;

	const [mempool]: Array<{ count: number; latesttimestamp: Date, earliesttimestamp: Date }> = await prisma.$queryRaw`
		SELECT COUNT(*) as count, MAX(first_seen) as latestTimestamp, MIN(first_seen) as earliestTimestamp
		FROM mempool_transactions;
	`;

	return {
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

export async function getSwapsV3MempoolShare() {
	type QueryResultType = {
		block_number: Number;
		block_ts: Date;
		avg_mempool_true: Number;
		avg_mempool_false: Number;
		avg_total: Number;
	};
	const queryResult: QueryResultType[] = await prisma.$queryRaw`
    SELECT
      *
    FROM
      mempool_share
  `;

  return queryResult.map((row) => ({
    blockNumber: Number(row.block_number),
    blockTimestamp: new Date(row.block_ts),
    mempoolTrue: Number(row.avg_mempool_true),
    mempoolFalse: Number(row.avg_mempool_false),
    total: Number(row.avg_total),
  }));
}


export async function getPermutationSimulations(searchParams: URLSearchParams) {
  const { orderBy, order, limit, skip } = Object.fromEntries(searchParams.entries());

  const queryResult = await prisma.permutation_simulations.findMany({
    orderBy: {
      [orderBy ?? 'n_swaps']: order ?? 'desc',
    },
    take: Number(limit ?? 5),
    skip: Number(skip ?? 0),
  })

  return queryResult;
}