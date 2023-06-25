import type { PageServerLoad } from './$types';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

type TransactionCount = {
	hour: Date;
	transactionCount: number;
};

export const load: PageServerLoad = async () => {
	console.log('In load (yeet!)');

	const transactionCounts: TransactionCount[] = await prisma.$queryRaw`
    SELECT 
      DATE_TRUNC('hour', first_seen) as hour,
      COUNT(hash) as transactionCount
    FROM
      mempool_transactions
    GROUP BY
      DATE_TRUNC('hour', first_seen)
    ORDER BY
      DATE_TRUNC('hour', first_seen);
  `.then((result) =>
		result.map((row) => ({
			hour: row.hour,
			transactionCount: row.transactioncount,
		})),
	);

	console.log('Ran query, got results', transactionCounts);

	// Convert big int to number
	transactionCounts.forEach((row) => {
		row.transactionCount = Number(row.transactionCount);
	});

	console.log('Converted big int to number returning');

	return { transactionCounts };
};
