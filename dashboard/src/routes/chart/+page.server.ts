import type { PageServerLoad } from './$types';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

type TransactionCount = {
	hour: Date;
	transactionCount: number;
};

export const load: PageServerLoad = async () => {
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

	// Convert big int to number
	transactionCounts.forEach((row) => {
		row.transactionCount = Number(row.transactionCount);
	});

	return { transactionCounts };
};
