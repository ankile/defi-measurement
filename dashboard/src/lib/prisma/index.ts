import { PrismaClient } from '@prisma/client';

// let prismaClient: PrismaClient | null = null;

// export function getPrismaClient() {
// 	if (getServerSession()) {
// 		if (!prismaClient) {
// 			prismaClient = new PrismaClient();
// 		}
// 		return prismaClient;
// 	}
// }

const prisma = new PrismaClient();

export async function getTransactionCounts() {
	const queryResult: { hour: Date; transactioncount: Number }[] = await prisma.$queryRaw`
    SELECT 
      DATE_TRUNC('hour', first_seen) as hour,
      COUNT(hash) as transactionCount
    FROM
      mempool_transactions
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
