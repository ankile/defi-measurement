import { getTransactionCounts } from '$lib/prisma';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	console.log('In load (yeet!)');

	const transactionCounts = await getTransactionCounts();

	// Convert big int to number
	transactionCounts.forEach((row) => {
		row.transactionCount = Number(row.transactionCount);
	});

	return { transactionCounts };
};
