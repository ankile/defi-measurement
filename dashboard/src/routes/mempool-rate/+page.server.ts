import { getTransactionCounts } from '$lib/prisma';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {

	const transactionCounts = await getTransactionCounts();

	return { transactionCounts };
};
