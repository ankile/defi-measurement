import type { PageServerLoad } from './$types';
import { getTableCounts } from '$lib/prisma';

export const load: PageServerLoad = async ({ depends }) => {
	const counts = await getTableCounts();

	depends('document-count');

	return counts;
};
