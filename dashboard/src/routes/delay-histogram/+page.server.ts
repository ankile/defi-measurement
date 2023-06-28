import { getMempoolBlockDelayHistogram } from '$lib/prisma';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
	const delayHistogram = await getMempoolBlockDelayHistogram();

	return { delayHistogram };
};
