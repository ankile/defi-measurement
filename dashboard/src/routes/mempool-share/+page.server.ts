import type { PageServerLoad } from './$types';
import { getSwapsV3MempoolShare } from '$lib/prisma';

export const load: PageServerLoad = async () => {
	const v3MempoolShare = await getSwapsV3MempoolShare();

	return {
		mempoolTrue: v3MempoolShare.map((row) => row.mempoolTrue),
		mempoolFalse: v3MempoolShare.map((row) => row.mempoolFalse),
		blockNumber: v3MempoolShare.map((row) => row.blockNumber),
		blockTimestamp: v3MempoolShare.map((row) => row.blockTimestamp),
	};
};
