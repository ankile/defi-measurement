import type { PageServerLoad } from './$types';

import clientPromise from '$lib/mongo';

export const load = (async ({ depends }) => {
	const client = await clientPromise;
	const db = client.db('transactions');
	const collection = db.collection('mempool');

	const documentCount = await collection.estimatedDocumentCount();

	// Find when the latest document was created
	const newestDocument = await collection.findOne({}, { sort: { ts: -1 } });
	const lastUpdated = newestDocument?.ts ?? 0;

	depends('document-count');

	return { documentCount, lastUpdated };
}) satisfies PageServerLoad;
