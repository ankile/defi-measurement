import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

import clientPromise from '$lib/mongo';

export const load = (async () => {
	const client = await clientPromise;
	const db = client.db('transactions');
	const collection = db.collection('mempool');

	const documentCount = await collection.estimatedDocumentCount();

	if (documentCount) {
		return { documentCount };
	}

	throw error(404, 'Not found');
}) satisfies PageServerLoad;
