import { getPermutationSimulations } from '$lib/prisma';
import type { PermutationSimulation } from '@prisma/client';
import type { PageServerLoad } from './$types';

async function extractData(run: PermutationSimulation) {
	const { dataLocation } = run;
	const data = await fetch(dataLocation);
	const json = await data.json();

	return {
		originalPrices: json.original_prices,
		permutationPrices: json.permutation_prices,
		stats: json.stats,
	};
}

export const load: PageServerLoad = async () => {
	const simRuns = await getPermutationSimulations();

	const data = await Promise.all(simRuns.map(extractData));

	const simulationData = simRuns.map((run, i) => ({
		...run,
		...data[i],
	}));


	return { simulationData };
};
