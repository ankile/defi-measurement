import type { PageLoad } from './$types';

export const load: PageLoad = (async () => {
	return {
		labels: ['BBS/SSB', 'BSB/SBS', 'SBB/BSS'],
		counts: [202608, 119904, 157488],
	};
});
