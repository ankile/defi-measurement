<script lang="ts">
	import { invalidate } from '$app/navigation';
	import type { PageData } from './$types';

	export let data: PageData;

	// Format the date
	$: formattedDate = Intl.DateTimeFormat('en-FR', {
		dateStyle: 'full',
		timeStyle: 'long'
	}).format(new Date(data.lastUpdated));

	// Format the number with spaces
	$: formattedNumber = data.documentCount.toLocaleString('fr-FR');

	let now = new Date().getTime() / 1000;
	$: lastUpdate = new Date(data.lastUpdated).getTime() / 1000;

	setInterval(() => {
		now = new Date().getTime() / 1000;
	}, 1000);
</script>

<h6>Number of transaction</h6>
<h1>
	<kbd>{formattedNumber}</kbd>
</h1>

<p>Latest update: {formattedDate}</p>
<p>{Math.max(0, Math.floor(now - lastUpdate))} seconds ago</p>

<a role="button" href="/" on:click={() => invalidate('document-count')}>Refresh</a>
