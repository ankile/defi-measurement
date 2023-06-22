<script lang="ts">
	import type { PageData } from './$types';

	export let data: PageData;

	// Format the date
	$: formattedDate = Intl.DateTimeFormat('en-FR', {
		dateStyle: 'full',
		timeStyle: 'long'
	}).format(new Date(data.lastUpdated));

	// Format the number with spaces
	$: formattedNumber = data.documentCount.toLocaleString('fr-FR');
</script>

<h1>Dashboard!</h1>
<h3>Number of documents (estimate)</h3>
<p class="big-number">{formattedNumber}</p>

<p>Latest update: {formattedDate}</p>

<style>
	.big-number {
		font-size: 3rem;
		font-weight: bold;
	}
</style>
