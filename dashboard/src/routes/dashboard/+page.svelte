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

	// setInterval(() => {
	// 	invalidate('document-count');
	// }, 1000);

	let now = new Date().getTime() / 1000;
	$: lastUpdate = new Date(data.lastUpdated).getTime() / 1000;

	setInterval(() => {
		now = new Date().getTime() / 1000;
	}, 1000);
</script>

<h1>Dashboard!</h1>
<h3>Number of documents (estimate)</h3>
<p class="big-number">{formattedNumber}</p>

<span>Latest update: {formattedDate}</span>
<span>{Math.max(0, Math.floor(now - lastUpdate))} seconds ago</span>

<button class="refresh-button" on:click={() => invalidate('document-count')}>Refresh</button>

<style>
	.big-number {
		font-size: 3rem;
		font-weight: bold;
	}

	.refresh-button {
		margin-top: 1rem;
		width: 150px;
		height: 50px;
		background-color: #fff;
		border: none;
		border-radius: 5px;

		&:hover {
			background-color: #eee;
			cursor: pointer;
		}
	}
</style>
