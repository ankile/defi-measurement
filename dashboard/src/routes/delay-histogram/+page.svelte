<script lang="ts">
	import { browser } from '$app/environment';
	import '@fontsource/merriweather';
	import type { PageData } from './$types';
	import { DateTime } from 'luxon';

	import { Chart, registerables } from 'chart.js';
	import { onMount } from 'svelte';

	Chart.register(...registerables);

	let chartElement: HTMLCanvasElement;

	export let data: PageData;

	const { delayHistogram } = data;

	onMount(() => {
		if (browser) {
			new Chart(chartElement, {
				type: 'bar',
				data: {
					labels: delayHistogram.map((item) => item.diffInSeconds),
					datasets: [
						{
							label: 'Transactions per hour',
							data: delayHistogram.map((item) => item.frequency),
							backgroundColor: 'rgba(75, 192, 192, 0.2)', // Translucent teal
							borderColor: 'rgba(75, 192, 192, 1)', // Teal
						},
					],
				},
				options: {
					responsive: true,
					maintainAspectRatio: false,
				},
			});
		}
	});
</script>

<main class="main-container">
	<h1>Number of seconds in the mempool before included in a block</h1>
	<section class="chart-container">
		<canvas bind:this={chartElement} />
	</section>
</main>

<style>
	.chart-container {
		width: 70%;
		height: 400px; /* or any suitable height */
		margin: auto;
	}

	@media (max-width: 1300px) {
		.chart-container {
			width: 100%;
		}
	}
</style>
