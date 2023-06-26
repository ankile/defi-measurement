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

	const { transactionCounts } = data;

	onMount(() => {
		if (browser) {
			new Chart(chartElement, {
				type: 'line',
				data: {
					labels: transactionCounts.map((item) =>
						DateTime.fromJSDate(item.hour).toFormat('yyyy-MM-dd HH:mm'),
					),
					datasets: [
						{
							label: 'Transactions per hour',
							data: transactionCounts.map((item) => item.transactionCount),
							backgroundColor: 'rgba(75, 192, 192, 0.2)', // Translucent teal
							borderColor: 'rgba(75, 192, 192, 1)', // Teal
							fill: 'origin', // fills the area under the line
							tension: 0.1,
						},
					],
				},
				options: {
					responsive: true,
					maintainAspectRatio: false,
					scales: {
						x: {
							ticks: {
								autoSkip: true,
								maxRotation: 45,
								minRotation: 45,
							},
						},
					},
				},
			});
		}
	});
</script>

<main class="main-container">
	<h1>Transactions per Hour</h1>
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
</style>
