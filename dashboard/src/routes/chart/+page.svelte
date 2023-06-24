<script lang="ts">
	import { browser } from '$app/environment';
	import '@fontsource/merriweather';

	import { Chart, registerables } from 'chart.js';
	import { onMount } from 'svelte';

	Chart.register(...registerables);

	const data = [
		{ block_number: 17495345, mempool_true: 2, mempool_false: 1 },
		{ block_number: 17495346, mempool_true: 2, mempool_false: 2 },
		{ block_number: 17495347, mempool_true: 5, mempool_false: 12 },
		{ block_number: 17495348, mempool_true: 3, mempool_false: 5 },
		{ block_number: 17495349, mempool_true: 0, mempool_false: 3 },
		{ block_number: 17495351, mempool_true: 6, mempool_false: 1 },
		{ block_number: 17495352, mempool_true: 3, mempool_false: 4 },
		{ block_number: 17495353, mempool_true: 4, mempool_false: 3 },
		{ block_number: 17495354, mempool_true: 1, mempool_false: 11 },
		{ block_number: 17495355, mempool_true: 8, mempool_false: 1 }
	];

	let chartElement: HTMLCanvasElement;

	data.forEach((item) => {
		const total = item.mempool_true + item.mempool_false;
		item.mempool_true = (item.mempool_true / total) * 100;
		// item.mempool_false = item.mempool_true;
		item.mempool_false = (item.mempool_false / total) * 100;
	});

	onMount(() => {
		if (browser) {
			new Chart(chartElement, {
				type: 'line',
				data: {
					labels: data.map((item) => item.block_number),
					datasets: [
						{
							label: 'mempool_true',
							data: data.map((item) => item.mempool_true),
							backgroundColor: 'rgba(75, 192, 192, 0.2)', // Translucent teal
							borderColor: 'rgba(75, 192, 192, 1)', // Teal
							fill: 'origin', // fills the area under the line
							tension: 0.1
						},
						{
							label: 'mempool_false',
							data: data.map((item) => item.mempool_false),
							backgroundColor: 'rgba(255, 99, 132, 0.2)', // Translucent red
							borderColor: 'rgba(255, 99, 132, 1)', // Red
							fill: 'origin', // fills the area under the line
							tension: 0.1
						}
					]
				},
				options: {
					scales: {
						y: {
							beginAtZero: true,
							stacked: true
							// ticks: {
							// 	callback: function (value) {
							// 		return value + '%'; // transform numbers into percentage values
							// 	}
							// }
						}
					}
				}
			});
		}
	});
</script>

<main class="main-container">
	<h1>State of JS 2021 Backend Framework Satisfaction</h1>
	<section>
		<canvas bind:this={chartElement} />
	</section>
</main>
