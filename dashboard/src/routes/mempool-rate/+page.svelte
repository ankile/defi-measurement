<script lang="ts">
	import type { PageData } from './$types';

	import Highcharts from 'highcharts';
	import { onMount } from 'svelte';

	export let data: PageData;

	const { transactionCounts } = data;

	onMount(() => {
		Highcharts.chart({
			chart: {
				type: 'line',
				renderTo: 'chart-container',
				// backgroundColor: '#f9f9f9', // Light grey background color
				height: 700,
			},
			title: {
				useHTML: true,
				text: 'Transaction Counts per Hour', // Add a title for your chart
				align: 'left',
			},
			xAxis: {
				categories: transactionCounts.map((item) =>
					Highcharts.dateFormat('%b %e %H:%M', Number(item.hour)),
				),
				crosshair: true, // Enable a crosshair for easier data reading
				tickInterval: 10,
			},
			yAxis: {
				title: {
					text: 'Transaction Counts', // Add a label for the y-axis
				},
				min: 0, // Start the y-axis at 0
			},
			series: [
				{
					name: 'Transactions per hour',
					type: 'line',
					data: transactionCounts.map((item) => item.transactionCount),
					color: '#7cb5ec', // Blue color for the line
					marker: {
						enabled: false, // Disable markers for a cleaner look
					},
					lineWidth: 2,
				},
			],
			credits: {
				enabled: false, // Disable the highcharts.com credits text
			},
			responsive: {
				// Make the chart responsive
				rules: [
					{
						condition: {
							maxWidth: 500,
						},
						chartOptions: {
							legend: {
								layout: 'horizontal',
								align: 'center',
								verticalAlign: 'bottom',
							},
						},
					},
				],
			},
		});
	});
</script>

<div id="chart-container" />
