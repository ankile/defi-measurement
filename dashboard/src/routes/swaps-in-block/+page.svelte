<script lang="ts">
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import Highcharts from 'highcharts';

	export let data: PageData;

	const { count, frequency } = data;

	let scale = 'linear';
	let chart: Highcharts.Chart;

	onMount(() => {
		chart = Highcharts.chart({
			credits: {
				enabled: false,
			},
			chart: {
				renderTo: 'chart-container',
				type: 'column',
				height: 700,
				// Turn off the link to highcharts.com
			},
			title: {
				useHTML: true,
				text: 'Number of swaps per pool in a block',
				align: 'left',
			},
			xAxis: {
				categories: count.map((item) => String(item)),
				title: {
					text: 'Number swaps per block for a pool',
				},
				crosshair: true,
			},
			yAxis: {
				title: {
					text: 'Frequency',
				},
				type: scale,
			},
			plotOptions: {
				bar: {
					dataLabels: {
						enabled: true,
					},
				},
			},
			series: [
				{
					name: 'Transactions per hour',
					showInLegend: false,
					type: 'column',
					data: frequency,
					// Make a nice color that is not too dark, e.g., purple
					color: 'rgba(75, 0, 130, 1)',
				},
			],
			responsive: {
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

	function changeType() {
		scale = scale === 'linear' ? 'logarithmic' : 'linear';
		chart.update({
			yAxis: { type: scale },
		});
	}
</script>

<div id="chart-container" />

<button on:click={changeType}>Change to: {scale === 'logarithmic' ? 'Linear' : 'Logarithmic'}</button>
