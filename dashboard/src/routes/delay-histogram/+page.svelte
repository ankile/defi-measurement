<script lang="ts">
	import '@fontsource/merriweather';
	import type { PageData } from './$types';

	import { onMount } from 'svelte';
	import Highcharts from 'highcharts';


	export let data: PageData;

	const { delayHistogram } = data;

	onMount(() => {
		Highcharts.chart({
			chart: {
				renderTo: 'chart-container',
				type: 'column',
				// backgroundColor: '#f9f9f9', // Light grey background color
				height: 700,
			},
			title: {
				useHTML: true,
				text: 'Number of seconds in the mempool before included in a block',
				align: 'left',
			},
			xAxis: {
				categories: delayHistogram.map((item) => String(item.diffInSeconds)),
				title: {
					text: 'Delay in Seconds',
				},
				crosshair: true,
			},
			yAxis: {
				min: 0,
				title: {
					text: 'Frequency',
				},
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
					type: 'column',
					data: delayHistogram.map((item) => item.frequency),
					color: 'rgba(75, 192, 192, 1)', // Teal
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
</script>

<div id="chart-container" />
