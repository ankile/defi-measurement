<script lang="ts">
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import Highcharts from 'highcharts';

	export let data: PageData;

	const { count, frequency } = data;

	const total = frequency.reduce((a, b) => a + b, 0);

	function dynamicToFixed(value: number) {
		let absoluteValue = Math.abs(value);
		if (absoluteValue >= 1) {
			return value.toFixed(0);
		} else if (absoluteValue >= 0.1) {
			return value.toFixed(1);
		} else if (absoluteValue >= 0.01) {
			return value.toFixed(2);
		} else if (absoluteValue >= 0.001) {
			return value.toFixed(3);
		} else if (absoluteValue >= 0.0001) {
			return value.toFixed(4);
		} else if (absoluteValue >= 0.00001) {
			return value.toFixed(5);
		} else if (absoluteValue >= 0.000001) {
			return value.toFixed(6);
		} else {
			return value.toFixed(0);
		}
	}

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
			tooltip: {
				formatter: function () {
					const count = Math.round(this.y * total);
					return `Percentage: ${(this.y * 100).toFixed(2)}%<br>Absolute Count: ${count}`;
				},
			},
			yAxis: {
				type: scale,
				title: { text: 'Percentage of Total' },
				labels: {
					formatter: function () {
						return dynamicToFixed(this.value * 100) + '%';
					},
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
					name: 'Swaps per block',
					showInLegend: false,
					type: 'column',
					data: frequency.map((item) => item / total),
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

<button on:click={changeType}
	>Change to: {scale === 'logarithmic' ? 'Linear' : 'Logarithmic'}</button
>
