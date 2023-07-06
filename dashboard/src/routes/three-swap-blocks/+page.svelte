<script lang="ts">
	import { onMount } from 'svelte';
	import type { PageData } from './$types';
	import Highcharts from 'highcharts';

	export let data: PageData;

	const { labels, counts } = data;

	const total = counts.reduce((a, b) => a + b, 0);

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
				text: 'Percentage of Counts of Each Transaction Order',
				align: 'left',
			},
			xAxis: {
				categories: labels,
				title: {
					text: 'Transaction Order (B = Buy, S = Sell))',
				},
				crosshair: true,
			},
			tooltip: {
				formatter: function () {
					const count = Math.round((this as any).y * total);
					return `Percentage: ${((this as any).y * 100).toFixed(2)}%<br>Absolute Count: ${count}`;
				},
			},
			yAxis: {
				title: { text: 'Percentage of Total' },
				labels: {
					formatter: function () {
						return (this as any).value * 100 + '%';
					},
				},
				plotLines: [
					{
						color: 'red', // Color value
						dashStyle: 'LongDash', // Style of the plot line. Default to solid
						value: 0.33, // Value of where the line will appear
						width: 2, // Width of the line
						label: {
							text: '33%', // Content of the label.
							align: 'right', // Positioning of the label.
							style: {
								color: 'gray',
							},
						},
					},
				],
			},
			plotOptions: {
    column: {
      dataLabels: {
        enabled: true,
        formatter: function() {
          return ((this as any).y * 100).toFixed(2) + '%';
        }
      },
    },
  },
			series: [
				{
					name: 'Swaps per block',
					showInLegend: false,
					type: 'column',
					data: counts.map((item) => item / total),
					// Make different color for each
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
</script>

<div id="chart-container" />
