<script lang="ts">
	import { onMount } from 'svelte';
	import Highcharts, { dateFormat } from 'highcharts';
	import type { PageData } from './$types';

	export let data: PageData;

	const { mempoolTrue, mempoolFalse, blockTimestamp } = data;

	onMount(() => {
		Highcharts.chart({
			chart: {
				type: 'area',
				renderTo: 'chart-container',
			},
			title: {
				useHTML: true,
				text: 'Share of Uniswap V3 Swaps Originating from the Mempool',
				align: 'left',
			},

			accessibility: {
				point: {
					valueDescriptionFormat:
						'{index}. {point.category}, {point.y:,.1f} billions, {point.percentage:.1f}%.',
				},
			},
			xAxis: {
				// Format date as "Jun 25 16:31"
				categories: blockTimestamp.map((date) => dateFormat('%b %e %H:%M', Number(date))),
				tickInterval: 500,
			},
			yAxis: {
				labels: {
					format: '{value}%',
				},
				title: {
					text: null,
				},
			},
			tooltip: {
				pointFormat:
					'<span style="color:{series.color}">{series.name}</span>: <b>{point.percentage:.1f}%</b> ({point.y:,.1f} swaps/block)<br/>',
				split: true,
			},
			plotOptions: {
				area: {
					stacking: 'percent',
					marker: {
						enabled: false,
					},
				},
			},
			credits: {
				enabled: false,
			},
			series: [
				{
					name: 'From mempool',
					type: 'area',
					data: mempoolTrue,
				},
				{
					name: 'Other sources',
					type: 'area',
					data: mempoolFalse,
				},
			],
		});
	});
</script>

<div id="chart-container" />
