<script lang="ts">
	import type { PageData } from './$types';
	import highcharts from '$lib/highcharts-action';

	export let data: PageData;

	const { simulationData } = data;

	type TooltipData = {
		points: Array<{
			y: Number;
		}>;
	};

	const chartConfigList = simulationData.map((simulation) => {
		const { blockNumber, nSwaps, nPermutations, originalPrices, permutationPrices } = simulation;
		return {
			...simulation,
			chartConfig: {
				chart: {
					type: 'line',
					height: 750,
				},
				title: {
					text: `Block ${blockNumber} - ${nSwaps - 1} swaps - ${nPermutations} permutations`,
				},
				xAxis: {
					title: {
						text: 'Swap Number in Block',
					},
				},
				yAxis: {
					title: {
						text: 'Price',
					},
				},
				tooltip: {
					shared: true,
					formatter: function (): string {
						// Get last point
						let actualPrice: Number = Math.round(this.points[this.points.length - 1].y * 100) / 100;

						let averagePrice = 0;
						for (let i = 0; i < this.points.length - 1; i++) {
							averagePrice += this.points[i].y;
						}
						averagePrice = Math.round((averagePrice / (this.points.length - 1)) * 100) / 100;

						return `Swap ${this.x}<br>Price: ${actualPrice}<br>Average: ${averagePrice}`;
					},
				},
				series: [
					...permutationPrices.map((series: Array<Number>) => ({
						type: 'line',
						name: 'Permutation',
						showInLegend: false,
						data: series,
						marker: {
							enabled: false,
						},
						color: 'rgba(0, 0, 0, 0.1)', // black color with 20% opacity
					})),
					{
						name: 'Original',
						type: 'line',
						data: originalPrices as Array<Number>,
						marker: {
							enabled: true,
						},
						color: '#ff0000',
					},
					{
						name: 'Random Permutation',
						type: 'line',
						data: [],
						color: '#000000',
						marker: {
							enabled: false,
						},
						showInLegend: true,
					},
				],
			},
		};
	});
</script>

<h1>Permutations</h1>

{#each chartConfigList as { chartConfig, originalStd, meanPermutationStd, originalArea, meanPermutationArea, maxAbsOriginalDeviation, meanMaxAbsPermutationDeviation }}
	<div class="container">
		<h3>Random permutations</h3>
		<div class="chart" use:highcharts={chartConfig} />

		<h3>Measurements</h3>
		<table>
			<thead>
				<td>Metric</td>
				<td>Realized Swaps (Real)</td>
				<td>Mean Random Permutations (Simulations)</td>
				<td>Percentage Difference</td>
			</thead>
			<tbody>
				<tr>
					<td>Standard Deviation</td>
					<td>{Math.round(originalStd * 100) / 100}</td>
					<td>{Math.round(meanPermutationStd * 100) / 100}</td>
					<td>{Math.round(((meanPermutationStd - originalStd) / originalStd) * 10000) / 100}%</td>
				</tr>
				<tr>
					<td>Absolute Area to Baseline</td>
					<td>{Math.round(originalArea * 100) / 100}</td>
					<td>{Math.round(meanPermutationArea * 100) / 100}</td>
					<td>{Math.round(((meanPermutationArea - originalArea) / originalArea) * 10000) / 100}%</td>
				</tr>
				<tr>
					<td>Max Absolute Deviation</td>
					<td>{Math.round(maxAbsOriginalDeviation * 100) / 100}</td>
					<td>{Math.round(meanMaxAbsPermutationDeviation * 100) / 100}</td>
					<td>{Math.round(((meanMaxAbsPermutationDeviation - maxAbsOriginalDeviation) / maxAbsOriginalDeviation) * 10000) / 100}%</td>
				</tr></tbody
			>
		</table>
	</div>
	<!-- Make a separator line -->
	<hr />
{/each}

<style>
	table {
		margin-top: 2rem;
	}

	.container {
		margin-bottom: 5rem;
	}
</style>
