<script lang="ts">
	import type { PageData } from './$types';
	import { invalidate } from '$app/navigation';
	import { format as formatDate } from 'date-fns';

	export let data: PageData;

	const { uniswapV2, uniswapV3, mempool } = data;

	// Format numbers with space as thousand separator
	const formatNumber = (num: Number) => num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
</script>

<div class="grid">
	<div>
		<h6>Uniswap V2 Swaps</h6>
		<h1>
			<kbd>{formatNumber(uniswapV2.count)}</kbd>
		</h1>
		<span>Last updated: {formatDate(uniswapV2.latestTimestamp, 'dd-MM-yyyy HH:MM')}</span>
	</div>
	<div>
		<h6>Uniswap V3 Swaps</h6>
		<h1>
			<kbd>{formatNumber(uniswapV3.count)}</kbd>
		</h1>
		<span>Last updated: {formatDate(uniswapV2.latestTimestamp, 'dd-MM-yyyy HH:MM')}</span>
	</div>
	<div>
		<h6>Mempool Transactions</h6>
		<h1>
			<kbd>{formatNumber(mempool.count)}</kbd>
		</h1>
		<span>Last updated: {formatDate(uniswapV2.latestTimestamp, 'dd-MM-yyyy HH:MM')}</span>
	</div>
</div>

<style>
	span {
		font-size: 0.8rem;
		color: #777;
		font-style: italic;
	}

	h1 {
		margin-bottom: 2rem;
	}
</style>
