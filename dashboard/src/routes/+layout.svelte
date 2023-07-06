<script lang="ts">
	let theme = 'light';

	function toggleTheme() {
		theme = theme === 'light' ? 'dark' : 'light';
		document.documentElement.setAttribute('data-theme', theme);
	}
</script>

<nav>
	<ul>
		<li><h3>DEX MEV Dashboard</h3></li>
	</ul>
	<ul>
		<li>
			<h3>
				<a href="#top" on:click={toggleTheme}>{theme === 'light' ? '🌚' : '🌞'}</a>
			</h3>
		</li>
	</ul>
</nav>

<ul>
	<li><a href="/">Home</a></li>
	<li><a href="/mempool-rate">Mempool Rate</a></li>
	<li><a href="/delay-histogram">Delay Histogram</a></li>
	<li><a href="/mempool-share">Mempool Share</a></li>
	<li><a href="/permutations">Transaction Order Permutations</a></li>
	<li><a href="/swaps-in-block">Swaps in Blocks per Pool</a></li>
</ul>

<body data-sveltekit-preload-data="hover">
	<slot />
</body>

<style>
	@import 'https://unpkg.com/@picocss/pico@latest/css/pico.min.css';

	:global(body) {
		padding: 2rem;
	}

	:global(input, button) {
		border-radius: 1rem;
	}

	nav {
		padding: 0 2rem;
	}
</style>
