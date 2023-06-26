<script lang="ts">
	let theme = 'light';

	function toggleTheme() {
		theme = theme === 'light' ? 'dark' : 'light';
		document.documentElement.setAttribute('data-theme', theme);
	}
</script>

<nav>
	<ul>
		<li><h3>DEX MEV Dashboard (Manual)</h3></li>
	</ul>
	<ul>
		<li>
			<h3>
				<a href="#top" on:click={toggleTheme}>{theme === 'light' ? '🌚' : '🌞'}</a>
			</h3>
		</li>
	</ul>
</nav>

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
