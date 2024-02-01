<script>
	import { page } from '$app/stores'
	$: idx = $page.url.pathname.replace('/q', '')
	let theme = 0
</script>

<svelte:head>
	<link
		href="https://fonts.googleapis.com/css2?family=Material+Icons&display=swap"
		rel="stylesheet"
	/>
	<link rel="icon" href="/unige-logo.svg" />
  <title>IA - Oral - Presentation</title>
	<link rel="stylesheet" href="/themes/common.css" />
	<!-- HINT: Theme chooser -->
	{#if theme == 0}
		<link rel="stylesheet" href="/themes/fixation.css" />
	{:else if theme == 1}
		<link rel="stylesheet" href="/themes/material.css" />
	{:else if theme == 2}
		<link rel="stylesheet" href="/themes/svelte.css" />
	{:else if theme == 3}
		<link rel="stylesheet" href="/themes/metro.css" />
	{:else if theme == 4}
		<link rel="stylesheet" href="/themes/unity.css" />
	{/if}
</svelte:head>

<slot />
