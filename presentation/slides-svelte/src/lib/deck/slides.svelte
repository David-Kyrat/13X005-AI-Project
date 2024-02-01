<script>
	import { onMount } from 'svelte'

	import Reveal from 'reveal.js'
	import Highlight from 'reveal.js/plugin/highlight/highlight'
	import Markdown from 'reveal.js/plugin/markdown/markdown'
	import Notes from 'reveal.js/plugin/notes/notes'
	import RevealMath from 'reveal.js/plugin/math/math'

	import 'reveal.js/dist/reveal.css'
	import 'reveal.js/dist/theme/sky.css'
	// import 'reveal.js/dist/theme/black.css'
	import 'reveal.js/plugin/highlight/monokai.css'

	onMount(() => {
		const deck = new Reveal({
			plugins: [Markdown, Highlight, Notes, RevealMath.KaTeX],
			autoAnimateEasing: 'ease',
			autoAnimateDuration: 0.8,
			hash: true,
			theme: 'sky',
			slideNumber: 'c/t',
			center: true,
			progress: false
		})
		deck.initialize()
	})
</script>

<!-- <div class="reveal h-[113%] border border-red-400"> -->
<div class="reveal h-dvh absolute top-2">
	<div class="slides">
		<slot />
	</div>
</div>

<style>
	/*.slides {*/
		/* transform:none !important;  */
		/* inset: 40% auto auto 0% !important; */
	/*}*/
</style>
