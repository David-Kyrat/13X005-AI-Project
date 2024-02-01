<script>
	import PresMd from '$lib/presMd.svelte'
	import '$lib/app.postcss'
	import { page } from '$app/stores'
	import Slide from '$lib/deck/slide.svelte'
	function lastPathName() {
		let pathname = $page.url.pathname
		const splitPn = pathname.split('/')
		return splitPn[splitPn.length - 1]
	}
	let basename = lastPathName()
	let idx = basename.substr(basename.length - 2, basename.length)
</script>

<PresMd filepaths={[`/md/${basename}.md`]}>
	<Slide>
		<h2 class="[word-spacing:20px]">Digital Image Processing<br /> Exam</h2>
		<p class="text-blue-900">Question {idx}</p>

		<p class="author">Noah Munz (19-815-489)</p>
		<p class="date text-slate-600">January 30th 2024</p>
	</Slide>
	<!--  Content from given Markdown files follows right after -->
</PresMd>
