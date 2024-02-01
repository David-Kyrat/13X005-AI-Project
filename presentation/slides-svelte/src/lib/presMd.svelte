<script>
  import Slides from "$lib/deck/slides.svelte"
  import Markdown from "$lib/deck/markdown.svelte"
  import HomeBtn from "$lib/homeBtn.svelte"
  /* import Drawer from '$lib/drawer.svelte'
	import IconButton from '@smui/icon-button'

	let drawerOpen = false
	function setOpen(value) {
		drawerOpen = value
	}
	function switchOpen() {
		drawerOpen = !drawerOpen
	} */

  import "$lib/app.postcss"
  export let filepaths = ["/example.md"]
  export let clas = "text-xl overflow-y-auto overflow-x-auto no-scrollbar prez"
</script>

<div class="flex items-baseline justify-between">
  <!-- <IconButton class="material-icons z-50" on:click={switchOpen}>menu</IconButton> -->
  <HomeBtn />
</div>
<!-- <Drawer open={drawerOpen} fun={setOpen} /> -->

<Slides>
  <slot />
  {#each filepaths as subpres}
    <Markdown name={subpres} external {clas} />
  {/each}
  <!-- <section data-markdown={name} class={clas} /> -->
</Slides>
