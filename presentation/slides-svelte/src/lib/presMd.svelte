<script>
  import Slides from "$lib/deck/slides.svelte"
  import Slide from "$lib/deck/slide.svelte"
  import Markdown from "$lib/deck/markdown.svelte"
  import HomeBtn from "$lib/homeBtn.svelte"
  import "$lib/app.postcss"
  export let filepaths = ["/example.md"]
  export let clas = ""
  $: _clas = `text-xl overflow-y-auto overflow-x-auto no-scrollbar prez ${clas}`
</script>

<div class="flex items-baseline justify-between">
  <HomeBtn />
</div>

<Slides>
  <slot />
  {#each filepaths as subpres}
    <Slide>
      <Markdown name={subpres} external clas={_clas} />
    </Slide>
  {/each}
</Slides>
