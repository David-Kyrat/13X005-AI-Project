<script src="node_modules/reveal.js-menu/menu.js">
  import { onMount } from "svelte"

  import Reveal from "reveal.js"
  import Highlight from "reveal.js/plugin/highlight/highlight"
  import Markdown from "reveal.js/plugin/markdown/markdown"
  import Notes from "reveal.js/plugin/notes/notes"
  import RevealMath from "reveal.js/plugin/math/math"
  import RevealZoom from "reveal.js/plugin/zoom/zoom"
  // import RevealMenu from "reveal.js-menu/menu"
  import RevealMenu from "reveal.js-menu/menu.js"

  import "reveal.js/dist/reveal.css"
  import "reveal.js/dist/theme/sky.css"
  // import 'reveal.js/dist/theme/black.css'
  import "reveal.js/plugin/highlight/monokai.css"

  onMount(() => {
    const deck = new Reveal({
      plugins: [Markdown, Highlight, Notes, RevealMath.KaTeX, RevealZoom, RevealMenu],
      autoAnimateEasing: "ease",
      autoAnimateDuration: 0.8,
      hash: true,
      theme: "sky",
      slideNumber: "c/t",
      center: true,
      progress: false,
      pdfMaxPagesPerSlide: 1,
      menu: {
        // Specifies which side of the presentation the menu will
        // be shown. Use 'left' or 'right'.
        side: "left",

        // Specifies the width of the menu.
        // Can be one of the following:
        // 'normal', 'wide', 'third', 'half', 'full', or
        // any valid css length value
        width: "normal",

        // Add slide numbers to the titles in the slide list.
        // Use 'true' or format string (same as reveal.js slide numbers)
        numbers: false,
        titleSelector: "h1, h2, h3, h4, h5, h6",
        useTextContentForMissingTitles: true
      }
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
