<script>
  import "$lib/app.postcss"
  import LayoutGrid, { Cell } from "@smui/layout-grid"
  import Button, { Label } from "@smui/button"
  import Drawer from "$lib/drawer.svelte"
  import IconButton from "@smui/icon-button"
  import Paper, { Title, Subtitle, Content } from "@smui/paper"

  let oral_question_nb = 21
  // i: int -> string
  function fmtIdx(i) {
    return i < 10 ? `0${i}` : `${i}`
  }
  let drawerOpen = false
  function setOpen(value) {
    drawerOpen = value
  }
  function switchOpen() {
    drawerOpen = !drawerOpen
  }
  function range(length) {
    return Array.from({ length }, (_, i) => i)
  }
  let questions = [
    "Q1. HVS: explain the image formation in the eye. What are the differences in focusing between the ordinary cameras and the human eyes?",
    "Q2. HVS. Explain the fundamentals of color vision. Explain the additive and subtractive color mixings.",
    "Q3. HVS. Explain the CMYK model. Exemplify the reflection of colors. Explain the principles of halftoning. Why is K used in CMYK model in practice, while it is enough to have CMY to simulate any color?",
    "Q4. HVS. Explain the chromaticity color diagram. What is the achievable color gamut? Which factors determine the achievable color gamut for the screens and printers?",
    "Q5. Image acquisition. Explain the role of a sensor in imaging. Explain the principle of photon counting. Which size of sensors is of preference in practice? Explain the trade-offs.",
    "Q6. Image acquisition. Explain the design of modern sensors.",
    "Q7. Image acquisition. Explain the different types of noise in digital images. How can one reduce the noise in digital images?",
    "Q8. Image acquisition. Explain the principles of modern color imaging. What is a color filter array? What is the demosaicing?",
    "Q9. Histogram transformation. Explain and exemplify the example of contrast stretching and gamma correction. Where do we use these operations in practice?",
    "Q10. Histogram transformation. Explain and exemplify the example of histogram matching.",
    "Q11. Geometrical transformations. Explain what is the affine matrix. Explain the shift, scaling, flipping and change of aspect ratio.",
    "Q12. Geometrical transformations. Explain the affine family of transformations. Explain the rotation and shearing.",
    "Q13. Geometrical transformations. Explain the projective transformations.",
    "Q14. Spatial filters: Explain the difference between the convolution and correlation. Give examples.",
    "Q15. Spatial filters: Explain the border effects during the computation of 2D convolution. Define the total size of filtered images. Give examples.",
    "Q16. Spatial filters: Explain the properties of convolution.",
    "Q17. Spatial filters: Explain the advantages of separable filters for the computation of 2D convolution.",
    "Q18. Spatial filters: Explain the low-pass filtering. Exemplify the box and Gaussian filters.",
    "Q19. Spatial filters: Explain the high-pass filtering. Exemplify the Laplacian filter.",
    "Q20. Spatial filters: Explain the high-pass filtering. Exemplify the gradients.",
    "Q21. Spatial filters: Explain the unsharpened mask and its link to the Laplacian sharpening."
  ]
</script>

<h1>test</h1>

<!-- <div class="flex items-baseline sticky top-0 bg-white z-30 border border-slate-200 rounded-lg">
  <IconButton class="material-icons" on:click={switchOpen}>menu</IconButton>
  <h1 class="w-fit text-3xl block mx-auto mt-2 font-bold">Choose a question</h1>
</div>
<Drawer open={drawerOpen} fun={setOpen} />

<LayoutGrid>
  {#each Array(oral_question_nb + 1) as _unused, i}
    <Cell class="drop-shadow-md hover:drop-shadow-xl my-5">
      <a href="/q{fmtIdx(i)}" target="_blank">
        <Button variant="raised" class="w-full h-[190%] button-shaped-round demo-cell">
          <Label>{i == 0 ? "Example" : `Q${fmtIdx(i)}`}</Label>
        </Button>
      </a>
    </Cell>
  {/each}
</LayoutGrid>

<hr />

<div class="paper-container px-10 py-5 pt-20 block w-fit mx-auto">
  <Paper>
    <Title class="w-fit text-2xl block mx-auto font-bold">Question Description</Title>
    <Subtitle class="text-slate-600">All 21 questions of oral</Subtitle>
    <Content>
      <ul class="flex flex-col justify-items-start px-5">
        {#each questions as question, i}
          <a href="q{fmtIdx(i + 1)}" class="text-black hover:text-slate-700" target="_top">
            <li class="py-2 pl-0 ml-0 text-left text-lg">{question}</li>
          </a>
        {/each}
      </ul>
    </Content>
  </Paper>
</div> -->
