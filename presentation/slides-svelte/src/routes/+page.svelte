<script>
  import PresMd from "$lib/presMd.svelte"
  import "$lib/app.postcss"
  import { page } from "$app/stores"
  import Slide from "$lib/deck/slide.svelte"
  function lastPathName() {
    let pathname = $page.url.pathname
    const splitPn = pathname.split("/")
    return splitPn[splitPn.length - 1]
  }
  let basename = lastPathName()
  let idx = basename.substr(basename.length - 2, basename.length)
</script>

<PresMd
  filepaths={[
    `/presentation/Regression_MJJD.md`,
    `/presentation/feature_analyse_NAP.md`,
    `/presentation/`
  ]}
>
  <Slide>
    <div class="title-slide">
      <h2>Intelligence Artificielle<br /> Projet - Examen Oral</h2>
      <h3 class="text-blue-900 text-6xl">Regression Logistique <br /> & Naive Bayes</h3>
      <p class="author text-4xl">
        Gregory Sedykh, Leandre Catogni,<br /> Noah Peterschmitt, Noah Munz, Michel Donnet
      </p>
      <p class="date text-slate-600 text-4xl">02 Fevrier 2024</p>
    </div>
  </Slide>
  <!--  Content from given Markdown files follows right after -->
</PresMd>
