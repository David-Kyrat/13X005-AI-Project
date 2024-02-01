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

<PresMd filepaths={[`/md/q01.md`]}>
  <Slide>
    <div class="title-slide">
      <h2>Intelligence Artificielle<br /> Projet - Examen Oral</h2>
      <h3>Regression Logistique <br /> & Naive Bayes</h3>

      <p class="text-blue-900">Question {idx}</p>

      <p class="author">Noah Munz (19-815-489)</p>
      <p class="date text-slate-600">January 30th 2024</p>
    </div>
  </Slide>
  <!--  Content from given Markdown files follows right after -->
</PresMd>
