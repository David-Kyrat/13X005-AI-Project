<script>
	// @ts-nocheck
	import Slide from './slide.svelte'
	import Code from './code.svelte'
	import Markdown from './markdown.svelte'
</script>

<Slide>
	<p class="capitalize">Svelte deck 🔥</p>
</Slide>

<section data-auto-animate>
	<h2>HStack Example</h2>
	<div class="r-hstack">
		<p style="padding: 0.50em; background: #eee; margin: 0.25em">One</p>
		<p style="padding: 0.75em; background: #eee; margin: 0.25em">Two</p>
		<p style="padding: 1.00em; background: #eee; margin: 0.25em">Three</p>
	</div>
</section>

<section id="vstack">
	<h2>VStack</h2>
	<p>Stacks multiple elements vertically.</p>
	<pre><code class="html" data-trim data-line-numbers>
    <div class="r-vstack">
      &lt;img width="450" height="300" src="..."&gt;
      &lt;img width="300" height="450" src="..."&gt;
      &lt;img width="400" height="400" src="..."&gt;
    </div>
  </code></pre>
</section>

<section data-auto-animate>
	<h2>VStack Example</h2>
	<div class="r-hstack flex justify-between">
		<div class="r-vstack">
			<p style="padding: 0.50em; background: #eee; margin: 0.25em">one</p>
			<p style="padding: 0.75em; background: #eee; margin: 0.25em">two</p>
			<p style="padding: 0.50em; background: #eee; margin: 0.25em">one</p>
			<p style="padding: 0.50em; background: #eee; margin: 0.25em">one</p>
			<p style="padding: 0.75em; background: #eee; margin: 0.25em">two</p>
			<p style="padding: 0.75em; background: #eee; margin: 0.25em">two</p>
		</div>
		<div class="r-vstack ml-50 pl-50">
			<p style="padding: 0.50em; background: #eee; margin: 0.25em">three</p>
      <img  class="border border-black relative right-0"   src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Schematic_diagram_of_the_human_eye_en.svg/1920px-Schematic_diagram_of_the_human_eye_en.svg.png" alt="">
			<p style="padding: 0.75em; background: #eee; margin: 0.25em">four</p>
		</div>
	</div>
</section>

<Slide id="an1" animate>
	<p class="capitalize">Across slides</p>
</Slide>

<Slide id="an1" animate>
	<p class="capitalize">Animate elements</p>
	<p class="capitalize">Across slides</p>
</Slide>

<Markdown name="/example.md" external clas="" />
<Slide>
	<p class="capitalize">Horizontal slide</p>
</Slide>

<Slide>
	<Slide>
		<p class="capitalize">Vertical slide 1</p>
	</Slide>
	<Slide>
		<p class="capitalize">Vertical slide 2</p>
	</Slide>
</Slide>

<Slide id="animate" animate>
	<p class="capitalize">Animate elements</p>
</Slide>

<Slide id="animate" animate>
	<p class="capitalize">Animate elements</p>
	<p class="capitalize">Across slides</p>
</Slide>

<Slide id="list" animate>
	<ul>
		<li>React</li>
		<li>Solid</li>
		<li>Svelte</li>
		<li>Vue</li>
	</ul>
</Slide>

<Slide id="list" animate>
	<ul>
		<li>Svelte</li>
		<li>React</li>
		<li>Solid</li>
		<li>Vue</li>
	</ul>
</Slide>

<Slide id="list" animate>
	<ul>
		<li>Svelte ❤️</li>
		<li>React</li>
		<li>Solid</li>
		<li>Vue</li>
	</ul>
</Slide>

<Slide>
	<p class="capitalize">Code Blocks</p>
</Slide>

<Slide animate>
	<Code id="code" lines="1|3">
		{`
			<Slide animate>
				<!-- .. -->
			</Slide>
		`}
	</Code>
</Slide>

<Slide animate>
	<Code id="code" lines="2|4">
		{`
			<Slide animate>
				<Code lines>
					<!-- ... -->
				</Code>
			</Slide>
		`}
	</Code>
</Slide>

<Slide animate>
	<Code id="code" lines="3-5|1-7">
		{`
			<Slide animate>
				<Code lines>
					{code\`
						...
					\`}
				</Code>
			</Slide>
		`}
	</Code>
</Slide>

<Slide id="boxes" animate>
	<p class="capitalize">Cool transitions</p>
	<div class="flex gap-2 place-content-center">
		<div data-id="box-1" class="w-[100px] h-[100px] bg-gray-400" />
		<div data-id="box-2" class="w-[100px] h-[100px] bg-gray-400" />
		<div data-id="box-3" class="w-[100px] h-[100px] bg-gray-400" />
	</div>
</Slide>

<Slide id="boxes" animate>
	<p class="capitalize">Moving boxes</p>
	<div class="flex gap-2 place-content-center">
		<div data-id="box-1" class="w-[200px] h-[200px] bg-red-400" />
		<div data-id="box-2" class="w-[200px] h-[200px] bg-green-400" />
		<div data-id="box-3" class="w-[200px] h-[200px] bg-blue-400" />
	</div>
	<p class="capitalize">Cool transitions</p>
</Slide>

<Slide id="boxes" animate>
	<p>Moving boxes</p>
	<div class="flex gap-2 place-content-center">
		<div data-id="box-1" class="w-[200px] h-[200px] bg-gray-400" />
		<img src="https://i.giphy.com/media/ToMjGpnXBTw7vnokxhu/giphy.webp" alt="Whoa" />
		<div data-id="box-3" class="w-[200px] h-[200px] bg-gray-400" />
	</div>
	<p>Whoa!</p>
</Slide>

<Markdown
	>{`
## Markdown ❤️

You can use **HTML** or **Markdown** for slides.

`}
</Markdown>

<Markdown name="/example.md" external clas="" />

<Slide>
	<p class="capitalize">Get started 😎</p>
</Slide>
