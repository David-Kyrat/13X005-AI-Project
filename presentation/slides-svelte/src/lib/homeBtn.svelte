<script>
	import Fab, { Icon } from '@smui/fab'
  export let title = null;
</script>

<div class="flex items-baseline">
	<a href="/">
		<Fab color="primary" href="/" target="_top" class="z-50 m-1" mini>
			<Icon class="material-icons" id="tmp">home</Icon>
		</Fab>
	</a>
  {#if title}
    <h1 class="text-2xl w-fit mx-auto block mt-0 font-bold">{title}</h1>
  {/if}
	<!-- <h1 class="text-2xl w-fit mx-auto block mt-0 font-bold">{title}</h1> -->
</div>
