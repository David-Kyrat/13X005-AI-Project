<script>
	import Drawer, {  Content, Header, Title, Subtitle, Scrim } from '@smui/drawer'

	import List, { Item, Text } from '@smui/list'

	export let title = 'Navigation Menu'
	export let subtitle = 'Select another question or go home.'
	let oral_question_nb = 21
	function fmtIdx(i) {
		return i < 10 ? `0${i}` : `${i}`
	}

	function buildDrawerContent() {
		let questions = [{ path: '/', text: '' }]
		for (let i = 1; i <= oral_question_nb + 1; i++) {
			let fmt = fmtIdx(i)
			questions.push({ path: `/q${fmt}`, text: `Q${fmt}` })
		}
		return questions
	}
	export let content = buildDrawerContent()

	export let open = false
	export let fun
	let active = 'Inbox'
	$: fun(open)
	function setActive(value) {
		active = value
		open = false
	}
	$: zindex = open ? 'drawer-container z-50' : 'drawer-container z-0'
</script>

<!-- <div class="drawer-container z-50"> -->
<div class={zindex}>
	<Drawer variant="modal" bind:open class="bg-white">
		<Header>
			<Title>{title}</Title>
			<Subtitle>{subtitle}</Subtitle>
		</Header>
		<Content>
			<List>
				{#each content as item}
					<Item
						class="px-7 text-base font-bold"
						href={item.path}
            target="_top"
						on:click={() => setActive(item)}
						activated={active === item}
					>
						{#if item.path === '/'}
							<span class="material-icons">home</span>
						{:else}
							<Text>{item.text}</Text>
						{/if}
					</Item>
				{/each}
			</List>
		</Content>
	</Drawer>
	<Scrim />
</div>

<style>
	/* These classes are only needed because the
    drawer is in a container on the page. */
	.drawer-container {
		position: absolute;
		display: flex;
		height: 350px;
		overflow: hidden;
		z-index: 0;
	}

	* :global(.app-content) {
		flex: auto;
		position: relative;
		flex-grow: 1;
	}

	.main-content {
		overflow: auto;
		padding: 16px;
		height: 100%;
		box-sizing: border-box;
	}
</style>
