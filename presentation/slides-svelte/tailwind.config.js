/** @type {import('tailwindcss').Config}*/
const config = {
	content: ['./src/**/*.{html,js,svelte,ts,md,markdown}'],

	theme: {
		extend: {
			fontFamily: {
				sans: ['Roboto', 'Geneva', 'Helvetica', 'ui-sans-serif', 'system-ui']
			}
		}
	},

	plugins: [],
	important: true
}

export default config
