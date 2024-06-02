/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./pages/**/*.{html,js}"],
  theme: {
    extend: {},
  },
  daisyui: {
    themes: "light",
    base: false,
    styled: false,
    utils: true,
  },
  plugins: [require('daisyui')],
}

