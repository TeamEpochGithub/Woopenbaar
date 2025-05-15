/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Outfit', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      colors: {
        primary: '#4B4B4B',
        'primary-light': '#4B4B4B',
        'primary-dark': '#4B4B4B',
        accent: '#4B4B4B',
        'accent-light': '#f1f5f9',
        'accent-dark': '#334155',
      },
      maxWidth: {
        '3/4': '75%',
      },
    },
  },
  plugins: [],
} 