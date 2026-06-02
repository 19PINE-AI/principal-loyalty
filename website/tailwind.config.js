/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#1a1a2e",
        leak: "#dc2626",
        bound: "#ea580c",
        capit: "#d97706",
        post: "#ca8a04",
        author: "#65a30d",
        moder: "#0891b2",
        sanity: "#2563eb",
        calibrated: "#16a34a",
        intermediate: "#ca8a04",
        overrefuse: "#dc2626",
        mech1: "#7c3aed",
        mech2: "#0891b2",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        serif: ["ui-serif", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
}
