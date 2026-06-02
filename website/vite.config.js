import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Use relative base so the build can be hosted from any subpath
// (GitHub Pages project sites, local file://, etc.).
export default defineConfig({
  plugins: [react()],
  base: './',
})
