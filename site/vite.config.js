import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads ../level-*/results*.json and the level write-ups at build time,
// so every number on the page is generated from the repo, never typed.
export default defineConfig({
  base: '/cuda-kernels-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
