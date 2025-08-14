import { defineConfig } from "vite";

export default defineConfig({
  build: {
    sourcemap: true,
    target: 'es2022',
    outDir: 'dist',
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Split LangChain packages into separate chunks
          if (id.includes('@langchain/openai')) {
            return 'langchain-openai';
          }
          if (id.includes('@langchain/anthropic')) {
            return 'langchain-anthropic';
          }
          if (id.includes('@langchain/google-genai')) {
            return 'langchain-google';
          }
          // Split large vendor libraries
          if (id.includes('@langchain/core')) {
            return 'langchain-core';
          }
          if (id.includes('zod')) {
            return 'zod';
          }
          // Other vendor packages
          if (id.includes('node_modules')) {
            return 'vendor';
          }
        }
      }
    }
  },
  esbuild: {
    target: 'es2022'
  }
});
