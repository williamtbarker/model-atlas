import { defineConfig } from "vite";
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        landscape: "index.html",
        microscope: "microscope.html",
        architecture: "architecture.html",
      },
    },
  },
});
