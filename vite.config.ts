import { defineConfig } from "vite";
export default defineConfig({
  build: {
    rollupOptions: {
      input: { microscope: "index.html", architecture: "architecture.html" },
    },
  },
});
