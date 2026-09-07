#!/usr/bin/env node
/** Serve the supplied build with Node's standard library; npm install is unnecessary. */
import { createServer } from "node:http";
import { createReadStream, existsSync, statSync } from "node:fs";
import { resolve, extname, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { createGzip } from "node:zlib";
const root = fileURLToPath(new URL("./dist/", import.meta.url));
const port = Number(process.env.ATLAS_PORT ?? 4173);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".gz": "application/octet-stream",
  ".png": "image/png",
  ".svg": "image/svg+xml",
};
if (!existsSync(resolve(root, "index.html"))) {
  console.error(
    "The built viewer is missing. Run npm install and npm run build.",
  );
  process.exit(1);
}
createServer((req, res) => {
  let path;
  try {
    path = decodeURIComponent(
      new URL(req.url ?? "/", `http://127.0.0.1:${port}`).pathname,
    );
  } catch {
    res.writeHead(400);
    res.end("Invalid path");
    return;
  }
  const filename = resolve(
    root,
    "." + (path.endsWith("/") ? path + "index.html" : path),
  );
  if (
    !filename.startsWith(root.endsWith(sep) ? root : root + sep) ||
    !existsSync(filename) ||
    !statSync(filename).isFile()
  ) {
    res.writeHead(404);
    res.end("Not found");
    return;
  }
  const compress =
    /\b(?:html|js|css|json)\b/.test(extname(filename).slice(1)) &&
    String(req.headers["accept-encoding"]).includes("gzip");
  res.writeHead(200, {
    "Content-Type": types[extname(filename)] ?? "application/octet-stream",
    "Cache-Control": "no-cache",
    "X-Content-Type-Options": "nosniff",
    Vary: "Accept-Encoding",
    ...(compress ? { "Content-Encoding": "gzip" } : {}),
  });
  if (req.method === "HEAD") {
    res.end();
    return;
  }
  const stream = createReadStream(filename);
  if (compress) stream.pipe(createGzip()).pipe(res);
  else stream.pipe(res);
})
  .on("error", (e) => {
    console.error(
      e.code === "EADDRINUSE"
        ? `Port ${port} is busy. Try ATLAS_PORT=4174 node serve.mjs`
        : e.message,
    );
    process.exit(1);
  })
  .listen(port, "127.0.0.1", () =>
    console.log(
      `Model Atlas is ready: http://127.0.0.1:${port}\nLeave this terminal open. Press Ctrl+C to stop.`,
    ),
  );
