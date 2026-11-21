// Match the Docker standalone runtime or static export runtime. No dev overlay.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = path.join(root, 'out');
const standalone = path.join(root, '.next', 'standalone');
const server = [
  path.join(standalone, 'server.js'),
  path.join(standalone, 'frontend', 'server.js'),
].find(fs.existsSync);

if (fs.existsSync(outDir)) {
  const port = parseInt(process.env.PORT || '3000', 10);
  const hostname = process.env.HOSTNAME || '127.0.0.1';

  const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.webp': 'image/webp',
    '.woff2': 'font/woff2',
    '.woff': 'font/woff',
    '.ttf': 'font/ttf',
  };

  const httpServer = http.createServer((req, res) => {
    try {
      const parsedUrl = new URL(req.url || '/', `http://${hostname}:${port}`);
      let pathname = decodeURIComponent(parsedUrl.pathname);

      let filePath = path.join(outDir, pathname);

      if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
        filePath = path.join(filePath, 'index.html');
      }

      if (!fs.existsSync(filePath)) {
        if (fs.existsSync(filePath + '.html')) {
          filePath = filePath + '.html';
        } else if (fs.existsSync(path.join(outDir, '404.html'))) {
          filePath = path.join(outDir, '404.html');
          res.statusCode = 404;
        } else if (fs.existsSync(path.join(outDir, '_not-found', 'index.html'))) {
          filePath = path.join(outDir, '_not-found', 'index.html');
          res.statusCode = 404;
        } else {
          filePath = path.join(outDir, 'index.html');
        }
      }

      const ext = path.extname(filePath).toLowerCase();
      res.setHeader('Content-Type', MIME_TYPES[ext] || 'application/octet-stream');
      fs.createReadStream(filePath).pipe(res);
    } catch {
      res.statusCode = 500;
      res.end('Internal Server Error');
    }
  });

  httpServer.listen(port, hostname, () => {
    console.log(`Serving static export from ${outDir} at http://${hostname}:${port}`);
  });
} else if (server) {
  const directory = path.dirname(server);
  if (fs.existsSync(path.join(root, 'public'))) {
    fs.cpSync(path.join(root, 'public'), path.join(directory, 'public'), { recursive: true });
  }
  fs.cpSync(path.join(root, '.next', 'static'), path.join(directory, '.next', 'static'), { recursive: true });
  process.env.HOSTNAME ??= '127.0.0.1';
  process.env.PORT ??= '3000';
  await import(pathToFileURL(server).href);
} else {
  throw new Error('Production build missing. Run npm run build first.');
}
