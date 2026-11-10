// Match the Docker standalone runtime, including static assets. No dev overlay.
import { cpSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const standalone = path.join(root, '.next', 'standalone');
const server = [path.join(standalone, 'server.js'), path.join(standalone, 'frontend', 'server.js')].find(existsSync);
if (!server) throw new Error('Production build missing. Run npm run build first.');
const directory = path.dirname(server);
if (existsSync(path.join(root, 'public'))) cpSync(path.join(root, 'public'), path.join(directory, 'public'), { recursive: true });
cpSync(path.join(root, '.next', 'static'), path.join(directory, '.next', 'static'), { recursive: true });
process.env.HOSTNAME ??= '127.0.0.1';
process.env.PORT ??= '3000';
await import(pathToFileURL(server).href);
