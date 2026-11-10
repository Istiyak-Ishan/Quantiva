import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { baseURL: 'http://localhost:3100', browserName: 'chromium', headless: true, screenshot: 'only-on-failure' },
  webServer: { command: 'npm run start', env: { PORT: '3100', HOSTNAME: '127.0.0.1' }, url: 'http://localhost:3100/login', reuseExistingServer: false, timeout: 120000 },
  workers: 1,
});
