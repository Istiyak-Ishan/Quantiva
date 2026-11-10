import { expect, test, Page } from '@playwright/test';

const sessionId = '11111111-1111-4111-8111-111111111111';

async function mockApi(page: Page, options: { signedIn?: boolean; streamError?: boolean; sessionError?: boolean } = {}) {
  let signedIn = options.signedIn ?? false;
  await page.route('**/api/v1/**', async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const json = (data: unknown, status = 200, headers = {}) => route.fulfill({ status, contentType: 'application/json', headers, body: JSON.stringify(data) });
    if (path.endsWith('/auth/login') || path.endsWith('/auth/register')) {
      signedIn = true;
      return json({}, 200, { 'set-cookie': 'quantiva_access=browser-test; HttpOnly; SameSite=Lax; Path=/api/v1' });
    }
    if (path.endsWith('/auth/logout')) {
      signedIn = false;
      return route.fulfill({ status: 204, body: '', headers: { 'set-cookie': 'quantiva_access=; HttpOnly; Path=/api/v1; Max-Age=0' } });
    }
    if (path.endsWith('/auth/me')) return json(signedIn ? { id: 'owner', email: 'browser@example.com', tier: 'free' } : { detail: 'Not authenticated' }, signedIn ? 200 : 401);
    if (path.endsWith('/auth/refresh')) return json({ detail: 'Invalid refresh token' }, 401);
    if (!signedIn) return json({ detail: 'Not authenticated' }, 401);
    if (path.endsWith('/chat/sessions') && route.request().method() === 'POST') return options.sessionError ? json({ detail: 'Session service unavailable' }, 503) : json({ session_id: sessionId });
    if (path.endsWith('/chat/sessions')) return json([{ id: sessionId, title: 'Test conversation', created_at: new Date().toISOString() }]);
    if (path.endsWith('/history') || path.endsWith('/documents')) return json([]);
    if (path.endsWith('/message/stream')) {
      expect(url.searchParams.has('token')).toBe(false);
      expect(url.searchParams.get('session_id')).toBe(sessionId);
      const body = options.streamError
        ? 'event: error\ndata: {"error":"Provider unavailable","done":true}\n\n'
        : 'data: {"token":"**Grounded answer**\\n\\n| A | B |\\n| - | - |\\n| 1 | 2 |","done":false}\n\ndata: {"token":"","done":true,"answer":"**Grounded answer**\\n\\n| A | B |\\n| - | - |\\n| 1 | 2 |","confidence":0.8,"citations":[]}\n\n';
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body });
    }
    return json({ detail: 'Unexpected test request' }, 404);
  });
}

test('anonymous dashboard redirects to login without rendering private workspace', async ({ page }) => {
  await mockApi(page);
  await page.goto(`/chat/${sessionId}`);
  await expect(page.getByRole('heading', { name: 'Quantiva · Sign in' })).toBeVisible();
  await expect(page.getByLabel('Chat messages')).toHaveCount(0);
});

test('login uses HttpOnly cookies, sends no URL credentials, and renders Markdown tables', async ({ page, context }) => {
  await mockApi(page);
  await page.goto('/login');
  await page.getByLabel('Email', { exact: true }).fill('browser@example.com');
  await page.getByLabel('Password', { exact: true }).fill('SecurePass123!');
  await page.getByRole('button', { name: 'Sign in', exact: true }).click();
  await expect(page).toHaveURL(`/chat/${sessionId}`);
  await expect(page.getByText('Quantiva Assistant', { exact: true })).toBeVisible();
  expect((await context.cookies()).find((c) => c.name === 'quantiva_access')?.httpOnly).toBe(true);
  expect(await page.evaluate(() => localStorage.getItem('token'))).toBeNull();
  const input = page.getByRole('textbox', { name: 'Chat message input' });
  await input.fill('hello');
  await input.press('Enter');
  await expect(page.locator('strong', { hasText: 'Grounded answer' })).toBeVisible();
  await expect(page.getByRole('table')).toBeVisible();
});

test('registration form creates a real API session path rather than a fallback ID', async ({ page }) => {
  await mockApi(page);
  await page.goto('/register');
  await page.getByLabel('Email', { exact: true }).fill('new@example.com');
  await page.getByLabel('Password', { exact: true }).fill('SecurePass123!');
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await expect(page).toHaveURL(`/chat/${sessionId}`);
});

test('session creation failure stays visible and never fabricates a route', async ({ page }) => {
  await mockApi(page, { signedIn: true, sessionError: true });
  await page.goto('/');
  await expect(page.getByRole('alert').filter({ hasText: 'Session service unavailable' })).toBeVisible();
  await expect(page).toHaveURL('/');
});

test('provider stream error is displayed, not finalized as a successful assistant answer', async ({ page }) => {
  await mockApi(page, { signedIn: true, streamError: true });
  await page.goto(`/chat/${sessionId}`);
  const input = page.getByRole('textbox', { name: 'Chat message input' });
  await input.fill('hello');
  await input.press('Enter');
  await expect(page.getByRole('alert').filter({ hasText: 'Provider unavailable' })).toBeVisible();
  await expect(page.getByText('Analysis complete.')).toHaveCount(0);
});

test('logout removes workspace and routes back to login', async ({ page }) => {
  await mockApi(page, { signedIn: true });
  await page.goto(`/chat/${sessionId}`);
  await expect(page.getByRole('button', { name: 'Sign out' })).toBeVisible();
  await page.getByRole('button', { name: 'Sign out' }).click({ timeout: 5000, noWaitAfter: true });
  await expect(page).toHaveURL('/login');
});
