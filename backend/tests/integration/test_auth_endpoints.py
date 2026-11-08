import pytest

class TestRegister:
    async def test_register_success(self, client):
        resp = await client.post("/api/v1/auth/register", json={
            "email": "newuser@example.com",
            "password": "SecurePass1!",
        })
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_register_duplicate_email(self, client):
        payload = {"email": "dup@example.com", "password": "Pass1234!"}
        await client.post("/api/v1/auth/register", json=payload)
        resp = await client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 400
        assert "already" in resp.json()["detail"].lower()

    async def test_register_missing_password(self, client):
        resp = await client.post("/api/v1/auth/register", json={"email": "a@b.com"})
        assert resp.status_code == 422

class TestLogin:
    async def test_login_success(self, client):
        email, pw = "logintest@example.com", "LoginPass1!"
        await client.post("/api/v1/auth/register", json={"email": email, "password": pw})
        resp = await client.post("/api/v1/auth/login", json={"email": email, "password": pw})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_login_wrong_password(self, client):
        email, pw = "wrongpw@example.com", "CorrectPass1!"
        await client.post("/api/v1/auth/register", json={"email": email, "password": pw})
        resp = await client.post("/api/v1/auth/login", json={"email": email, "password": "WrongPass!"})
        assert resp.status_code == 401

    async def test_login_unknown_email(self, client):
        resp = await client.post("/api/v1/auth/login", json={
            "email": "nobody@nowhere.com", "password": "whatever"
        })
        assert resp.status_code == 401

class TestMe:
    async def test_me_returns_email(self, client, auth_headers):
        resp = await client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert "email" in resp.json()

    async def test_me_unauthenticated(self, client):
        resp = await client.get("/api/v1/auth/me")
        assert resp.status_code == 401

