"""Authentication regressions exercised through the real ASGI routes and DB."""
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from app.core.security import create_access_token
from app.core.security import hash_password
from app.models.user import User


async def register(client, email="security@example.com"):
    response = await client.post("/api/v1/auth/register", json={"email": email, "password": "SecurePass123!"})
    assert response.status_code == 200, response.text
    return response


async def test_mock_token_is_rejected_without_querying_or_creating_users(client, db_session):
    for populated in (False, True):
        if populated:
            await register(client)
        response = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer mock-token"})
        assert response.status_code == 401
    users = (await db_session.scalars(select(User))).all()
    assert len(users) == 1
    assert users[0].email == "security@example.com"


async def test_url_credentials_are_not_accepted(client):
    tokens = (await register(client)).json()
    client.cookies.clear()
    response = await client.get("/api/v1/auth/me", params={"token": tokens["access_token"]})
    assert response.status_code == 401


async def test_refresh_cannot_authenticate_as_access(client):
    tokens = (await register(client)).json()
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert response.status_code == 401


async def test_refresh_rotation_replay_and_logout_revoke_access(client):
    original = (await register(client)).json()
    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]})
    assert response.status_code == 200, response.text
    rotated = response.json()
    assert rotated["refresh_token"] != original["refresh_token"]
    replay = await client.post("/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]})
    assert replay.status_code == 401
    logout = await client.post("/api/v1/auth/logout", json={"refresh_token": rotated["refresh_token"]})
    assert logout.status_code == 204
    for token in (original["access_token"], rotated["access_token"]):
        response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 401
    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": rotated["refresh_token"]})
    assert response.status_code == 401


async def test_disabled_user_cannot_login_access_or_refresh(client, db_session):
    tokens = (await register(client)).json()
    user = await db_session.scalar(select(User))
    user.is_active = False
    await db_session.commit()
    response = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert response.status_code == 401
    response = await client.post("/api/v1/auth/login", json={"email": user.email, "password": "SecurePass123!"})
    assert response.status_code == 401
    response = await client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert response.status_code == 401


async def test_cookies_are_httponly_and_mutations_require_trusted_origin(client):
    response = await register(client)
    cookies = response.headers.get_list("set-cookie")
    assert len(cookies) == 2
    assert all("HttpOnly" in cookie and "SameSite=lax" in cookie for cookie in cookies)
    assert response.headers["cache-control"] == "no-store"
    assert (await client.get("/api/v1/auth/me")).status_code == 200
    for headers in ({}, {"Origin": "https://attacker.example"}):
        response = await client.post("/api/v1/chat/sessions", json={"document_ids": []}, headers=headers)
        assert response.status_code == 403
    response = await client.post("/api/v1/chat/sessions", json={"document_ids": []}, headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    response = await client.post("/api/v1/auth/refresh", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200


async def test_login_csrf_blocked(client):
    response = await client.post("/api/v1/auth/login", json={"email": "x@example.com", "password": "x"}, headers={"Origin": "https://attacker.example"})
    assert response.status_code == 403


async def test_legacy_mixed_case_email_still_authenticates_without_data_rewrite(client, db_session):
    legacy = User(email="LegacyUser@example.com", hashed_password=hash_password("LegacyPass123!"))
    db_session.add(legacy)
    await db_session.commit()
    response = await client.post("/api/v1/auth/login", json={"email": "legacyuser@example.com", "password": "LegacyPass123!"})
    assert response.status_code == 200
    await db_session.refresh(legacy)
    assert legacy.email == "LegacyUser@example.com"
    response = await client.post("/api/v1/auth/register", json={"email": "legacyuser@example.com", "password": "OtherPass123!"})
    assert response.status_code == 400


async def test_cors_allows_only_explicit_origin(client):
    for origin, allowed in (("http://localhost:3000", True), ("https://evil.railway.app", False)):
        response = await client.options("/api/v1/chat/sessions", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
        assert (response.headers.get("access-control-allow-origin") == origin) is allowed


@pytest.mark.parametrize("password", ["short", "x" * 73, "é" * 40])
async def test_invalid_password_lengths_rejected(client, password):
    response = await client.post("/api/v1/auth/register", json={"email": "password@example.com", "password": password})
    assert response.status_code == 422
