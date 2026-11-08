import pytest
from datetime import timedelta
from app.core.security import hash_password, verify_password, create_access_token, decode_token

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("mysecret")
        assert hashed != "mysecret"

    def test_correct_password_verifies(self):
        hashed = hash_password("correct-horse-battery")
        assert verify_password("correct-horse-battery", hashed) is True

    def test_wrong_password_rejected(self):
        hashed = hash_password("correct-horse-battery")
        assert verify_password("wrong-password", hashed) is False

    def test_same_password_different_hashes(self):
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2
        assert verify_password("same-password", h1)
        assert verify_password("same-password", h2)

class TestJWT:
    def test_token_created_and_decoded(self):
        token = create_access_token(user_id="user-abc", expires_delta=timedelta(minutes=5))
        assert isinstance(token, str)
        assert len(token) > 20
        payload = decode_token(token)
        assert payload["sub"] == "user-abc"

    def test_expired_token_rejected(self):
        token = create_access_token(user_id="user-xyz", expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception):
            decode_token(token)

    def test_tampered_token_rejected(self):
        token = create_access_token(user_id="user-abc", expires_delta=timedelta(minutes=5))
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            decode_token(tampered)

