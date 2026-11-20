"""Tests for CredentialManager and HardwareInspector.

Covers:
- API key masking edge cases (short, None, empty, long)
- Fallback encrypted store round-trip (store/retrieve/delete)
- Keyring-unavailable path forces fallback automatically
- Fallback store encryption (ciphertext != plaintext)
- Key isolation: different config_id values must not alias each other
- Hardware tier classification against mocked HardwareProfile values
- Full HardwareInspector.detect() and recommend() with nvidia-smi mocked
- detect() graceful degradation when subprocesses fail
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

from app.security.credentials import (
    CredentialManager,
    get_credential_manager,
    mask_api_key,
    reset_credential_manager,
)


# ── 1. mask_api_key ─────────────────────────────────────────────────────────

class TestMaskApiKey:
    def test_mask_long_key(self):
        key = "AIzaSyD_abc123XYZ789Q_extra_chars"
        masked = CredentialManager.mask_api_key(key)
        assert masked.startswith("AIza")
        assert "..." in masked
        assert masked.endswith(key[-4:])
        # Plaintext middle must not appear
        assert "abc123XYZ789Q" not in masked

    def test_mask_short_key(self):
        # Short key: first char + stars
        key = "sk-short"
        masked = CredentialManager.mask_api_key(key)
        assert masked.startswith("s")
        assert "*" in masked
        # Must not contain most of the key
        assert "short" not in masked

    def test_mask_none(self):
        assert CredentialManager.mask_api_key(None) == "<not set>"

    def test_mask_empty(self):
        assert CredentialManager.mask_api_key("") == "<not set>"

    def test_mask_exact_boundary(self):
        # Exactly 11 chars (4 prefix + 3 dots + 4 suffix) → still masked
        key = "ABCD_MID_XYZ"  # 12 chars
        masked = CredentialManager.mask_api_key(key)
        assert "ABCD" in masked
        assert "_XYZ" in masked

    def test_mask_module_function(self):
        # Module-level convenience function
        key = "sk-proj-abcdefghij1234567890"
        masked = mask_api_key(key)
        assert "..." in masked
        assert masked.startswith("sk-p")

    def test_mask_does_not_equal_plaintext(self):
        key = "sk-prod-SuperSecretApiKey1234567890"
        masked = CredentialManager.mask_api_key(key)
        assert masked != key
        assert len(masked) < len(key)


# ── 2. Fallback encrypted store ──────────────────────────────────────────────

@pytest.fixture
def fallback_cm(tmp_path: Path):
    """CredentialManager forced to use the encrypted file fallback."""
    cm = CredentialManager(service_name="QuantivaTest", fallback_dir=tmp_path)
    # Force fallback mode regardless of OS keyring availability
    cm._keyring_ok = False
    return cm


class TestFallbackEncryptedStore:
    def test_set_and_get_round_trip(self, fallback_cm: CredentialManager):
        fallback_cm.set_api_key("cfg_openai", "sk-test-12345ABCDE")
        retrieved = fallback_cm.get_api_key("cfg_openai")
        assert retrieved == "sk-test-12345ABCDE"

    def test_ciphertext_is_not_plaintext(self, fallback_cm: CredentialManager, tmp_path: Path):
        fallback_cm.set_api_key("cfg_gemini", "AIzaSySecret_Key_Here")
        store_file = tmp_path / ".creds_store.enc"
        assert store_file.is_file()
        raw_bytes = store_file.read_bytes()
        # Plaintext key must not appear in the encrypted file
        assert b"AIzaSySecret_Key_Here" not in raw_bytes
        assert b"cfg_gemini" not in raw_bytes

    def test_delete_removes_key(self, fallback_cm: CredentialManager):
        fallback_cm.set_api_key("cfg_tmp", "tmp_key_value_here")
        assert fallback_cm.get_api_key("cfg_tmp") == "tmp_key_value_here"
        deleted = fallback_cm.delete_api_key("cfg_tmp")
        assert deleted is True
        assert fallback_cm.get_api_key("cfg_tmp") is None

    def test_delete_nonexistent_returns_false(self, fallback_cm: CredentialManager):
        result = fallback_cm.delete_api_key("cfg_does_not_exist_xyz")
        assert result is False

    def test_key_isolation_different_config_ids(self, fallback_cm: CredentialManager):
        """Different config_id values must never alias each other."""
        fallback_cm.set_api_key("cfg_a", "key_for_a_111")
        fallback_cm.set_api_key("cfg_b", "key_for_b_222")
        assert fallback_cm.get_api_key("cfg_a") == "key_for_a_111"
        assert fallback_cm.get_api_key("cfg_b") == "key_for_b_222"
        assert fallback_cm.get_api_key("cfg_a") != fallback_cm.get_api_key("cfg_b")

    def test_missing_key_returns_none(self, fallback_cm: CredentialManager):
        assert fallback_cm.get_api_key("cfg_nonexistent") is None

    def test_overwrite_key(self, fallback_cm: CredentialManager):
        fallback_cm.set_api_key("cfg_update", "first_value_xyz")
        fallback_cm.set_api_key("cfg_update", "second_value_abc")
        assert fallback_cm.get_api_key("cfg_update") == "second_value_abc"

    def test_multiple_keys_persist(self, fallback_cm: CredentialManager):
        keys = {f"cfg_{i}": f"key_value_{i}_abcdef" for i in range(5)}
        for cid, k in keys.items():
            fallback_cm.set_api_key(cid, k)
        for cid, k in keys.items():
            assert fallback_cm.get_api_key(cid) == k

    def test_empty_config_id_raises(self, fallback_cm: CredentialManager):
        with pytest.raises(ValueError):
            fallback_cm.set_api_key("", "some_key")

    def test_empty_api_key_raises(self, fallback_cm: CredentialManager):
        with pytest.raises(ValueError):
            fallback_cm.set_api_key("cfg_x", "")

    def test_has_api_key(self, fallback_cm: CredentialManager):
        assert fallback_cm.has_api_key("cfg_new") is False
        fallback_cm.set_api_key("cfg_new", "key_value_12345")
        assert fallback_cm.has_api_key("cfg_new") is True

    def test_list_config_ids(self, fallback_cm: CredentialManager):
        fallback_cm.set_api_key("cfg_list_a", "val_a")
        fallback_cm.set_api_key("cfg_list_b", "val_b")
        ids = fallback_cm.list_config_ids()
        assert "cfg_list_a" in ids
        assert "cfg_list_b" in ids

    def test_thread_safety_concurrent_writes(self, fallback_cm: CredentialManager):
        """Concurrent writes from multiple threads must not corrupt the store."""
        errors: list[Exception] = []

        def writer(idx: int):
            try:
                for _ in range(10):
                    cid = f"thread_key_{idx}"
                    fallback_cm.set_api_key(cid, f"thread_{idx}_value")
                    retrieved = fallback_cm.get_api_key(cid)
                    assert retrieved == f"thread_{idx}_value"
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"


# ── 3. Encryption key survives re-instantiation ──────────────────────────────

def test_encrypted_store_survives_reinstantiation(tmp_path: Path):
    """A fresh CredentialManager on the same fallback_dir must decrypt stored keys."""
    cm1 = CredentialManager(service_name="QuantivaTest", fallback_dir=tmp_path)
    cm1._keyring_ok = False
    cm1.set_api_key("cfg_persist", "persisted_secret_xyz_123")

    # Entirely new instance pointing to the same directory
    cm2 = CredentialManager(service_name="QuantivaTest", fallback_dir=tmp_path)
    cm2._keyring_ok = False
    assert cm2.get_api_key("cfg_persist") == "persisted_secret_xyz_123"


# ── 4. Keyring path (mocked) ─────────────────────────────────────────────────

def test_keyring_path_set_get_delete(tmp_path: Path):
    """When keyring works, set/get/delete delegate to keyring."""
    cm = CredentialManager(service_name="QuantivaTest", fallback_dir=tmp_path)
    cm._keyring_ok = True  # Force keyring path

    _store: dict[str, str] = {}

    def mock_set(service, username, password):
        _store[(service, username)] = password

    def mock_get(service, username):
        return _store.get((service, username))

    def mock_delete(service, username):
        import keyring.errors
        if (service, username) not in _store:
            raise keyring.errors.PasswordDeleteError("not found")
        del _store[(service, username)]

    with patch("keyring.set_password", side_effect=mock_set), \
         patch("keyring.get_password", side_effect=mock_get), \
         patch("keyring.delete_password", side_effect=mock_delete):
        cm.set_api_key("cfg_kr", "kr_secret_value_abcdef")
        assert cm.get_api_key("cfg_kr") == "kr_secret_value_abcdef"
        assert cm.delete_api_key("cfg_kr") is True
        assert cm.get_api_key("cfg_kr") is None


# ── 5. Singleton ─────────────────────────────────────────────────────────────

def test_singleton_and_reset(tmp_path: Path):
    reset_credential_manager()
    c1 = get_credential_manager(fallback_dir=tmp_path)
    c2 = get_credential_manager()
    assert c1 is c2
    reset_credential_manager()
    c3 = get_credential_manager(fallback_dir=tmp_path)
    assert c3 is not c1
    reset_credential_manager()


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

from app.core.hardware import (
    GPUInfo,
    HardwareInspector,
    HardwareProfile,
    ModelTierRecommendation,
    _classify_tier,
)


def _make_profile(
    ram_gb: float,
    vram_mb: int = 0,
    disk_free_gb: float = 100.0,
    cpu_model: str = "Test CPU",
) -> HardwareProfile:
    gpus = [GPUInfo(name="NVIDIA Test GPU", vram_total_mb=vram_mb)] if vram_mb > 0 else []
    return HardwareProfile(
        cpu_model=cpu_model,
        physical_cores=4,
        logical_cores=8,
        ram_total_gb=ram_gb,
        ram_available_gb=ram_gb * 0.5,
        disk_free_gb=disk_free_gb,
        gpus=gpus,
    )


# ── 6. Tier classification ───────────────────────────────────────────────────

class TestTierClassification:
    def test_cloud_api_tier_very_low_ram(self):
        profile = _make_profile(ram_gb=2.0, vram_mb=0)
        rec = _classify_tier(profile)
        assert rec.tier == "cloud_api"
        assert rec.llm_models == []

    def test_ultra_light_tier_low_ram_no_gpu(self):
        profile = _make_profile(ram_gb=6.0, vram_mb=0)
        rec = _classify_tier(profile)
        assert rec.tier == "ultra_light"
        assert any("1.5b" in m or "1b" in m for m in rec.llm_models)
        assert any("MiniLM" in m or "minilm" in m.lower() for m in rec.embedding_models)

    def test_ultra_light_tier_low_vram(self):
        profile = _make_profile(ram_gb=8.0, vram_mb=2 * 1024)  # 2 GB VRAM
        rec = _classify_tier(profile)
        assert rec.tier == "ultra_light"

    def test_balanced_tier_mid_ram_no_gpu(self):
        profile = _make_profile(ram_gb=12.0, vram_mb=0)
        rec = _classify_tier(profile)
        assert rec.tier == "balanced"
        assert any("3b" in m.lower() or "7b" in m.lower() for m in rec.llm_models)
        assert any("e5-small" in m.lower() for m in rec.embedding_models)

    def test_balanced_tier_mid_vram(self):
        profile = _make_profile(ram_gb=16.0, vram_mb=6 * 1024)
        rec = _classify_tier(profile)
        assert rec.tier == "balanced"

    def test_powerful_tier_high_ram_high_vram(self):
        """16+ GB RAM with 8+ GB VRAM should give powerful tier."""
        profile = _make_profile(ram_gb=32.0, vram_mb=8 * 1024)
        rec = _classify_tier(profile)
        assert rec.tier == "powerful"
        assert any("7b" in m.lower() or "14b" in m.lower() for m in rec.llm_models)
        assert any("e5-base" in m.lower() for m in rec.embedding_models)
        assert any("ms-marco" in m.lower() or "MiniLM" in m for m in rec.reranker_models)

    def test_powerful_tier_exact_vram_boundary(self):
        # Exactly 8 GB VRAM should be powerful
        profile = _make_profile(ram_gb=16.0, vram_mb=8 * 1024)
        rec = _classify_tier(profile)
        assert rec.tier == "powerful"

    def test_balanced_just_below_powerful_vram(self):
        # 7 GB VRAM, 16 GB RAM  → balanced (< 8 GB VRAM)
        profile = _make_profile(ram_gb=16.0, vram_mb=7 * 1024)
        rec = _classify_tier(profile)
        assert rec.tier == "balanced"

    def test_recommendation_has_rationale(self):
        for ram, vram in [(2, 0), (6, 0), (12, 0), (16, 8192), (32, 10240)]:
            profile = _make_profile(ram_gb=float(ram), vram_mb=vram)
            rec = _classify_tier(profile)
            assert rec.rationale, f"Empty rationale for ram={ram}, vram={vram}"

    def test_all_tiers_have_embedding_models(self):
        cases = [(2, 0), (6, 0), (12, 0), (16, 8192)]
        for ram, vram in cases:
            profile = _make_profile(ram_gb=float(ram), vram_mb=vram)
            rec = _classify_tier(profile)
            assert len(rec.embedding_models) > 0, f"No embeddings for tier={rec.tier}"

    def test_powerful_tier_has_reranker(self):
        profile = _make_profile(ram_gb=32.0, vram_mb=10 * 1024)
        rec = _classify_tier(profile)
        assert rec.tier == "powerful"
        assert rec.reranker_models

    def test_non_powerful_tiers_no_reranker(self):
        for ram, vram in [(6, 0), (12, 0)]:
            profile = _make_profile(ram_gb=float(ram), vram_mb=vram)
            rec = _classify_tier(profile)
            assert rec.tier in ("ultra_light", "balanced", "cloud_api")
            assert not rec.reranker_models, f"Unexpected reranker for {rec.tier}"


# ── 7. HardwareInspector.detect() ───────────────────────────────────────────

class TestHardwareInspectorDetect:
    def test_detect_returns_hardware_profile(self, tmp_path: Path):
        inspector = HardwareInspector(data_dir=tmp_path)
        profile = inspector.detect()
        assert isinstance(profile, HardwareProfile)
        assert profile.ram_total_gb > 0
        assert profile.physical_cores >= 1
        assert profile.logical_cores >= profile.physical_cores
        assert profile.disk_free_gb >= 0

    def test_detect_with_nvidia_smi_mocked(self, tmp_path: Path):
        """Verify nvidia-smi output is parsed correctly."""
        mock_smi_output = "NVIDIA GeForce RTX 4060, 8188, 6000\n"

        def fake_run(cmd, **kwargs):
            m = MagicMock()
            m.returncode = 0
            m.stdout = mock_smi_output
            return m

        with patch("subprocess.run", side_effect=fake_run), \
             patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            inspector = HardwareInspector(data_dir=tmp_path)
            profile = inspector.detect()

        assert len(profile.gpus) == 1
        gpu = profile.gpus[0]
        assert "RTX 4060" in gpu.name
        assert gpu.vram_total_mb == 8188
        assert gpu.vram_total_gb == pytest.approx(7.99, abs=0.1)
        assert gpu.vram_free_mb == 6000

    def test_detect_no_nvidia_smi_no_gpu(self, tmp_path: Path):
        """When nvidia-smi is absent and torch unavailable, gpus list should be empty."""
        with patch("shutil.which", return_value=None), \
             patch("app.core.hardware._detect_gpus_torch", return_value=[]):
            inspector = HardwareInspector(data_dir=tmp_path)
            profile = inspector.detect()
        assert profile.gpus == []
        assert not profile.has_gpu

    def test_detect_graceful_on_subprocess_error(self, tmp_path: Path):
        """Errors in GPU detection must not raise; they appear in detection_errors."""
        def raise_error(*args, **kwargs):
            raise OSError("subprocess blocked")

        with patch("subprocess.run", side_effect=raise_error), \
             patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("app.core.hardware._detect_gpus_torch", return_value=[]):
            inspector = HardwareInspector(data_dir=tmp_path)
            profile = inspector.detect()
        # Should not raise; gpus may be empty
        assert isinstance(profile, HardwareProfile)

    def test_recommend_uses_detected_profile(self, tmp_path: Path):
        """Inspector.recommend() without arg calls detect() internally."""
        inspector = HardwareInspector(data_dir=tmp_path)
        rec = inspector.recommend()
        assert isinstance(rec, ModelTierRecommendation)
        assert rec.tier in ("cloud_api", "ultra_light", "balanced", "powerful")

    def test_full_report_structure(self, tmp_path: Path):
        inspector = HardwareInspector(data_dir=tmp_path)
        report = inspector.full_report()
        assert "hardware" in report
        assert "recommendation" in report
        hw = report["hardware"]
        assert "ram_total_gb" in hw
        assert "gpus" in hw
        rec = report["recommendation"]
        assert "tier" in rec
        assert "llm_models" in rec
        assert "embedding_models" in rec
        assert "reranker_models" in rec
        assert "rationale" in rec

    def test_gpu_info_vram_gb_property(self):
        gpu = GPUInfo(name="Test GPU", vram_total_mb=8192, vram_free_mb=4096)
        assert gpu.vram_total_gb == pytest.approx(8.0, abs=0.01)

    def test_hardware_profile_to_dict(self, tmp_path: Path):
        inspector = HardwareInspector(data_dir=tmp_path)
        profile = inspector.detect()
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["gpus"], list)
        assert isinstance(d["has_gpu"], bool)
        assert isinstance(d["ram_total_gb"], float)

    def test_tier_recommendation_to_dict(self):
        profile = _make_profile(ram_gb=32.0, vram_mb=8192)
        rec = _classify_tier(profile)
        d = rec.to_dict()
        assert d["tier"] == "powerful"
        assert isinstance(d["llm_models"], list)
        assert isinstance(d["embedding_models"], list)
        assert isinstance(d["reranker_models"], list)

    def test_multiple_gpus_aggregated_vram(self):
        profile = HardwareProfile(
            cpu_model="Dual GPU Server",
            physical_cores=16,
            logical_cores=32,
            ram_total_gb=128.0,
            ram_available_gb=64.0,
            disk_free_gb=1000.0,
            gpus=[
                GPUInfo(name="GPU A", vram_total_mb=8192),
                GPUInfo(name="GPU B", vram_total_mb=8192),
            ],
        )
        assert profile.total_vram_mb == 16384
        assert profile.total_vram_gb == pytest.approx(16.0, abs=0.01)
        rec = _classify_tier(profile)
        assert rec.tier == "powerful"
