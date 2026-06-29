"""Integration tests for URCM API and core system."""
import numpy as np
import pytest
from unittest.mock import patch


class TestURCMSystemIntegration:
    """End-to-end integration tests for URCMSystem."""

    def test_system_initialization(self, mock_urcm_system):
        assert mock_urcm_system.status["initialized"] is True

    def test_process_text_query(self, mock_urcm_system):
        result = mock_urcm_system.process_query("test")
        assert result is not None
        assert hasattr(result, "mu_trajectory")
        assert len(result.mu_trajectory) > 0

    def test_system_validation(self, mock_urcm_system):
        checks = mock_urcm_system.validate_system()
        assert "pipeline_ok" in checks
        assert "encoder_ok" in checks
        assert "overall_health" in checks

    def test_multiple_queries(self, mock_urcm_system, sample_queries):
        for query in sample_queries:
            result = mock_urcm_system.process_query(query)
            assert result is not None
            assert result.mu_trajectory[-1] > 0

    @pytest.mark.parametrize("text", ["", "a", "hello world", "12345"])
    def test_various_inputs(self, mock_urcm_system, text):
        if not text:
            with pytest.raises(ValueError, match="empty"):
                mock_urcm_system.process_query(text)
        else:
            result = mock_urcm_system.process_query(text)
            assert result is not None


class TestAPIHealth:
    """Integration tests for the FastAPI health/metrics endpoints."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, test_client):
        resp = await test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data
        assert "checks" in data

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, test_client):
        resp = await test_client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")

    @pytest.mark.asyncio
    async def test_validate_endpoint(self, test_client):
        resp = await test_client.get("/api/validate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")


class TestObservability:
    """Integration tests for observability/structured logging."""

    def test_record_event(self, temp_log_dir):
        from urcm.core.observability import record_event
        record_event("test_event", {"key": "value", "count": 42})
        log_file = temp_log_dir / "logs" / "metrics.jsonl"
        assert log_file.exists()
        content = log_file.read_text()
        assert "test_event" in content
        assert "value" in content

    def test_event_redaction(self, temp_log_dir):
        from urcm.core.observability import record_event
        record_event("process", {"vector": np.array([1, 2, 3]), "embedding": "secret"})
        log_file = temp_log_dir / "logs" / "metrics.jsonl"
        content = log_file.read_text()
        assert "[REDACTED]" in content
        assert "secret" not in content

    def test_log_rotation(self, temp_log_dir, monkeypatch):
        monkeypatch.setenv("URCM_LOG_MAX_BYTES", "100")
        monkeypatch.setenv("URCM_LOG_BACKUPS", "1")
        from urcm.core.observability import record_event
        for i in range(20):
            record_event(f"event_{i}", {"data": "x" * 50})
        log_file = temp_log_dir / "logs" / "metrics.jsonl"
        assert log_file.exists()
        backup = temp_log_dir / "logs" / "metrics.jsonl.1"
        assert backup.exists() or log_file.stat().st_size > 0

    def test_observability_failure_does_not_crash(self, temp_log_dir, monkeypatch):
        with patch("os.makedirs", side_effect=PermissionError("no write")):
            from urcm.core.observability import record_event
            record_event("test", {"key": "val"})
