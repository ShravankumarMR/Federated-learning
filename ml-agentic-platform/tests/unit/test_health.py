from app.services.health import health_payload


def test_health_payload() -> None:
    payload = health_payload()
    assert payload["status"] == "ok"
    assert "environment" in payload
