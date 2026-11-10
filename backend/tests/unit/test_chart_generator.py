import uuid
from unittest.mock import patch

from app.services.tools.chart_generator import ChartGeneratorTool

OWNER = uuid.uuid4()


async def test_chart_is_real_png_and_persists_owner_scoped_path():
    with patch("app.services.tools.chart_generator.ensure_bucket_exists"), patch("app.services.tools.chart_generator.s3_client") as storage:
        result = await ChartGeneratorTool().generate(
            {"x": ["Q1", "Q2"], "y": [12, 14]}, "bar", "Revenue", owner_id=str(OWNER)
        )
    assert result["success"]
    assert result["chart_url"].startswith(f"/api/v1/charts/{OWNER}/")
    assert result["chart_url"].endswith(".png")
    assert storage.put_object.call_args.kwargs["Key"].startswith(f"charts/{OWNER}/")
    assert storage.put_object.call_args.kwargs["Body"].startswith(b"\x89PNG\r\n\x1a\n")


async def test_bad_chart_data_fails_without_storage():
    with patch("app.services.tools.chart_generator.s3_client") as storage:
        result = await ChartGeneratorTool().generate({"x": [1], "y": []}, "bar", "Bad", owner_id=str(OWNER))
    assert not result["success"]
    storage.put_object.assert_not_called()


async def test_chart_without_owner_is_rejected():
    with patch("app.services.tools.chart_generator.s3_client") as storage:
        result = await ChartGeneratorTool().generate({"x": [1], "y": [2]}, "bar", "No owner", owner_id=None)
    assert not result["success"]
    storage.put_object.assert_not_called()


async def test_chart_route_renews_url_and_enforces_ownership(client):
    registered = await client.post(
        "/api/v1/auth/register",
        json={"email": "charts@example.com", "password": "SecurePass123!"},
    )
    assert registered.status_code == 200
    user_id = (await client.get("/api/v1/auth/me")).json()["id"]
    chart_name = f"{uuid.uuid4().hex}.png"
    with patch("app.api.v1.charts.presigned_url", return_value="http://public.example/signed") as sign:
        response = await client.get(f"/api/v1/charts/{user_id}/{chart_name}")
        foreign = await client.get(f"/api/v1/charts/{uuid.uuid4()}/{chart_name}")
    assert response.status_code == 307
    assert response.headers["location"] == "http://public.example/signed"
    assert sign.call_args.args[0] == f"charts/{user_id}/{chart_name}"
    assert foreign.status_code == 404
