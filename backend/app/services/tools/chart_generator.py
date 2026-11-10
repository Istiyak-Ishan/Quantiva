"""Per-request Matplotlib figures, off-loop rendering, renewable owner URLs."""
import asyncio
import io
from threading import Lock
import uuid

from matplotlib.figure import Figure

from app.core.storage import s3_client, BUCKET_NAME, ensure_bucket_exists


class ChartGeneratorTool:
    def __init__(self):
        self._lock = Lock()

    def _generate(self, data, chart_type, title, x_label, y_label, owner_id):
        if not owner_id:
            return {"success": False, "chart_url": None, "error": "Chart owner is required"}
        x, y = data.get("x", []), data.get("y", [])
        if not x or len(x) != len(y) or len(x) > 1000 or chart_type not in {"bar", "line", "scatter", "pie"}:
            return {"success": False, "chart_url": None, "error": "Invalid or oversized chart data"}
        with self._lock:
            figure = Figure(figsize=(8, 5))
            axes = figure.subplots()
            if chart_type == "pie":
                axes.pie(y, labels=x, autopct="%1.1f%%")
            else:
                getattr(axes, "plot" if chart_type == "line" else chart_type)(x, y)
                axes.set_xlabel(x_label)
                axes.set_ylabel(y_label)
            axes.set_title(title)
            figure.tight_layout()
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=150)
            figure.clear()
        ensure_bucket_exists()
        chart_name = f"{uuid.uuid4().hex}.png"
        key = f"charts/{owner_id}/{chart_name}"
        s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=buffer.getvalue(), ContentType="image/png")
        # Persist a stable authenticated path; /api/v1/charts re-signs on read.
        return {"success": True, "chart_url": f"/api/v1/charts/{owner_id}/{chart_name}", "error": None}

    async def generate(self, data, chart_type, title, x_label="", y_label="", *, owner_id):
        return await asyncio.to_thread(self._generate, data, chart_type, title, x_label, y_label, owner_id)


chart_generator = ChartGeneratorTool()
