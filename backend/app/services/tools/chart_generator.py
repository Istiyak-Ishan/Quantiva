import io
import uuid
from typing import Dict, Any, List
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
from app.core.storage import s3_client, BUCKET_NAME, ensure_bucket_exists

class ChartGeneratorTool:
    def __init__(self):
        pass

    async def generate(self, data: Dict[str, List[Any]], chart_type: str, title: str, x_label: str = "", y_label: str = "") -> Dict[str, Any]:
        """
        Generates a chart from provided data, saves it to MinIO, and returns the public URL.
        """
        if not plt:
            return {"success": False, "chart_url": None, "error": "Matplotlib not installed."}
            
        # Ensure 'x' and 'y' keys exist in data
        x_data = data.get('x', [])
        y_data = data.get('y', [])
        
        if not x_data or not y_data or len(x_data) != len(y_data):
            return {"success": False, "chart_url": None, "error": "Invalid data format. Provide 'x' and 'y' arrays of equal length."}
            
        plt.figure(figsize=(8, 5))
        
        try:
            if chart_type == "bar":
                plt.bar(x_data, y_data, color='#3b82f6')
            elif chart_type == "line":
                plt.plot(x_data, y_data, marker='o', color='#10b981', linewidth=2)
            elif chart_type == "scatter":
                plt.scatter(x_data, y_data, color='#f59e0b')
            elif chart_type == "pie":
                plt.pie(y_data, labels=x_data, autopct='%1.1f%%', startangle=90)
            else:
                return {"success": False, "chart_url": None, "error": f"Unsupported chart_type: {chart_type}"}
                
            plt.title(title)
            if chart_type != "pie":
                plt.xlabel(x_label)
                plt.ylabel(y_label)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save chart to a bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            file_bytes = buf.getvalue()
            
            # Upload to MinIO
            ensure_bucket_exists()
            object_name = f"charts/{uuid.uuid4().hex}.png"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=object_name,
                Body=file_bytes,
                ContentType="image/png"
            )
            
            # Generate a presigned URL valid for 24 hours
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': BUCKET_NAME, 'Key': object_name},
                ExpiresIn=86400
            )
            
            return {"success": True, "chart_url": url, "error": None}
            
        except Exception as e:
            return {"success": False, "chart_url": None, "error": str(e)}
        finally:
            plt.close() # Free memory

chart_generator = ChartGeneratorTool()
