import pandas as pd
import io
from typing import List, Dict, Any

def parse_csv(file_bytes: bytes, batch_size: int = 100) -> List[Dict[str, Any]]:
    """
    Parses a CSV file and splits it into chunks of `batch_size` rows.
    Returns markdown tables for each chunk.
    """
    chunks = []
    df = pd.read_csv(io.BytesIO(file_bytes))
    
    total_rows = len(df)
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        md_table = batch_df.to_markdown(index=False)
        
        chunks.append({
            "text": md_table,
            "row_range": f"{start_idx + 1}-{end_idx}",
            "block_type": "table"
        })
        
    return chunks
