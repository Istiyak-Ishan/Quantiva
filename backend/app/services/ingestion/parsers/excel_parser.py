import pandas as pd
import io
from typing import List, Dict, Any

def parse_excel(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Parses an Excel file using pandas/openpyxl and returns each sheet
    as a markdown table to preserve the grid structure for LLMs.
    """
    chunks = []
    
    # Read all sheets into a dictionary of DataFrames
    sheets = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
            
        # Convert the dataframe to markdown
        md_table = df.to_markdown(index=False)
        row_count = len(df)
        
        chunks.append({
            "text": f"Sheet: {sheet_name}\n\n{md_table}",
            "sheet_name": sheet_name,
            "row_range": f"1-{row_count}",
            "block_type": "table"
        })
        
    return chunks
