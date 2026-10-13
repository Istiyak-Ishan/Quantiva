from typing import List, Dict, Any

def build_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Builds the system prompt and the user message containing the query and retrieved context chunks.
    """
    
    # 1. The System Prompt (Rules of engagement)
    system_prompt = (
        "You are Quantiva, a professional AI document assistant. "
        "Your task is to answer the user's question accurately using ONLY the provided document context. "
        "Do not hallucinate or use outside knowledge. "
        "If the answer is not contained in the context, politely state that you cannot answer based on the documents provided. "
        "Always cite your sources by referencing the page number or chunk index provided in the context."
    )
    
    # 2. Format the retrieved chunks into a readable string
    context_text = ""
    for idx, chunk in enumerate(chunks):
        payload = chunk.get("payload", {})
        page = payload.get("page", "Unknown")
        text = payload.get("text", "")
        # score = chunk.get("score")
        
        context_text += f"\n--- [Source {idx+1} | Page {page}] ---\n"
        context_text += f"{text}\n"
        
    # 3. Create the final user message
    user_message = (
        f"Answer the following question based solely on the context provided below.\n\n"
        f"QUESTION: {query}\n\n"
        f"=== CONTEXT ==={context_text}\n============="
    )
    
    return {
        "system_prompt": system_prompt,
        "user_message": user_message
    }
