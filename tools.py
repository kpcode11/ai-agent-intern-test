import json
import numpy as np
import re

_INDEX_CACHE = None

def get_order_status(order_id: str) -> dict:
    """
    Looks up an order by ID and returns its status and details.
    Removes internal fields and PII before returning.
    """
    if not order_id:
        return {"error": "order_id is required."}
        
    order_id = order_id.strip().upper()
    try:
        with open("data/orders.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "Order database not found. Cannot look up orders."}
        
    for order in data.get("orders", []):
        if order.get("order_id", "").upper() == order_id:
            safe_order = json.loads(json.dumps(order))
            
            if "customer" in safe_order:
                safe_order["customer"].pop("email", None)
                safe_order["customer"].pop("shipping_address", None)
                
            safe_order.pop("internal", None)
            
            # Avoid reporting stale delivery fields for cancelled or returned orders
            if safe_order.get("status") in ["cancelled", "returned"]:
                safe_order["carrier"] = None
                safe_order["tracking_number"] = None
                safe_order["estimated_delivery"] = None
                
            return safe_order
            
    return {"error": f"Order {order_id} not found."}


def search_knowledge_base(query: str, client, top_k: int = 5) -> list:
    """
    Embeds the query and searches the in-memory index for relevant policy chunks.
    Boosts active documents over legacy ones.
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        try:
            with open("index.json", "r", encoding="utf-8") as f:
                _INDEX_CACHE = json.load(f)
        except FileNotFoundError:
            return [{"error": "Knowledge base index not found. Please notify the system administrator."}]
            
    documents = _INDEX_CACHE
        
    try:
        response = client.models.embed_content(
            model='gemini-embedding-2',
            contents=query
        )
        query_embedding = np.array(response.embeddings[0].values)
    except Exception as e:
        return [{"error": f"Failed to embed query: {e}"}]
        
    results = []
    for doc in documents:
        doc_emb = np.array(doc["embedding"])
        # Cosine similarity
        norm = np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        if norm == 0:
            continue
            
        similarity = np.dot(query_embedding, doc_emb) / norm
        
        status = doc["metadata"].get("status", "")
        if status == "active":
            similarity += 0.05
        elif status == "legacy" or status == "superseded":
            similarity -= 0.05
            
        # Strongly downrank internal/scratchpad documents
        if doc["metadata"].get("customer_answering") == False:
            similarity -= 1.0
            
        results.append({
            "filename": doc["filename"],
            "heading": doc["heading"],
            "content": doc["content"],
            "metadata": doc["metadata"],
            "similarity": float(similarity)
        })
        
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Only return necessary fields to the LLM
    final_results = []
    for r in results[:top_k]:
        final_results.append({
            "source": f"{r['filename']} > {r['heading']}",
            "metadata": r["metadata"],
            "content": r["content"]
        })
        
    return final_results
