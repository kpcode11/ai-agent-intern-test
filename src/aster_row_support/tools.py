import json
import re
from pathlib import Path

import numpy as np

_INDEX_CACHE = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CUSTOMER_SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "items",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

ITEM_SAFE_FIELDS = {"name", "quantity", "final_sale", "sku"}


def get_order_status(order_id: str) -> dict:
    """
    Looks up an order by ID and returns its status and details.
    Never returns customer PII or internal fields.
    """
    if not order_id:
        return {"error": "order_id is required."}

    order_id = re.sub(r"[^A-Za-z0-9-]", "", order_id.strip()).upper()
    try:
        with (PROJECT_ROOT / "data" / "orders.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "Order database not found. Cannot look up orders."}

    for order in data.get("orders", []):
        if order.get("order_id", "").upper() == order_id:
            safe_order = {key: order.get(key) for key in CUSTOMER_SAFE_FIELDS if key in order}
            items = []
            for item in order.get("items") or []:
                items.append({key: item.get(key) for key in ITEM_SAFE_FIELDS if key in item})
            safe_order["items"] = items

            # Avoid reporting stale delivery fields for cancelled or returned orders
            if safe_order.get("status") in ["cancelled", "returned"]:
                safe_order["carrier"] = None
                safe_order["tracking_number"] = None
                safe_order["estimated_delivery"] = None
                safe_order["delivered_at"] = None

            return safe_order

    return {"error": f"Order {order_id} not found."}


def search_knowledge_base(query: str, client, top_k: int = 8) -> list:
    """
    Embeds the query and searches the in-memory index for relevant policy chunks.
    Boosts active documents over legacy ones and excludes internal-only docs.
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        try:
            with (PROJECT_ROOT / "data" / "index.json").open("r", encoding="utf-8") as f:
                _INDEX_CACHE = json.load(f)
        except FileNotFoundError:
            return [{"error": "Knowledge base index not found. Please notify the system administrator."}]

    documents = _INDEX_CACHE

    try:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=query,
        )
        query_embedding = np.array(response.embeddings[0].values)
    except Exception as e:
        return [{"error": f"Failed to embed query: {e}"}]

    results = []
    for doc in documents:
        metadata = doc.get("metadata") or {}
        if metadata.get("customer_answering") is False:
            continue

        doc_emb = np.array(doc["embedding"])
        norm = np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
        if norm == 0:
            continue

        similarity = np.dot(query_embedding, doc_emb) / norm

        status = metadata.get("status", "")
        if status == "active":
            similarity += 0.05
        elif status in ("legacy", "superseded", "draft"):
            similarity -= 0.05

        results.append(
            {
                "filename": doc["filename"],
                "heading": doc["heading"],
                "content": doc["content"],
                "metadata": metadata,
                "similarity": float(similarity),
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)

    final_results = []
    for r in results[:top_k]:
        final_results.append(
            {
                "source": f"{r['filename']} > {r['heading']}",
                "metadata": r["metadata"],
                "content": r["content"],
                "similarity": round(r["similarity"], 4),
            }
        )

    return final_results
