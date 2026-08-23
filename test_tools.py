import pytest
from tools import get_order_status, search_knowledge_base
import tools

def test_get_order_status_valid():
    order = get_order_status("ORD-1007")
    assert order["status"] == "shipped"
    assert "internal" not in order
    assert "email" not in order.get("customer", {})
    assert "shipping_address" not in order.get("customer", {})
    assert order["customer"]["name"] == "Ava Morgan"

def test_get_order_status_stale_delivery_removed():
    order = get_order_status("ORD-1004")
    assert order["status"] == "cancelled"
    assert order.get("carrier") is None
    assert order.get("tracking_number") is None
    assert order.get("estimated_delivery") is None

def test_get_order_status_normalization():
    order = get_order_status(" ord-1007 ")
    assert order["status"] == "shipped"

def test_get_order_status_missing():
    order = get_order_status("")
    assert "error" in order

def test_get_order_status_not_found():
    order = get_order_status("ORD-9999")
    assert "error" in order

class MockClient:
    class Models:
        def embed_content(self, model, contents):
            class Response:
                class Emb:
                    values = [0.1, 0.2, 0.3]
                embeddings = [Emb()]
            return Response()
    def __init__(self):
        self.models = self.Models()

def test_search_knowledge_base_boosts():
    mock_index = [
        {
            "filename": "doc1",
            "heading": "h1",
            "content": "text1",
            "metadata": {"status": "active"},
            "embedding": [0.1, 0.2, 0.3]
        },
        {
            "filename": "doc2",
            "heading": "h2",
            "content": "text2",
            "metadata": {"status": "legacy"},
            "embedding": [0.1, 0.2, 0.3]
        },
        {
            "filename": "doc3",
            "heading": "h3",
            "content": "text3",
            "metadata": {"customer_answering": False},
            "embedding": [0.1, 0.2, 0.3]
        }
    ]
    tools._INDEX_CACHE = mock_index
    
    client = MockClient()
    results = search_knowledge_base("test", client)
    
    assert len(results) == 3
    assert results[0]["source"] == "doc1 > h1"
    assert results[1]["source"] == "doc2 > h2"
    assert results[2]["source"] == "doc3 > h3"
