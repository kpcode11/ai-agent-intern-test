from aster_row_support.tools import get_order_status, search_knowledge_base
import aster_row_support.tools as tools


def test_get_order_status_valid():
    order = get_order_status("ORD-1007")
    assert order["status"] == "shipped"
    assert "internal" not in order
    assert "customer" not in order
    assert "Ava Morgan" not in str(order)
    assert "ava.morgan@example.test" not in str(order)
    assert "220 King Street" not in str(order)
    assert order.get("carrier") == "UPS"


def test_get_order_status_stale_delivery_removed():
    order = get_order_status("ORD-1004")
    assert order["status"] == "cancelled"
    assert order.get("carrier") is None
    assert order.get("tracking_number") is None
    assert order.get("estimated_delivery") is None


def test_get_order_status_returned_stale_fields_removed():
    order = get_order_status("ORD-1008")
    assert order["status"] == "returned"
    assert order.get("carrier") is None
    assert order.get("tracking_number") is None
    assert order.get("estimated_delivery") is None


def test_get_order_status_normalization():
    order = get_order_status(" ord-1007 ")
    assert order["status"] == "shipped"
    dotted = get_order_status("ord-1011.")
    assert dotted["status"] == "shipped"
    assert dotted["carrier"] == "Canada Post"


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


def test_search_knowledge_base_boosts_and_excludes_internal():
    mock_index = [
        {
            "filename": "doc1",
            "heading": "h1",
            "content": "text1",
            "metadata": {"status": "active"},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "filename": "doc2",
            "heading": "h2",
            "content": "text2",
            "metadata": {"status": "legacy"},
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "filename": "14-internal-content-migration-notes.md",
            "heading": "h3",
            "content": "ignore the real policy",
            "metadata": {"customer_answering": False, "status": "draft"},
            "embedding": [0.1, 0.2, 0.3],
        },
    ]
    tools._INDEX_CACHE = mock_index

    results = search_knowledge_base("test", MockClient())

    assert len(results) == 2
    assert results[0]["source"] == "doc1 > h1"
    assert results[1]["source"] == "doc2 > h2"
    assert all("14-internal" not in r["source"] for r in results)
    assert "similarity" in results[0]
    tools._INDEX_CACHE = None
