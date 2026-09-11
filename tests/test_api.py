"""API integration tests."""

import io


def test_health_empty(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_search_ask_anomalies(client):
    content = (
        b"Service Agreement. The provider guarantees 99.9 percent uptime. "
        b"Termination requires 30 days written notice. Penalties apply for breach."
    )
    resp = client.post("/documents", files={"file": ("agreement.txt", io.BytesIO(content), "text/plain")})
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    resp = client.get("/documents")
    assert any(d["id"] == doc_id for d in resp.json())

    resp = client.get("/search", params={"q": "termination notice period", "top_k": 3})
    assert resp.status_code == 200
    assert resp.json()["hits"]

    resp = client.post("/ask", json={"question": "What is the termination notice period?"})
    assert resp.status_code == 200
    assert "Sources" in resp.json()["answer"]

    resp = client.get("/anomalies")
    assert resp.status_code == 200

    resp = client.delete(f"/documents/{doc_id}")
    assert resp.status_code == 204


def test_upload_rejects_bad_type(client):
    resp = client.post("/documents", files={"file": ("data.csv", io.BytesIO(b"a,b"), "text/csv")})
    assert resp.status_code == 415
