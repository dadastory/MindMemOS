import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos.api.app import register_exception_handlers
from mindmemos.api.schemas import AddRequest
from mindmemos.errors import BadRequestError
from pydantic import ValidationError


def test_request_validation_errors_return_one_message() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/add")
    async def add(payload: AddRequest):
        return {"code": "ok", "data": None}

    response = TestClient(app).post("/add", json={"user_id": "u1", "messages": []})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert body["data"] is None
    assert "body.messages" in body["message"]


def test_api_error_returns_one_message() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/bad")
    async def bad():
        raise BadRequestError("top_k must be <= 100; value=101", code="search.top_k_too_large")

    response = TestClient(app).get("/bad")

    assert response.status_code == 400
    assert response.json() == {
        "code": "search.top_k_too_large",
        "message": "top_k must be <= 100; value=101",
        "data": None,
    }


def test_document_blocks_are_mutually_exclusive_and_require_unique_ids() -> None:
    block = {"block_id": "block-1", "messages": [{"role": "user", "content": "fact"}]}

    request = AddRequest(user_id="u1", document_blocks=[block])
    assert request.messages == []
    assert request.document_blocks[0].block_id == "block-1"

    with pytest.raises(ValidationError, match="exactly one"):
        AddRequest(user_id="u1", messages=[{"role": "user", "content": "fact"}], document_blocks=[block])
    with pytest.raises(ValidationError, match="unique"):
        AddRequest(user_id="u1", document_blocks=[block, block])
