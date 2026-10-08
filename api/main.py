"""V8 FastAPI service for a selected, locally available model artifact."""

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.inference.runtime import load_predictor
from src.llm.inference.review import DraftStore, ReviewConflict


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticket text cannot be blank")
        return value


class TicketResponse(BaseModel):
    version: str
    label: str
    scores: dict[str, float]
    score_type: str


class DraftRequest(TicketRequest):
    ticket_id: str = Field(min_length=1, max_length=128)


class ReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=100)
    edited_response: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=1000)


def create_app(model_path: Path | None = None, generator=None,
               store_path: Path | None = None) -> FastAPI:
    path = model_path or Path(os.environ.get("TICKET_MODEL_PATH", "models/v1.json"))
    workflow_enabled = generator is not None or bool(os.environ.get("TICKET_LLM_PATH"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mode = os.environ.get("TICKET_MODE", "demo")
        if mode not in ("demo", "production"):
            raise RuntimeError("TICKET_MODE must be demo or production")
        api_key = os.environ.get("TICKET_API_KEY")
        if not api_key and os.environ.get("TICKET_ALLOW_UNAUTHENTICATED") != "1":
            raise RuntimeError("Set TICKET_API_KEY or TICKET_ALLOW_UNAUTHENTICATED=1 for local use")
        app.state.metadata, app.state.predictor = load_predictor(path)
        if workflow_enabled:
            if mode == "production":
                raise RuntimeError("LLM workflow is a local demo; production promotion is not implemented")
            if not api_key or not os.environ.get("TICKET_REVIEWER_API_KEY"):
                raise RuntimeError("Draft workflow requires TICKET_API_KEY and TICKET_REVIEWER_API_KEY")
            from src.llm.inference.generate import DemoLLMGenerator
            app.state.generator = generator or DemoLLMGenerator(Path(os.environ["TICKET_LLM_PATH"]))
            app.state.store = DraftStore(store_path or Path(os.environ.get(
                "TICKET_REVIEW_DB", "data/runtime/tickets.sqlite3")))
        if mode == "production":
            if not api_key:
                raise RuntimeError("Production requires TICKET_API_KEY")
            manifest = os.environ.get("TICKET_PROMOTION_MANIFEST")
            if not manifest:
                raise RuntimeError("Production requires TICKET_PROMOTION_MANIFEST")
            from src.inference.promotion import validate_promotion
            validate_promotion(Path(manifest), path, app.state.metadata)
        yield

    app = FastAPI(title="Enterprise Ticket Classifier", version="0.1.0", lifespan=lifespan)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("TICKET_API_KEY")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def require_reviewer_key(x_reviewer_key: str | None = Header(default=None)) -> None:
        expected = os.environ.get("TICKET_REVIEWER_API_KEY")
        if not expected or not hmac.compare_digest(x_reviewer_key or "", expected):
            raise HTTPException(status_code=401, detail="Invalid reviewer key")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": app.state.metadata["version"]}

    @app.post("/predict", response_model=TicketResponse, dependencies=[Depends(require_key)])
    def predict_ticket(request: TicketRequest) -> TicketResponse:
        result = app.state.predictor(request.text)
        version = app.state.metadata["version"]
        score_type = "cosine_similarity" if version in ("v2", "v6") else "model_probability"
        return TicketResponse(version=version, label=result["label"], scores=result["scores"],
                              score_type=score_type)

    if workflow_enabled:
        @app.post("/drafts", dependencies=[Depends(require_key)])
        def create_draft(request: DraftRequest) -> dict:
            classification = app.state.predictor(request.text)
            suggestion = app.state.generator.draft(request.text, classification["label"])
            return app.state.store.create(request.ticket_id, request.text,
                                          classification["label"], app.state.metadata["version"],
                                          app.state.generator.version, suggestion)

        @app.get("/drafts/{draft_id}", dependencies=[Depends(require_reviewer_key)])
        def get_draft(draft_id: str) -> dict:
            result = app.state.store.get(draft_id)
            if result is None:
                raise HTTPException(status_code=404, detail="Draft not found")
            return result

        @app.post("/drafts/{draft_id}/approve", dependencies=[Depends(require_reviewer_key)])
        def approve_draft(draft_id: str, request: ReviewRequest) -> dict:
            try:
                return app.state.store.decide(draft_id, request.reviewer, True,
                                              request.edited_response)
            except KeyError:
                raise HTTPException(status_code=404, detail="Draft not found") from None
            except ReviewConflict as error:
                raise HTTPException(status_code=409, detail=str(error)) from None
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from None

        @app.post("/drafts/{draft_id}/reject", dependencies=[Depends(require_reviewer_key)])
        def reject_draft(draft_id: str, request: ReviewRequest) -> dict:
            try:
                return app.state.store.decide(draft_id, request.reviewer, False,
                                              reason=request.reason)
            except KeyError:
                raise HTTPException(status_code=404, detail="Draft not found") from None
            except ReviewConflict as error:
                raise HTTPException(status_code=409, detail=str(error)) from None
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from None

        @app.get("/tickets/{ticket_id}/responses", dependencies=[Depends(require_reviewer_key)])
        def ticket_responses(ticket_id: str) -> list[dict]:
            return app.state.store.ticket_responses(ticket_id)

    return app


app = create_app()
