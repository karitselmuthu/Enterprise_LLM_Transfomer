import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import create_app
from src.baselines.traditional import fit


class FakeGenerator:
    version = "test-generator"

    def draft(self, ticket_text: str, label: str) -> str:
        return f"Please review this {label} issue: {ticket_text}"


class WorkflowApiTests(unittest.TestCase):
    def test_classify_draft_review_and_local_ticket_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = fit([{"text": "VPN connection down", "label": "network"},
                         {"text": "Reset login password", "label": "access"}])
            model_path = root / "classifier.json"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            settings = {"TICKET_API_KEY": "user-key", "TICKET_REVIEWER_API_KEY": "review-key",
                        "TICKET_MODE": "demo"}
            with patch.dict(os.environ, settings):
                app = create_app(model_path, FakeGenerator(), root / "review.sqlite3")
                with TestClient(app) as client:
                    draft_response = client.post("/drafts", headers={"X-API-Key": "user-key"},
                        json={"ticket_id": "INC-1", "text": "VPN connection down"})
                    self.assertEqual(draft_response.status_code, 200)
                    draft = draft_response.json()
                    self.assertEqual(draft["status"], "pending")
                    self.assertEqual(draft["label"], "network")
                    path = f"/tickets/INC-1/responses"
                    self.assertEqual(client.get(path, headers={"X-Reviewer-Key": "review-key"}).json(), [])
                    self.assertEqual(client.post(f"/drafts/{draft['id']}/approve",
                        headers={"X-Reviewer-Key": "wrong"},
                        json={"reviewer": "alice"}).status_code, 401)
                    approved = client.post(f"/drafts/{draft['id']}/approve",
                        headers={"X-Reviewer-Key": "review-key"},
                        json={"reviewer": "alice", "edited_response": "Approved reply"})
                    self.assertEqual(approved.status_code, 200)
                    self.assertEqual(approved.json()["final_response"], "Approved reply")
                    replies = client.get(path, headers={"X-Reviewer-Key": "review-key"}).json()
                    self.assertEqual(len(replies), 1)
                    self.assertEqual(replies[0]["response"], "Approved reply")
                    repeated = client.post(f"/drafts/{draft['id']}/approve",
                        headers={"X-Reviewer-Key": "review-key"}, json={"reviewer": "alice"})
                    self.assertEqual(repeated.status_code, 409)


if __name__ == "__main__":
    unittest.main()
