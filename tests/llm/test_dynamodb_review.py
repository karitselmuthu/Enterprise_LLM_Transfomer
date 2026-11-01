import unittest

import boto3
from botocore.validate import validate_parameters

from src.llm.inference.dynamodb_review import DynamoDraftStore
from src.llm.inference.review import ReviewConflict


class FakeDynamoClient:
    def __init__(self):
        self.drafts = {}
        self.responses = []
        self.calls = []

    def put_item(self, **request):
        self.calls.append(("PutItem", request))
        item = request["Item"]
        self.drafts[item["id"]["S"]] = item

    def get_item(self, **request):
        self.calls.append(("GetItem", request))
        item = self.drafts.get(request["Key"]["id"]["S"])
        return {"Item": item} if item else {}

    def transact_write_items(self, **request):
        self.calls.append(("TransactWriteItems", request))
        update = request["TransactItems"][0]["Update"]
        item = self.drafts[update["Key"]["id"]["S"]]
        if item["status"]["S"] != "pending":
            raise AssertionError("Transaction should have been rejected before this call")
        for name, value in zip(update["ExpressionAttributeNames"].values(),
                               update["ExpressionAttributeValues"].values()):
            if name != "status" or value != {"S": "pending"}:
                item[name] = value
        if len(request["TransactItems"]) == 2:
            self.responses.append(request["TransactItems"][1]["Put"]["Item"])

    def query(self, **request):
        self.calls.append(("Query", request))
        ticket_id = request["ExpressionAttributeValues"][":ticket_id"]["S"]
        return {"Items": [item for item in self.responses
                          if item["ticket_id"]["S"] == ticket_id]}


class DynamoDraftStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = DynamoDraftStore("drafts", "responses", FakeDynamoClient())

    def test_approval_persists_one_response_and_rejects_second_decision(self):
        draft = self.store.create("INC/2026", "VPN down", "network", "v5", "demo", "Reply")
        self.assertEqual(draft["status"], "pending")
        self.assertEqual(self.store.ticket_responses("INC/2026"), [])
        approved = self.store.decide(draft["id"], "alice", True, "Edited reply")
        self.assertEqual(approved["final_response"], "Edited reply")
        responses = self.store.ticket_responses("INC/2026")
        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0]["response"], "Edited reply")
        with self.assertRaises(ReviewConflict):
            self.store.decide(draft["id"], "alice", True)

    def test_rejection_does_not_create_response(self):
        draft = self.store.create("INC-2", "Issue", "access", "v5", "demo", "Reply")
        rejected = self.store.decide(draft["id"], "bob", False, reason="Needs investigation")
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(self.store.ticket_responses("INC-2"), [])

    def test_requests_match_boto3_dynamodb_shapes(self):
        draft = self.store.create("INC-3", "Issue", "access", "v5", "demo", "Reply")
        self.store.decide(draft["id"], "alice", True)
        self.store.ticket_responses("INC-3")
        service = boto3.client("dynamodb", region_name="us-east-1",
                               aws_access_key_id="test", aws_secret_access_key="test")
        for operation, request in self.store.client.calls:
            shape = service.meta.service_model.operation_model(operation).input_shape
            validate_parameters(request, shape)


if __name__ == "__main__":
    unittest.main()
