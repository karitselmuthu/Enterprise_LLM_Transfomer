"""DynamoDB draft store for the hosted fictional-ticket demo."""

from datetime import datetime, timezone
from uuid import uuid4

from src.llm.inference.review import ReviewConflict


def _attributes(item: dict) -> dict:
    return {key: ({"NULL": True} if value is None else {"S": str(value)})
            for key, value in item.items()}


def _values(item: dict) -> dict:
    return {key: None if "NULL" in value else value["S"] for key, value in item.items()}


class DynamoDraftStore:
    def __init__(self, drafts_table: str, responses_table: str, client=None):
        if not drafts_table or not responses_table:
            raise ValueError("Both DynamoDB table names are required")
        if client is None:
            import boto3
            client = boto3.client("dynamodb")
        self.client = client
        self.drafts_table = drafts_table
        self.responses_table = responses_table

    def create(self, ticket_id: str, ticket_text: str, label: str,
               classifier_version: str, generator_version: str,
               suggested_response: str) -> dict:
        item = {
            "id": str(uuid4()), "ticket_id": ticket_id, "ticket_text": ticket_text,
            "label": label, "classifier_version": classifier_version,
            "generator_version": generator_version, "suggested_response": suggested_response,
            "status": "pending", "reviewer": None, "final_response": None,
            "review_reason": None, "created_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_at": None,
        }
        self.client.put_item(TableName=self.drafts_table, Item=_attributes(item),
                             ConditionExpression="attribute_not_exists(id)")
        return item

    def get(self, draft_id: str) -> dict | None:
        result = self.client.get_item(TableName=self.drafts_table,
                                      Key={"id": {"S": draft_id}}, ConsistentRead=True)
        return _values(result["Item"]) if "Item" in result else None

    def decide(self, draft_id: str, reviewer: str, approve: bool,
               edited_response: str | None = None, reason: str | None = None) -> dict:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("Reviewer is required")
        if approve:
            if edited_response is not None and not edited_response.strip():
                raise ValueError("Approved response cannot be blank")
        elif not reason or not reason.strip():
            raise ValueError("Rejection reason is required")
        draft = self.get(draft_id)
        if draft is None:
            raise KeyError(draft_id)
        if draft["status"] != "pending":
            raise ReviewConflict("Draft has already been reviewed")

        now = datetime.now(timezone.utc).isoformat()
        final = (edited_response.strip() if edited_response is not None else
                 draft["suggested_response"]) if approve else None
        changes = {"status": "approved" if approve else "rejected", "reviewer": reviewer,
                   "final_response": final, "review_reason": reason.strip() if reason else None,
                   "reviewed_at": now}
        names = {f"#n{index}": key for index, key in enumerate(changes)}
        values = {f":v{index}": _attributes({"value": value})["value"]
                  for index, value in enumerate(changes.values())}
        update = {
            "TableName": self.drafts_table, "Key": {"id": {"S": draft_id}},
            "ConditionExpression": "#status = :pending",
            "UpdateExpression": "SET " + ", ".join(
                f"#n{index} = :v{index}" for index in range(len(changes))),
            "ExpressionAttributeNames": {"#status": "status", **names},
            "ExpressionAttributeValues": {":pending": {"S": "pending"}, **values},
        }
        actions = [{"Update": update}]
        if approve:
            response = {"id": str(uuid4()), "ticket_id": draft["ticket_id"],
                        "draft_id": draft_id, "response": final, "reviewer": reviewer,
                        "created_at": now, "response_key": f"{now}#{draft_id}"}
            actions.append({"Put": {"TableName": self.responses_table,
                                    "Item": _attributes(response),
                                    "ConditionExpression": "attribute_not_exists(ticket_id)"}})
        try:
            self.client.transact_write_items(TransactItems=actions)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code == "TransactionCanceledException":
                current = self.get(draft_id)
                if current and current["status"] != "pending":
                    raise ReviewConflict("Draft has already been reviewed") from None
            raise
        return self.get(draft_id)

    def ticket_responses(self, ticket_id: str) -> list[dict]:
        request = {"TableName": self.responses_table,
                   "KeyConditionExpression": "ticket_id = :ticket_id",
                   "ExpressionAttributeValues": {":ticket_id": {"S": ticket_id}},
                   "ConsistentRead": True}
        responses = []
        while True:
            page = self.client.query(**request)
            for raw in page.get("Items", []):
                response = _values(raw)
                response.pop("response_key", None)
                responses.append(response)
            if "LastEvaluatedKey" not in page:
                return responses
            request["ExclusiveStartKey"] = page["LastEvaluatedKey"]
