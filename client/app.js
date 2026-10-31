"use strict";

const field = (id) => document.getElementById(id);
let loadedDraft = null;

function status(id, message, error = false) {
  const element = field(id);
  element.textContent = message;
  element.classList.toggle("error", error);
}

async function apiRequest(path, keyHeader, key, method = "GET", body = null) {
  const base = field("api-url").value.trim().replace(/\/+$/, "");
  const headers = { [keyHeader]: key };
  if (body !== null) headers["Content-Type"] = "application/json";
  const response = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === null ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const result = await response.json();
  if (!response.ok) {
    const detail = typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail);
    throw new Error(`HTTP ${response.status}: ${detail}`);
  }
  return result;
}

function showDraft(draft) {
  loadedDraft = draft;
  field("draft-panel").hidden = false;
  field("draft-id").value = draft.id;
  field("draft-ticket").textContent = draft.ticket_id;
  field("draft-label").textContent = draft.label;
  field("draft-status").textContent = draft.status;
  field("draft-text").textContent = draft.ticket_text;
  field("reply").value = draft.final_response ?? draft.suggested_response;
  const pending = draft.status === "pending";
  field("approve").disabled = !pending;
  field("reject").disabled = !pending;
}

field("submit-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  status("submit-status", "Creating draft…");
  try {
    const draft = await apiRequest("/drafts", "X-API-Key", field("submit-key").value, "POST", {
      ticket_id: field("ticket-id").value.trim(),
      text: field("ticket-text").value,
    });
    loadedDraft = null;
    field("draft-panel").hidden = true;
    field("draft-id").value = draft.id;
    status("submit-status", `Draft ${draft.id} created. Enter the reviewer key and load it.`);
  } catch (error) {
    status("submit-status", error.message, true);
  }
});

field("load-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  status("review-status", "Loading draft…");
  try {
    const id = encodeURIComponent(field("draft-id").value.trim());
    const draft = await apiRequest(`/drafts/${id}`, "X-Reviewer-Key", field("review-key").value);
    showDraft(draft);
    status("review-status", `Draft is ${draft.status}.`);
  } catch (error) {
    loadedDraft = null;
    field("draft-panel").hidden = true;
    status("review-status", error.message, true);
  }
});

field("decision-form").addEventListener("submit", (event) => event.preventDefault());

async function decide(action) {
  if (!loadedDraft || loadedDraft.status !== "pending") return;
  const reviewer = field("reviewer").value.trim();
  const reply = field("reply").value.trim();
  const reason = field("reason").value.trim();
  if (!reviewer || (action === "approve" && !reply) || (action === "reject" && !reason)) {
    status("review-status", "Enter a reviewer name and a reply or rejection reason.", true);
    return;
  }
  status("review-status", `${action === "approve" ? "Approving" : "Rejecting"} draft…`);
  field("approve").disabled = true;
  field("reject").disabled = true;
  try {
    const body = action === "approve" ? { reviewer, edited_response: reply } : { reviewer, reason };
    const draft = await apiRequest(`/drafts/${encodeURIComponent(loadedDraft.id)}/${action}`,
      "X-Reviewer-Key", field("review-key").value, "POST", body);
    showDraft(draft);
    status("review-status", `Draft ${draft.status}.`);
    if (action === "approve") await refreshResponses();
  } catch (error) {
    field("approve").disabled = false;
    field("reject").disabled = false;
    status("review-status", error.message, true);
  }
}

field("approve").addEventListener("click", () => decide("approve"));
field("reject").addEventListener("click", () => decide("reject"));

async function refreshResponses() {
  const ticketId = loadedDraft?.ticket_id ?? field("ticket-id").value.trim();
  if (!ticketId || !field("review-key").value) {
    status("responses-status", "Enter a ticket ID and reviewer key first.", true);
    return;
  }
  try {
    const responses = await apiRequest(`/tickets/${encodeURIComponent(ticketId)}/responses`,
      "X-Reviewer-Key", field("review-key").value);
    field("responses").replaceChildren(...responses.map((item) => {
      const entry = document.createElement("li");
      entry.textContent = `${item.response} — ${item.reviewer}`;
      return entry;
    }));
    status("responses-status", `${responses.length} saved response(s) for ${ticketId}.`);
  } catch (error) {
    status("responses-status", error.message, true);
  }
}

field("refresh-responses").addEventListener("click", refreshResponses);
