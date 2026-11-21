"use strict";

const form = document.querySelector("#composer");
const input = document.querySelector("#message");
const send = document.querySelector("#send");
const thread = document.querySelector("#thread");
const reset = document.querySelector("#new-chat");
const icon = document.querySelector("#assistant-icon");
const samples = [...document.querySelectorAll(".sample")];
const datasets = new Map([...document.querySelectorAll("input[name=dataset]")].map(radio => [radio.value, radio.dataset.label]));
const decisions = new Map([
  ["review", ["Review", "The stored score meets the review threshold."]],
  ["no_alert", ["No alert", "The stored score is below the review threshold."]],
  ["quarantined", ["Quarantined", null]],
  ["not_scored", ["Not scored", "No score has been stored for the current model."]],
]);
const labels = new Map([["fraud", "Fraud"], ["not_fraud", "Not fraud"]]);
let active = null;

function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function updateControls() {
  send.disabled = Boolean(active) || !input.value.trim();
  for (const sample of samples) sample.disabled = Boolean(active);
  reset.disabled = thread.children.length === 1 && !input.value;
}

function appendMessage(who, content, state = "") {
  const message = node("div", `message ${who} ${state}`.trim());
  if (who === "assistant") message.append(icon.content.cloneNode(true));
  const wrapper = node("div", "message-content");
  wrapper.append(node("p", "speaker", who === "user" ? "You" : "Assistant"));
  const body = node("div", "message-body");
  body.append(content);
  wrapper.append(body);
  message.append(wrapper);
  thread.append(message);
  thread.scrollTop = thread.scrollHeight;
  return message;
}

function fact(list, name, value, wide = false) {
  const item = node("div", wide ? "wide" : "");
  item.append(node("dt", "", name), node("dd", "", value));
  list.append(item);
}

function score(value) {
  return Number.isFinite(value) ? value.toFixed(6) : "Not available";
}

function transactionCard(transaction, fallback) {
  const status = transaction.scoring_status;
  const decision = decisions.get(status);
  const card = node("section", "result");
  const head = node("div", "result-head");
  const identity = node("div");
  identity.append(node("h3", "", transaction.transaction_id),
    node("p", "result-dataset", datasets.get(transaction.dataset) ?? transaction.dataset));
  head.append(identity, node("span", decision ? `decision ${status}` : "decision not_scored", decision?.[0] ?? "Unknown"));
  const facts = node("dl", "result-facts");
  fact(facts, "Stored score", score(transaction.risk_score));
  fact(facts, "Review threshold", score(transaction.threshold));
  if (transaction.amount != null) fact(facts, "Amount", [transaction.amount, transaction.currency].filter(Boolean).join(" "));
  fact(facts, "Demo label", labels.get(transaction.fraud_label) ?? "Unlabeled");
  if (transaction.transaction_type) {
    fact(facts, transaction.dataset === "ieee_cis" ? "Product code" : "Transaction type", transaction.transaction_type);
  }
  if (transaction.model_version) fact(facts, "Model version", transaction.model_version, true);
  card.append(head, node("p", "result-description", decision?.[1] ?? fallback), facts);
  return card;
}

async function lookUp(message) {
  if (!message || active) return;
  const dataset = document.querySelector("input[name=dataset]:checked").value;
  appendMessage("user", node("p", "plain", message));
  const waiting = appendMessage("assistant", node("p", "plain", "Looking up the transaction..."), "pending");
  const controller = new AbortController();
  active = controller;
  updateControls();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(form.action, {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "application/json"},
      body: JSON.stringify({message, default_dataset: dataset}),
      signal: controller.signal,
    });
    const data = await response.json();
    if (active !== controller) return;
    waiting.remove();
    if (response.ok && data.transaction) {
      appendMessage("assistant", transactionCard(data.transaction, data.reply));
    } else {
      const reply = typeof data.reply === "string" ? data.reply : "The lookup could not be completed. Try again.";
      appendMessage("assistant", node("p", "plain", reply), "error");
    }
  } catch {
    if (active !== controller) return;
    waiting.remove();
    const reply = controller.signal.aborted ? "The lookup took too long. Try again." : "The chatbot is unavailable. Try again.";
    appendMessage("assistant", node("p", "plain", reply), "error");
  } finally {
    window.clearTimeout(timeout);
    if (active === controller) {
      active = null;
      updateControls();
    }
  }
}

input.addEventListener("input", updateControls);
form.addEventListener("submit", event => {
  event.preventDefault();
  if (active) return;
  const message = input.value.trim();
  input.value = "";
  lookUp(message);
  updateControls();
  input.focus();
});
for (const sample of samples) {
  sample.addEventListener("click", () => lookUp(`Check ${sample.dataset.transaction}`));
}
reset.addEventListener("click", () => {
  active?.abort();
  active = null;
  while (thread.lastElementChild !== thread.firstElementChild) thread.lastElementChild.remove();
  input.value = "";
  updateControls();
  input.focus();
});
updateControls();
