const storageKey = "pocket-ledger.entries.v1";
const form = document.querySelector("#entry-form");
const description = document.querySelector("#description");
const amount = document.querySelector("#amount");
const entriesElement = document.querySelector("#entries");
const balanceElement = document.querySelector("#balance");
const emptyState = document.querySelector("#empty-state");
const errorElement = document.querySelector("#form-error");

function loadEntries() { try { return JSON.parse(localStorage.getItem(storageKey)) || []; } catch { return []; } }
let entries = loadEntries();
function saveEntries() { localStorage.setItem(storageKey, JSON.stringify(entries)); }
function euros(value) { return new Intl.NumberFormat("en-IE", {style:"currency",currency:"EUR"}).format(value); }
function render() {
  entriesElement.replaceChildren();
  entries.forEach((entry) => { const item = document.createElement("li"); const name = document.createElement("span"); const value = document.createElement("strong"); name.textContent = entry.description; value.textContent = euros(entry.amount); value.className = entry.amount >= 0 ? "income" : "expense"; item.append(name, value); entriesElement.append(item); });
  const total = entries.reduce((sum, entry) => sum + entry.amount, 0); balanceElement.textContent = euros(total); emptyState.hidden = entries.length > 0;
}
form.addEventListener("submit", (event) => { event.preventDefault(); const parsed = Number(amount.value.replace(",", ".")); if (!description.value.trim() || !Number.isFinite(parsed) || parsed === 0) { errorElement.textContent = "Enter a description and a non-zero amount."; return; } entries.unshift({description:description.value.trim(), amount:parsed}); saveEntries(); form.reset(); errorElement.textContent=""; render(); description.focus(); });
document.querySelector("#clear-entries").addEventListener("click", () => { entries=[]; saveEntries(); render(); });
render();
