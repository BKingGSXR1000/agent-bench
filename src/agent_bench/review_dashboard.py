"""Local-only blinded browser workflow for M10 manual reviews."""

from __future__ import annotations

import json
import mimetypes
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent_bench.executor import ExperimentState
from agent_bench.manual_review import (
    CriterionResult,
    ManualReview,
    ManualReviewError,
    load_protocol,
    prepare_review_copy,
    review_queue,
    review_root,
    save_review,
    validate_review_against_protocol,
)


# Protocol-facing instructions, never agent prompts. They are invariant across
# harness, profile, prompt, repetition, and canonical run identity.
_STEPS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "entry-delete": (
        ("Delete Groceries", "Delete the seeded Groceries entry (€125.50 expense).", "Groceries disappears; Salary and Train pass remain."),
        ("Check balance", "Compare the balance before and after deletion.", "€1,825.00 becomes €1,950.50."),
        ("Check persistence", "Reload the application page normally.", "Groceries remains deleted."),
    ),
    "entry-filter": (
        ("Filter seeded data", "Enter gro in the filter/search control.", "Only Groceries is visible."),
        ("Check case handling", "Replace it with SAL.", "Salary is visible regardless of case."),
        ("Clear filter", "Clear the filter/search control.", "All three entries return and €1,825.00 is unchanged."),
    ),
    "entry-category": (
        ("Add categorized Coffee", "Add Coffee with amount −€4.50 and choose a category.", "The category is selectable and displayed."),
        ("Check persistence", "Reload the application page normally.", "Coffee and its category remain."),
        ("Check seeded entries", "Inspect Salary, Groceries, and Train pass.", "All still render and work."),
    ),
    "monthly-summary": (
        ("Check seeded totals", "Inspect the initial summary.", "Income €2,000.00; expenses €175.00; net €1,825.00."),
        ("Check update", "Add Coffee with amount −€4.50.", "Expenses become €179.50 and net becomes €1,820.50."),
        ("Check empty data", "Use a supported clear/delete flow if available.", "Valid zero values display without a crash; otherwise mark the relevant criterion UNREVIEWABLE and explain why."),
    ),
    "keyboard-entry": (
        ("Find shortcut", "Find the shortcut documentation or visible hint.", "Its purpose is discoverable."),
        ("Use shortcut", "Invoke it while focus is outside a text input.", "Focus moves to Description."),
        ("Check typing", "Type normally in Description and Amount.", "Typing is not hijacked and adding still works."),
    ),
}

_CRITERION_DETAILS: dict[str, tuple[str, str]] = {
    "delete_one_existing_entry": ("Delete Groceries using its normal control.", "The control completes the deletion."),
    "selected_entry_disappears": ("Inspect the list after deleting Groceries.", "Groceries is absent."),
    "unrelated_entries_remain": ("Inspect Salary and Train pass after deletion.", "Both remain."),
    "balance_updates": ("Compare the balance before and after deletion.", "€1,825.00 becomes €1,950.50."),
    "deletion_persists_after_reload": ("Reload the app after deleting Groceries.", "Groceries stays absent."),
    "labelled_filter_exists": ("Locate the filter/search input or control.", "It is visibly labeled or understandable."),
    "case_insensitive_description_filter": ("Search SAL, then gro.", "Salary and Groceries are found regardless of case."),
    "visible_set_changes_only": ("Apply a filter and inspect data/balance.", "Only visibility changes; entries and €1,825.00 balance do not."),
    "reset_restores_entries": ("Clear the filter.", "Salary, Groceries, and Train pass return."),
    "no_match_empty_state": ("Search for text absent from every entry.", "A safe empty state appears."),
    "entries_and_balance_unmutated": ("Filter, then clear the filter.", "Seeded entries and balance are unchanged."),
    "selectable_category_on_entry": ("Add Coffee −€4.50 and choose a category.", "A category can be selected."),
    "category_displayed_in_list": ("Inspect Coffee after it is added.", "Its chosen category is displayed."),
    "category_persists": ("Reload after adding categorized Coffee.", "Coffee and its category remain."),
    "legacy_entry_fallback_works": ("Inspect the seeded entries without categories.", "They still render and are usable."),
    "existing_entry_flow_works": ("Use an ordinary seeded-entry flow.", "Existing behavior is not broken."),
    "income_total_visible": ("Inspect the summary with seeded entries.", "Income €2,000.00 is visible."),
    "expense_total_visible": ("Inspect the summary with seeded entries.", "Expenses €175.00 are visible."),
    "net_visible": ("Inspect the summary with seeded entries.", "Net €1,825.00 is visible."),
    "totals_reflect_entries": ("Compare the summary with the three seeded amounts.", "2,000 − 125.50 − 49.50 equals €1,825.00."),
    "totals_update_after_change": ("Add Coffee −€4.50.", "Expenses become €179.50 and net becomes €1,820.50."),
    "empty_data_zero_values_valid": ("Use a supported clear/delete flow if available.", "Zero values are valid and no fatal error occurs."),
    "shortcut_is_documented": ("Find the shortcut hint or documentation.", "Its purpose is discoverable."),
    "shortcut_focuses_description": ("Invoke it outside text fields.", "Description receives focus."),
    "typing_in_input_not_hijacked": ("Type normally in Description and Amount.", "Typing is not intercepted."),
    "ordinary_add_entry_flow_works": ("Add an ordinary entry.", "The normal add flow still works."),
    "shortcut_has_no_unrelated_destructive_effect": ("Use the shortcut and inspect data.", "No unrelated data/action is changed."),
    "app_loads": ("Open the isolated application.", "It renders without an obvious fatal error."),
    "existing_entries_render": ("Inspect initial seeded data.", "Salary, Groceries, and Train pass are visible."),
    "adding_entry_works": ("Add Coffee −€4.50.", "The new entry appears."),
    "baseline_balance_updates": ("Add Coffee −€4.50.", "€1,825.00 becomes €1,820.50."),
    "no_obvious_fatal_runtime_error": ("Use the tested flow.", "No fatal screen or blocking crash occurs."),
    "unrelated_functionality_intact": ("Use a normal baseline flow unrelated to the task.", "It continues to work."),
}


def human_steps(task: str) -> list[dict[str, str]]:
    """Return the fixed, human-readable script for a semantic task."""
    try:
        return [{"label": label, "action": action, "expected": expected} for label, action, expected in _STEPS[task]]
    except KeyError as exc:
        raise ManualReviewError(f"no dashboard script for semantic task {task!r}") from exc


def human_criteria(criteria: object) -> list[dict[str, str]]:
    """Render canonical criterion IDs without adding hidden evaluation rules."""
    if not isinstance(criteria, list) or not all(isinstance(item, str) for item in criteria):
        raise ManualReviewError("review protocol criterion IDs must be a list of strings")
    rendered: list[dict[str, str]] = []
    for item in criteria:
        action, expected = _CRITERION_DETAILS.get(item, ("Perform the stated acceptance check.", "Record the observed result."))
        rendered.append({"criterion_id": item, "label": item.replace("_", " ").capitalize(), "action": action, "expected": expected})
    return rendered


class ReviewDashboardServer(ThreadingHTTPServer):
    """An intentionally loopback-only dashboard with disposable restored apps."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], experiment_output: Path, experiment_definition: Path, subject_root: Path) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("manual review dashboard must bind only to 127.0.0.1")
        self.experiment_output = experiment_output.expanduser().resolve()
        self.experiment_definition = experiment_definition.expanduser().resolve()
        self.subject_root = subject_root.expanduser().resolve()
        self.runtime_root = Path(tempfile.mkdtemp(prefix="agent-bench-manual-review-"))
        self.revealed_blind_ids: set[str] = set()
        super().__init__(address, ReviewDashboardHandler)

    @property
    def review_storage_root(self) -> Path:
        return review_root(self.experiment_output)


class ReviewDashboardHandler(BaseHTTPRequestHandler):
    server: ReviewDashboardServer

    def log_message(self, _format: str, *_args: object) -> None:
        """Normal review activity is not terminal telemetry."""

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, _HTML, "text/html")
        elif path == "/api/next":
            self._json(HTTPStatus.OK, self._next())
        elif path == "/api/reveal":
            self._reveal(parse_qs(urlparse(self.path).query).get("blind_review_id", [""])[0])
        elif path.startswith("/app/"):
            self._serve_app(path)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._request_json()
            if self.path == "/api/prepare":
                self._prepare(payload)
            elif self.path == "/api/save":
                self._save(payload)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (KeyError, ManualReviewError, ValueError, OSError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _next(self) -> dict[str, Any]:
        queue = review_queue(self.server.experiment_output, self.server.experiment_definition, self.server.subject_root)
        reviewable = [row for row in queue if row["state"] == "completed"]
        row = next((item for item in reviewable if not item["reviewed"]), None)
        progress = {"total": len(reviewable), "reviewed": sum(bool(item["reviewed"]) for item in reviewable), "remaining": sum(not bool(item["reviewed"]) for item in reviewable)}
        if row is None:
            return {"done": True, "progress": progress}
        protocol, _digest = load_protocol(self.server.subject_root)
        task = str(row["semantic_task"])
        definition = protocol["tasks"].get(task)
        if not isinstance(definition, dict):
            raise ManualReviewError("review protocol has no definition for selected semantic task")
        # No canonical identity or harness metadata crosses the blind boundary.
        return {
            "done": False,
            "blind_review_id": row["blind_review_id"],
            "semantic_task": task,
            "progress": progress,
            "steps": human_steps(task),
            "task_criteria": human_criteria(definition.get("criteria")),
            "regression_criteria": human_criteria(protocol.get("common_regression_criteria")),
            "priority_flags": list(row["priority_flags"]),
        }

    def _prepare(self, payload: dict[str, Any]) -> None:
        blind = _required(payload, "blind_review_id")
        row = self._row(blind)
        destination = self.server.runtime_root / blind
        if not destination.exists():
            prepare_review_copy(self.server.experiment_output, str(row["run_id"]), destination, self.server.subject_root)
        self._json(HTTPStatus.OK, {"app_url": f"/app/{blind}/review-fixture.html", "reset_behavior": "clears localStorage and sessionStorage before reseeding the fixed fixture"})

    def _save(self, payload: dict[str, Any]) -> None:
        blind = _required(payload, "blind_review_id")
        row = self._row(blind)
        if row["reviewed"]:
            raise ManualReviewError("review exists; use deliberate CLI amend for a new immutable revision")
        protocol, digest = load_protocol(self.server.subject_root)
        run_id, task = str(row["run_id"]), str(row["semantic_task"])
        state = ExperimentState.model_validate_json((self.server.experiment_output / "experiment-state.json").read_bytes())
        review = ManualReview.create(
            review_id=f"{run_id}-manual-review-r001", experiment_id=state.experiment_id, run_id=run_id,
            semantic_task=task, review_protocol_id=str(protocol["review_protocol_id"]), review_protocol_digest=digest,
            reviewed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), reviewer_id=_required(payload, "reviewer_id"),
            blind_review_id=blind, functional_outcome=_required(payload, "functional_outcome"),
            task_criteria=_criteria(payload, "task_criteria"), regression_criteria=_criteria(payload, "regression_criteria"),
            regression_outcome=_required(payload, "regression_outcome"), review_completeness="complete", notes=_optional(payload.get("notes")),
            source_artifact_manifest_sha256=_sha256(self.server.experiment_output / "artifacts" / run_id / "manifest.json"), revision=1,
        )
        validate_review_against_protocol(review, self.server.subject_root)
        save_review(self.server.experiment_output, review)
        self.server.revealed_blind_ids.add(blind)
        self._json(HTTPStatus.CREATED, {"saved": True, "revision": 1, "next": self._next()})

    def _reveal(self, blind: str) -> None:
        if blind not in self.server.revealed_blind_ids:
            self._json(HTTPStatus.FORBIDDEN, {"error": "canonical metadata is available only after this dashboard saved the review"})
            return
        row = self._row(blind)
        self._json(HTTPStatus.OK, {key: row[key] for key in ("run_id", "semantic_task")})

    def _row(self, blind: str) -> dict[str, Any]:
        if not blind.startswith("blind-"):
            raise ManualReviewError("invalid blind review ID")
        for row in review_queue(self.server.experiment_output, self.server.experiment_definition, self.server.subject_root):
            if row["blind_review_id"] == blind:
                return row
        raise ManualReviewError("unknown blind review ID")

    def _serve_app(self, path: str) -> None:
        parts = path.split("/")
        if len(parts) < 4 or not parts[2].startswith("blind-"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        root = (self.server.runtime_root / parts[2]).resolve()
        target = root.joinpath(*parts[3:]).resolve()
        if (root not in target.parents and target != root) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(HTTPStatus.OK, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("request must contain a reasonably sized JSON body")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request JSON must be an object")
        return value

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, separators=(",", ":")), "application/json")

    def _send(self, status: HTTPStatus, content: str | bytes, content_type: str) -> None:
        body = content.encode("utf-8") if isinstance(content, str) else content
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _required(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _optional(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("notes must be a string")
    return value.strip() or None


def _criteria(payload: dict[str, Any], name: str) -> tuple[CriterionResult, ...]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return tuple(CriterionResult.model_validate(item) for item in value)


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent Bench review</title>
<style>body{font:16px/1.45 system-ui,sans-serif;max-width:58rem;margin:2rem auto;padding:0 1rem;color:#18212b}.panel,fieldset{border:1px solid #c5cdd4;border-radius:.4rem;padding:1rem;margin:1rem 0}.choice{border:1px solid #75818d;background:white;border-radius:.3rem;margin:.2rem;padding:.45rem .65rem}.choice.selected{background:#173b5e;color:white}.choice:focus{outline:3px solid #f6b73c;outline-offset:2px}.primary{background:#146c43;color:white;border:0;border-radius:.3rem;padding:.65rem 1rem}.primary:disabled{opacity:.45}.error{color:#a61b1b}textarea,input{width:100%;box-sizing:border-box;padding:.5rem}</style>
<h1>Blinded functional review</h1><p>The dashboard does not reveal harness, profile, prompt variant, repetition, or canonical run identity before saving.</p><main id="app" aria-live="polite">Loading…</main>
<script>
let current,answers={task_criteria:{},regression_criteria:{}},savedBlind=null;var overall,regression;const $=x=>document.getElementById(x),esc=s=>String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
async function api(url,opts={}){let r=await fetch(url,{headers:{'Content-Type':'application/json'},...opts}),x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x}
function pick(group,id,value){return `<button class="choice ${answers[group]?.[id]?.outcome===value?'selected':''}" type="button" onclick="setPick('${group}','${id}','${value}')">${value}</button>`}
function criterion(group,x){let a=answers[group][x.criterion_id]||{};return `<fieldset><legend>${esc(x.label)}</legend><b>Action:</b> ${esc(x.action)}<br><b>Expected:</b> ${esc(x.expected)}<p>${pick(group,x.criterion_id,'PASS')}${pick(group,x.criterion_id,'FAIL')}${pick(group,x.criterion_id,'UNREVIEWABLE')}</p>${a.outcome==='UNREVIEWABLE'?`<label>Why not reviewable? <input data-note="${group}:${x.criterion_id}" value="${esc(a.notes||'')}" oninput="note(this)"></label>`:''}</fieldset>`}
function oPick(variable,value){return `<button class="choice ${window[variable]===value?'selected':''}" type="button" onclick="window.${variable}='${value}';render()">${value}</button>`}
function done(){if(!current||current.done||!overall||!regression)return false;for(let g of ['task_criteria','regression_criteria'])for(let x of current[g]){let a=answers[g][x.criterion_id];if(!a||!a.outcome||(a.outcome==='UNREVIEWABLE'&&!a.notes?.trim()))return false}return true}
function revealPanel(){return savedBlind?`<section class="panel"><b>Saved immutable review.</b> <button type="button" onclick="reveal()">Reveal canonical metadata for that saved review</button><pre id="revealed"></pre></section>`:''}function render(){if(current.done){$('app').innerHTML=revealPanel()+`<section class="panel"><h2>All completed runs have reviews</h2><p>${current.progress.reviewed} reviewed.</p></section>`;return}let flags=current.priority_flags.length?current.priority_flags.map(esc).join(', '):'None';$('app').innerHTML=revealPanel()+`<section class="panel"><b>Review ${esc(current.blind_review_id)}</b><br>Completed ${current.progress.reviewed} of ${current.progress.total}; ${current.progress.remaining} remaining.</section><section class="panel"><h2>${esc(current.semantic_task)}</h2><button class="primary" type="button" onclick="openApp()">Open isolated app</button> <button type="button" onclick="openApp()">RESET TEST STATE</button><p>Reset clears localStorage and sessionStorage, restores Salary €2,000.00, Groceries −€125.50, Train pass −€49.50, then opens the app. It never changes the sealed result.</p><details><summary>Optional execution review flags</summary>${flags}</details><h3>Acceptance script</h3><ol>${current.steps.map(x=>`<li><b>${esc(x.label)}</b><br>Action: ${esc(x.action)}<br>Expected: ${esc(x.expected)}</li>`).join('')}</ol></section><h2>Task criteria</h2>${current.task_criteria.map(x=>criterion('task_criteria',x)).join('')}<h2>Common regression checklist</h2>${current.regression_criteria.map(x=>criterion('regression_criteria',x)).join('')}<fieldset><legend>Overall functional outcome</legend>${['PASS','MOSTLY_PASS','PARTIAL','FAIL','UNREVIEWABLE'].map(x=>oPick('overall',x)).join(' ')}</fieldset><fieldset><legend>Overall regression outcome</legend>${['PASS','MINOR_REGRESSION','MAJOR_REGRESSION','UNREVIEWABLE'].map(x=>oPick('regression',x)).join(' ')}</fieldset><label>Reviewer ID <input id="reviewer" value="local-reviewer"></label><p><label>Optional overall notes <textarea id="notes"></textarea></label></p><p id="error" class="error"></p><button id="save" class="primary" type="button" onclick="save()" ${done()?'':'disabled'}>Save review and next (Alt+S)</button>`}
function setPick(g,id,value){answers[g][id]={outcome:value,notes:answers[g][id]?.notes||''};render()}function note(x){let[g,id]=x.dataset.note.split(':');answers[g][id].notes=x.value;$('save').disabled=!done()}
async function openApp(){try{let x=await api('/api/prepare',{method:'POST',body:JSON.stringify({blind_review_id:current.blind_review_id})});window.open(x.app_url,'agent-bench-review-app')}catch(e){alert(e.message)}}
async function save(){if(!done())return;try{let reviewer=$('reviewer').value.trim();if(!reviewer)throw Error('Reviewer ID is required.');let blind=current.blind_review_id,x=await api('/api/save',{method:'POST',body:JSON.stringify({blind_review_id:blind,reviewer_id:reviewer,functional_outcome:overall,regression_outcome:regression,task_criteria:Object.entries(answers.task_criteria).map(([criterion_id,v])=>({criterion_id,...v})),regression_criteria:Object.entries(answers.regression_criteria).map(([criterion_id,v])=>({criterion_id,...v})),notes:$('notes').value})});savedBlind=blind;current=x.next;answers={task_criteria:{},regression_criteria:{}};overall=regression=null;render()}catch(e){$('error').textContent=e.message}}async function reveal(){try{let x=await api('/api/reveal?blind_review_id='+encodeURIComponent(savedBlind));$('revealed').textContent=JSON.stringify(x,null,2)}catch(e){alert(e.message)}}
document.addEventListener('keydown',e=>{if(e.altKey&&e.key.toLowerCase()==='s'){e.preventDefault();if(done())save()}});(async()=>{try{current=await api('/api/next');render()}catch(e){$('app').innerHTML=`<p class="error">${esc(e.message)}</p>`}})();
</script></html>'''
