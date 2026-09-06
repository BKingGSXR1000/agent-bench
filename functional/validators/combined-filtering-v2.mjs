#!/usr/bin/env node
// v2 preserves the frozen priority baseline contract, but UI-driven filtering
// behavior is manual evidence until an implementation-neutral browser layer is
// available. It deliberately does not prescribe filter API or DOM names.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const workspace = process.argv[2];
if (!workspace) throw new Error("usage: combined-filtering-v2.mjs WORKSPACE");
const root = path.resolve(workspace);
const { TaskBoard } = await import(pathToFileURL(path.join(root, "src", "taskboard.js")).href + `?validation=${Date.now()}`);

class MemoryStorage { constructor(values = new Map()) { this.values = values; } getItem(key) { return this.values.get(key) ?? null; } setItem(key, value) { this.values.set(key, value); } }
function fresh() { const storage = new MemoryStorage(); return { storage, board: new TaskBoard(storage) }; }
function add(board, title, description = "", status = "Todo", priority = "Medium") { return board.createTask({ title, description, status, priority }); }
function names(board) { return board.visibleTasks().map((task) => task.title); }
function test(test_id, fn) { try { fn(); return { test_id, category: "baseline_regression", outcome: "passed", detail: "" }; } catch (error) { return { test_id, category: "baseline_regression", outcome: "failed", detail: error instanceof Error ? error.message : String(error) }; } }
function manual(test_id, category, detail) { return { test_id, category, outcome: "manual_review_required", detail }; }

const tests = [
  test("baseline-initializes", () => { assert.deepEqual(fresh().board.tasks, []); }),
  test("baseline-create", () => { assert.equal(add(fresh().board, "Create", "notes").title, "Create"); }),
  test("baseline-edit", () => { const { board } = fresh(); const task = add(board, "Draft", "notes"); assert.equal(board.editTask(task.id, { title: "Reviewed" }).title, "Reviewed"); }),
  test("baseline-delete", () => { const { board } = fresh(); const task = add(board, "Delete", "notes"); assert.equal(board.deleteTask(task.id), true); assert.equal(board.tasks.length, 0); }),
  test("baseline-status-change", () => { const { board } = fresh(); const task = add(board, "Move", "notes"); assert.equal(board.changeStatus(task.id, "Done").status, "Done"); }),
  test("baseline-persistence-reload", () => { const { storage, board } = fresh(); add(board, "Reload", "notes", "Doing", "High"); assert.equal(new TaskBoard(storage).tasks[0].title, "Reload"); }),
  test("baseline-status-filter", () => { const { board } = fresh(); add(board, "Todo item", "", "Todo"); add(board, "Done item", "", "Done"); board.setFilter("Done"); assert.deepEqual(names(board), ["Done item"]); }),
  test("baseline-priority-persistence", () => { const { storage, board } = fresh(); add(board, "Priority", "", "Todo", "High"); assert.equal(new TaskBoard(storage).tasks[0].priority, "High"); }),
  manual("search-title", "feature_requirement", "Check that search finds task titles."),
  manual("search-description", "feature_requirement", "Check that search finds task descriptions."),
  manual("search-case-insensitive", "feature_requirement", "Check case-insensitive search."),
  manual("search-hides-nonmatches", "feature_requirement", "Check nonmatching tasks are hidden."),
  manual("search-clear-restores", "feature_requirement", "Check clearing search restores matching visibility."),
  manual("filters-controls-visible", "feature_requirement", "Check search, status, priority, and clear-all controls exist and work; wording and markup are unconstrained."),
  manual("filter-status", "feature_requirement", "Check status filtering."),
  manual("filter-priority", "feature_requirement", "Check priority filtering."),
  manual("combine-search-status", "feature_requirement", "Check search plus status uses AND semantics."),
  manual("combine-search-priority", "feature_requirement", "Check search plus priority uses AND semantics."),
  manual("combine-status-priority", "feature_requirement", "Check status plus priority uses AND semantics."),
  manual("combine-all-filters", "feature_requirement", "Check all three filters work together with AND semantics."),
  manual("combine-and-not-or", "feature_requirement", "Check a task must satisfy every active filter."),
  manual("filter-state-search-persists", "feature_requirement", "Check search state persists after reload."),
  manual("filter-state-status-persists", "feature_requirement", "Check status state persists after reload."),
  manual("filter-state-priority-persists", "feature_requirement", "Check priority state persists after reload."),
  manual("filter-state-reload-visible", "feature_requirement", "Check restored filter state visibly affects the reloaded application."),
  manual("filter-edit-enters", "feature_requirement", "Check editing can make an item enter an active filter."),
  manual("filter-edit-leaves", "feature_requirement", "Check editing can make an item leave an active filter."),
  manual("filter-delete-active", "feature_requirement", "Check deletion while filtered works."),
  manual("filter-clear-all", "feature_requirement", "Check clearing all filters restores tasks."),
  manual("edge-zero-results-recover", "edge_case", "Check a zero-results state can recover after changing or clearing filters."),
];
process.stdout.write(JSON.stringify({ tests }) + "\n");
