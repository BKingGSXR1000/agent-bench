#!/usr/bin/env node
// Evaluator-owned v2 acceptance suite.  It deliberately avoids requiring a
// particular rendering helper or DOM shape for per-row priority presentation.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const workspace = process.argv[2];
if (!workspace) throw new Error("usage: task-priority-v2.mjs WORKSPACE");
const moduleUrl = pathToFileURL(path.join(workspace, "src", "taskboard.js")).href + `?validation=${Date.now()}`;
const { TASK_STORAGE_KEY } = await import(pathToFileURL(path.join(workspace, "src", "constants.js")).href);
const { TaskBoard } = await import(moduleUrl);

class MemoryStorage { constructor(values = new Map()) { this.values = values; } getItem(key) { return this.values.get(key) ?? null; } setItem(key, value) { this.values.set(key, value); } }
function fresh() { const storage = new MemoryStorage(); return { storage, board: new TaskBoard(storage) }; }
function test(test_id, category, fn) { try { fn(); return { test_id, category, outcome: "passed", detail: "" }; } catch (error) { return { test_id, category, outcome: "failed", detail: error instanceof Error ? error.message : String(error) }; } }
function manual(test_id, detail) { return { test_id, category: "feature_requirement", outcome: "manual_review_required", detail }; }

const tests = [
  test("baseline-initializes", "baseline_regression", () => { const { board } = fresh(); assert.deepEqual(board.tasks, []); }),
  test("baseline-create", "baseline_regression", () => { const { board } = fresh(); assert.equal(board.createTask({ title: "Write tests", status: "Todo" }).title, "Write tests"); }),
  test("baseline-edit", "baseline_regression", () => { const { board } = fresh(); const task = board.createTask({ title: "Draft", status: "Todo" }); assert.equal(board.editTask(task.id, { title: "Review draft" }).title, "Review draft"); }),
  test("baseline-delete", "baseline_regression", () => { const { board } = fresh(); const task = board.createTask({ title: "Remove", status: "Todo" }); assert.equal(board.deleteTask(task.id), true); assert.equal(board.tasks.length, 0); }),
  test("baseline-status-change", "baseline_regression", () => { const { board } = fresh(); const task = board.createTask({ title: "Move", status: "Todo" }); assert.equal(board.changeStatus(task.id, "Done").status, "Done"); }),
  test("baseline-persistence-reload", "baseline_regression", () => { const { storage, board } = fresh(); board.createTask({ title: "Reload", status: "Doing" }); assert.equal(new TaskBoard(storage).tasks[0].status, "Doing"); }),
  test("baseline-status-filter", "baseline_regression", () => { const { board } = fresh(); board.createTask({ title: "One", status: "Todo" }); board.createTask({ title: "Two", status: "Done" }); board.setFilter("Done"); assert.deepEqual(board.visibleTasks().map((task) => task.title), ["Two"]); }),
  test("priority-create-low", "feature_requirement", () => { const { board } = fresh(); assert.equal(board.createTask({ title: "Low", status: "Todo", priority: "Low" }).priority, "Low"); }),
  test("priority-create-medium", "feature_requirement", () => { const { board } = fresh(); assert.equal(board.createTask({ title: "Medium", status: "Todo", priority: "Medium" }).priority, "Medium"); }),
  test("priority-create-high", "feature_requirement", () => { const { board } = fresh(); assert.equal(board.createTask({ title: "High", status: "Todo", priority: "High" }).priority, "High"); }),
  manual("priority-input-visible", "Priority creation control is visual UI evidence; this evaluator has no implementation-neutral browser/DOM probe."),
  manual("priority-displayed", "Per-task priority presentation is visual UI evidence; it may be text, a select, or another editing control."),
  test("priority-persists", "feature_requirement", () => { const { storage, board } = fresh(); board.createTask({ title: "Persist", status: "Todo", priority: "High" }); assert.equal(new TaskBoard(storage).tasks[0].priority, "High"); }),
  test("priority-edit", "feature_requirement", () => { const { storage, board } = fresh(); const task = board.createTask({ title: "Edit", status: "Todo", priority: "Low" }); assert.equal(board.editTask(task.id, { priority: "High" }).priority, "High"); assert.equal(board.tasks[0].priority, "High"); assert.equal(new TaskBoard(storage).tasks[0].priority, "High"); }),
  test("edge-legacy-priority", "edge_case", () => { const storage = new MemoryStorage(new Map([[TASK_STORAGE_KEY, JSON.stringify([{ id: "legacy", title: "Before priority", status: "Todo" }])]])); const board = new TaskBoard(storage); assert.equal(board.tasks[0].priority, "Medium"); assert.equal(board.changeStatus("legacy", "Doing").status, "Doing"); }),
  test("edge-invalid-stored-priority", "edge_case", () => { const storage = new MemoryStorage(new Map([[TASK_STORAGE_KEY, JSON.stringify([{ id: "bad", title: "Bad priority", status: "Todo", priority: "Urgent" }])]])); const board = new TaskBoard(storage); assert.equal(board.tasks[0].priority, "Medium"); assert.equal(board.visibleTasks().length, 1); }),
];
process.stdout.write(JSON.stringify({ tests }) + "\n");
