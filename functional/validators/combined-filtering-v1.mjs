#!/usr/bin/env node
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import path from "node:path";

const workspace = process.argv[2];
if (!workspace) throw new Error("usage: combined-filtering-v1.mjs WORKSPACE");
const root = path.resolve(workspace);
const { TASK_STORAGE_KEY } = await import(pathToFileURL(path.join(root, "src", "constants.js")).href);
const { TaskBoard } = await import(pathToFileURL(path.join(root, "src", "taskboard.js")).href + `?validation=${Date.now()}`);
const indexHtml = readFileSync(path.join(root, "index.html"), "utf8");

class MemoryStorage { constructor(values = new Map()) { this.values = values; } getItem(key) { return this.values.get(key) ?? null; } setItem(key, value) { this.values.set(key, value); } }
function fresh() { const storage = new MemoryStorage(); return { storage, board: new TaskBoard(storage) }; }
function add(board, title, description, status = "Todo", priority = "Medium") { return board.createTask({ title, description, status, priority }); }
function names(board) { return board.visibleTasks().map((task) => task.title); }
function test(test_id, category, fn) { try { fn(); return { test_id, category, outcome: "passed", detail: "" }; } catch (error) { return { test_id, category, outcome: "failed", detail: error instanceof Error ? error.message : String(error) }; } }

const tests = [
  test("baseline-initializes", "baseline_regression", () => { assert.deepEqual(fresh().board.tasks, []); }),
  test("baseline-create", "baseline_regression", () => { assert.equal(add(fresh().board, "Create", "notes").title, "Create"); }),
  test("baseline-edit", "baseline_regression", () => { const { board } = fresh(); const task = add(board, "Draft", "notes"); assert.equal(board.editTask(task.id, { title: "Reviewed" }).title, "Reviewed"); }),
  test("baseline-delete", "baseline_regression", () => { const { board } = fresh(); const task = add(board, "Delete", "notes"); assert.equal(board.deleteTask(task.id), true); assert.equal(board.tasks.length, 0); }),
  test("baseline-status-change", "baseline_regression", () => { const { board } = fresh(); const task = add(board, "Move", "notes"); assert.equal(board.changeStatus(task.id, "Done").status, "Done"); }),
  test("baseline-persistence-reload", "baseline_regression", () => { const { storage, board } = fresh(); add(board, "Reload", "notes", "Doing", "High"); assert.equal(new TaskBoard(storage).tasks[0].title, "Reload"); }),
  test("baseline-status-filter", "baseline_regression", () => { const { board } = fresh(); add(board, "Todo item", "", "Todo"); add(board, "Done item", "", "Done"); board.setFilter("Done"); assert.deepEqual(names(board), ["Done item"]); }),
  test("baseline-priority-persistence", "baseline_regression", () => { const { storage, board } = fresh(); add(board, "Priority", "", "Todo", "High"); assert.equal(new TaskBoard(storage).tasks[0].priority, "High"); }),
  test("search-title", "feature_requirement", () => { const { board } = fresh(); add(board, "Plan release", ""); add(board, "Buy milk", ""); board.setFilters({ query: "release" }); assert.deepEqual(names(board), ["Plan release"]); }),
  test("search-description", "feature_requirement", () => { const { board } = fresh(); add(board, "One", "Needs customer review"); add(board, "Two", "Internal"); board.setFilters({ query: "customer" }); assert.deepEqual(names(board), ["One"]); }),
  test("search-case-insensitive", "feature_requirement", () => { const { board } = fresh(); add(board, "Release notes", ""); board.setFilters({ query: "ReLeAsE" }); assert.deepEqual(names(board), ["Release notes"]); }),
  test("search-hides-nonmatches", "feature_requirement", () => { const { board } = fresh(); add(board, "Match", "needle"); add(board, "Hide", "haystack"); board.setFilters({ query: "needle" }); assert.deepEqual(names(board), ["Match"]); }),
  test("search-clear-restores", "feature_requirement", () => { const { board } = fresh(); add(board, "One", "needle"); add(board, "Two", "haystack"); board.setFilters({ query: "needle" }); board.setFilters({ query: "" }); assert.deepEqual(names(board).sort(), ["One", "Two"]); }),
  test("filters-controls-visible", "feature_requirement", () => { for (const label of ["Search", "Status filter", "Priority filter", "Clear filters"]) assert.match(indexHtml, new RegExp(label, "i")); }),
  test("filter-status", "feature_requirement", () => { const { board } = fresh(); add(board, "Todo", "", "Todo"); add(board, "Done", "", "Done"); board.setFilters({ status: "Done" }); assert.deepEqual(names(board), ["Done"]); }),
  test("filter-priority", "feature_requirement", () => { const { board } = fresh(); add(board, "Low", "", "Todo", "Low"); add(board, "High", "", "Todo", "High"); board.setFilters({ priority: "High" }); assert.deepEqual(names(board), ["High"]); }),
  test("combine-search-status", "feature_requirement", () => { const { board } = fresh(); add(board, "Release todo", "", "Todo"); add(board, "Release done", "", "Done"); add(board, "Other done", "", "Done"); board.setFilters({ query: "release", status: "Done" }); assert.deepEqual(names(board), ["Release done"]); }),
  test("combine-search-priority", "feature_requirement", () => { const { board } = fresh(); add(board, "Release low", "", "Todo", "Low"); add(board, "Release high", "", "Todo", "High"); add(board, "Other high", "", "Todo", "High"); board.setFilters({ query: "release", priority: "High" }); assert.deepEqual(names(board), ["Release high"]); }),
  test("combine-status-priority", "feature_requirement", () => { const { board } = fresh(); add(board, "Done low", "", "Done", "Low"); add(board, "Done high", "", "Done", "High"); add(board, "Todo high", "", "Todo", "High"); board.setFilters({ status: "Done", priority: "High" }); assert.deepEqual(names(board), ["Done high"]); }),
  test("combine-all-filters", "feature_requirement", () => { const { board } = fresh(); add(board, "Release done high", "", "Done", "High"); add(board, "Release todo high", "", "Todo", "High"); add(board, "Release done low", "", "Done", "Low"); board.setFilters({ query: "release", status: "Done", priority: "High" }); assert.deepEqual(names(board), ["Release done high"]); }),
  test("combine-and-not-or", "feature_requirement", () => { const { board } = fresh(); add(board, "Exact", "release", "Done", "High"); add(board, "Search only", "release", "Todo", "Low"); add(board, "Status only", "other", "Done", "Low"); add(board, "Priority only", "other", "Todo", "High"); board.setFilters({ query: "release", status: "Done", priority: "High" }); assert.deepEqual(names(board), ["Exact"]); }),
  test("filter-state-search-persists", "feature_requirement", () => { const { storage, board } = fresh(); add(board, "Search state", "needle"); board.setFilters({ query: "needle" }); assert.equal(new TaskBoard(storage).filters.query, "needle"); }),
  test("filter-state-status-persists", "feature_requirement", () => { const { storage, board } = fresh(); board.setFilters({ status: "Done" }); assert.equal(new TaskBoard(storage).filters.status, "Done"); }),
  test("filter-state-priority-persists", "feature_requirement", () => { const { storage, board } = fresh(); board.setFilters({ priority: "High" }); assert.equal(new TaskBoard(storage).filters.priority, "High"); }),
  test("filter-state-reload-visible", "feature_requirement", () => { const { storage, board } = fresh(); add(board, "Plan", "release", "Done", "High"); add(board, "Other", "", "Todo", "Low"); board.setFilters({ query: "plan", status: "Done", priority: "High" }); const reloaded = new TaskBoard(storage); assert.deepEqual(reloaded.filters, { query: "plan", status: "Done", priority: "High" }); assert.deepEqual(names(reloaded), ["Plan"]); }),
  test("filter-edit-enters", "feature_requirement", () => { const { board } = fresh(); const task = add(board, "Enter", "", "Todo", "Low"); board.setFilters({ priority: "High" }); assert.deepEqual(names(board), []); board.editTask(task.id, { priority: "High" }); assert.deepEqual(names(board), ["Enter"]); }),
  test("filter-edit-leaves", "feature_requirement", () => { const { board } = fresh(); const task = add(board, "Leave", "", "Todo", "High"); board.setFilters({ priority: "High" }); board.editTask(task.id, { priority: "Low" }); assert.deepEqual(names(board), []); }),
  test("filter-delete-active", "feature_requirement", () => { const { board } = fresh(); const task = add(board, "Delete", "", "Todo", "High"); board.setFilters({ priority: "High" }); assert.equal(board.deleteTask(task.id), true); assert.deepEqual(names(board), []); }),
  test("filter-clear-all", "feature_requirement", () => { const { board } = fresh(); add(board, "One", "needle", "Done", "High"); add(board, "Two", "", "Todo", "Low"); board.setFilters({ query: "needle", status: "Done", priority: "High" }); board.clearFilters(); assert.deepEqual(board.filters, { query: "", status: "all", priority: "all" }); assert.deepEqual(names(board).sort(), ["One", "Two"]); }),
  test("edge-zero-results-recover", "edge_case", () => { const { board } = fresh(); add(board, "Recover", "needle"); board.setFilters({ query: "missing" }); assert.deepEqual(names(board), []); board.setFilters({ query: "needle" }); assert.deepEqual(names(board), ["Recover"]); }),
];
process.stdout.write(JSON.stringify({ tests }) + "\n");
