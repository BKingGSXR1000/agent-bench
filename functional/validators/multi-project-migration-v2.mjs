#!/usr/bin/env node
// v2 tests only the pre-task TaskBoard contract. Project, migration, and
// import/export behavior is manual evidence until a neutral interaction layer
// exists; no internal project schema or method names are assumed.
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import path from "node:path";

const workspace = process.argv[2];
if (!workspace) throw new Error("usage: multi-project-migration-v2.mjs WORKSPACE");
const root = path.resolve(workspace);
const { TaskBoard } = await import(pathToFileURL(path.join(root, "src", "taskboard.js")).href + `?validation=${Date.now()}`);
class Storage { constructor(values = new Map()) { this.data = values; } getItem(key) { return this.data.get(key) ?? null; } setItem(key, value) { this.data.set(key, value); } }
const fresh = () => { const storage = new Storage(); return { storage, board: new TaskBoard(storage) }; };
const add = (board, title, description = "", status = "Todo", priority = "Medium") => board.createTask({ title, description, status, priority });
const names = (board) => board.visibleTasks().map((task) => task.title);
function test(test_id, fn) { try { fn(); return { test_id, category: "baseline_regression", outcome: "passed", detail: "" }; } catch (error) { return { test_id, category: "baseline_regression", outcome: "failed", detail: error instanceof Error ? error.message : String(error) }; } }
function manual(test_id, category, detail) { return { test_id, category, outcome: "manual_review_required", detail }; }

const tests = [
  test("baseline-initializes", () => assert.deepEqual(fresh().board.tasks, [])),
  test("baseline-create", () => assert.equal(add(fresh().board, "Create").title, "Create")),
  test("baseline-edit", () => { const { board } = fresh(); const task = add(board, "Draft"); assert.equal(board.editTask(task.id, { title: "Edited" }).title, "Edited"); }),
  test("baseline-delete", () => { const { board } = fresh(); const task = add(board, "Delete"); assert.equal(board.deleteTask(task.id), true); assert.equal(board.tasks.length, 0); }),
  test("baseline-status", () => { const { board } = fresh(); const task = add(board, "Move"); assert.equal(board.changeStatus(task.id, "Done").status, "Done"); }),
  test("baseline-priority", () => { const { storage, board } = fresh(); add(board, "High", "", "Todo", "High"); assert.equal(new TaskBoard(storage).tasks[0].priority, "High"); }),
  test("baseline-combined-filtering", () => { const { board } = fresh(); add(board, "Match", "release", "Done", "High"); add(board, "Hide", "", "Todo", "Low"); board.setFilters({ query: "release", status: "Done", priority: "High" }); assert.deepEqual(names(board), ["Match"]); }),
  test("baseline-filter-persistence", () => { const { storage, board } = fresh(); board.setFilters({ query: "a", status: "Done", priority: "High" }); assert.equal(new TaskBoard(storage).filters.priority, "High"); }),
  test("baseline-reload", () => { const { storage, board } = fresh(); add(board, "Reload"); assert.equal(new TaskBoard(storage).tasks.length, 1); }),
  manual("project-create", "feature_requirement", "Check creating a persistent project."),
  manual("project-rename", "feature_requirement", "Check renaming a project."),
  manual("project-switch-persist", "feature_requirement", "Check switching the active project and persistence after reload."),
  manual("project-task-belongs-stable-id", "feature_requirement", "Check tasks belong to their intended project and task identifiers remain stable."),
  manual("project-isolation", "feature_requirement", "Check switching projects hides tasks from other projects."),
  manual("project-cross-mutation", "feature_requirement", "Check editing or deleting in one project does not corrupt another."),
  manual("archive-project-preserves-data", "feature_requirement", "Check archived project data is retained and exportable."),
  manual("archive-selection", "feature_requirement", "Check archiving changes the active selection appropriately."),
  manual("migration-empty", "feature_requirement", "Check old single-project saved data loads with a deterministic default project."),
  manual("migration-populated-preserves", "feature_requirement", "Check migration preserves existing task IDs and fields."),
  manual("migration-reload-no-duplicate", "feature_requirement", "Check repeated reload does not duplicate migrated tasks."),
  manual("migration-already-new", "feature_requirement", "Check already migrated project data remains usable after reload."),
  manual("export-includes-state", "feature_requirement", "Check export captures projects, tasks, active project, archived data, and relevant filters."),
  manual("import-roundtrip", "feature_requirement", "Check import/export round trip preserves projects, tasks, relationships, active project, and filters."),
  manual("import-malformed-rejected", "feature_requirement", "Check malformed import is rejected; error wording is unconstrained."),
  manual("import-invalid-structure", "feature_requirement", "Check invalid import is rejected; serialized layout is unconstrained."),
  manual("import-invalid-reference", "feature_requirement", "Check invalid task/project relationships are rejected."),
  manual("import-failed-atomic", "feature_requirement", "Check a rejected import does not corrupt existing data."),
  manual("export-archived-survives", "feature_requirement", "Check archived projects and their task data survive a round trip."),
  manual("interaction-filter-project-isolation", "feature_requirement", "Check filters apply only to the active project."),
  manual("interaction-switch-with-filters", "feature_requirement", "Check project switching with active filters remains isolated."),
  test("interaction-edit-visibility", () => { const { board } = fresh(); const task = add(board, "Task", "hide"); board.setFilters({ query: "needle" }); board.editTask(task.id, { description: "needle" }); assert.deepEqual(names(board), ["Task"]); }),
  test("interaction-delete-filtered", () => { const { board } = fresh(); const task = add(board, "Task", "needle"); board.setFilters({ query: "needle" }); assert.equal(board.deleteTask(task.id), true); assert.deepEqual(names(board), []); }),
  manual("interaction-migration-create-project", "edge_case", "Check migrated data remains usable when a new project is created and selected."),
];
process.stdout.write(JSON.stringify({ tests }) + "\n");
