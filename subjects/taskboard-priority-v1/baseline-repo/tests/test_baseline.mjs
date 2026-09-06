import assert from "node:assert/strict";
import { TaskBoard } from "../src/taskboard.js";
class MemoryStorage { constructor() { this.values = new Map(); } getItem(key) { return this.values.get(key) ?? null; } setItem(key, value) { this.values.set(key, value); } }
const storage = new MemoryStorage(), board = new TaskBoard(storage), task = board.createTask({ title: "Plan release", status: "Todo", priority: "High" });
assert.equal(task.priority, "High"); assert.equal(board.editTask(task.id, { priority: "Low" }).priority, "Low"); assert.equal(new TaskBoard(storage).tasks[0].priority, "Low"); board.changeStatus(task.id, "Doing"); board.setFilter("Doing"); assert.equal(board.visibleTasks().length, 1); assert.equal(board.deleteTask(task.id), true); console.log("taskboard priority baseline checks passed");
