import assert from "node:assert/strict";
import { TaskBoard } from "../src/taskboard.js";

class MemoryStorage { constructor() { this.values = new Map(); } getItem(key) { return this.values.get(key) ?? null; } setItem(key, value) { this.values.set(key, value); } }
const storage = new MemoryStorage();
const board = new TaskBoard(storage);
const task = board.createTask({ title: "Plan release", status: "Todo" });
board.editTask(task.id, { title: "Plan M12 release" });
board.changeStatus(task.id, "Doing");
assert.deepEqual(new TaskBoard(storage).tasks[0].title, "Plan M12 release");
board.setFilter("Doing"); assert.equal(board.visibleTasks().length, 1);
assert.equal(board.deleteTask(task.id), true);
console.log("taskboard baseline checks passed");
