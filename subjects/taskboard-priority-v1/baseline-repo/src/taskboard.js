import { TASK_STORAGE_KEY, STATUSES, validPriority, validStatus } from "./constants.js";

function taskId() { return `task-${Date.now()}-${Math.random().toString(16).slice(2)}`; }

function normalizeTask(value) {
  if (!value || typeof value !== "object" || typeof value.id !== "string" || typeof value.title !== "string") return null;
  return { id: value.id, title: value.title, status: validStatus(value.status) ? value.status : "Todo", priority: validPriority(value.priority) ? value.priority : "Medium" };
}

export function describeTask(task) { return `${task.title} · ${task.status} · ${task.priority}`; }

export class TaskBoard {
  constructor(storage) { this.storage = storage; this.filter = "all"; this.tasks = this.load(); }
  load() { try { const stored = JSON.parse(this.storage.getItem(TASK_STORAGE_KEY) || "[]"); return Array.isArray(stored) ? stored.map(normalizeTask).filter(Boolean) : []; } catch { return []; } }
  save() { this.storage.setItem(TASK_STORAGE_KEY, JSON.stringify(this.tasks)); }
  createTask({ title, status = "Todo", priority = "Medium" }) { const cleanTitle = typeof title === "string" ? title.trim() : ""; if (!cleanTitle || !validStatus(status) || !validPriority(priority)) throw new Error("A task title, valid status, and valid priority are required."); const task = { id: taskId(), title: cleanTitle, status, priority }; this.tasks.unshift(task); this.save(); return task; }
  editTask(id, changes) { const task = this.tasks.find((item) => item.id === id); if (!task) throw new Error("Task not found."); if ("title" in changes) { const title = String(changes.title).trim(); if (!title) throw new Error("Task title is required."); task.title = title; } if ("status" in changes) { if (!validStatus(changes.status)) throw new Error("Invalid task status."); task.status = changes.status; } if ("priority" in changes) { if (!validPriority(changes.priority)) throw new Error("Invalid task priority."); task.priority = changes.priority; } this.save(); return task; }
  deleteTask(id) { const before = this.tasks.length; this.tasks = this.tasks.filter((task) => task.id !== id); if (before !== this.tasks.length) this.save(); return before !== this.tasks.length; }
  changeStatus(id, status) { return this.editTask(id, { status }); }
  setFilter(filter) { if (filter !== "all" && !STATUSES.includes(filter)) throw new Error("Invalid task filter."); this.filter = filter; }
  visibleTasks() { return this.filter === "all" ? [...this.tasks] : this.tasks.filter((task) => task.status === this.filter); }
}
