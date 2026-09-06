import { TaskBoard, describeTask } from "./src/taskboard.js";

const board = new TaskBoard(localStorage);
const form = document.querySelector("#task-form");
const title = document.querySelector("#task-title");
const status = document.querySelector("#task-status");
const priority = document.querySelector("#task-priority");
const filter = document.querySelector("#status-filter");
const list = document.querySelector("#task-list");
const empty = document.querySelector("#empty-state");
const error = document.querySelector("#form-error");

function render() {
  list.replaceChildren();
  for (const task of board.visibleTasks()) {
    const item = document.createElement("li");
    const text = document.createElement("span"); text.className = "task-title"; text.textContent = describeTask(task);
    const actions = document.createElement("span"); actions.className = "task-actions";
    const next = document.createElement("button"); next.type = "button"; next.textContent = "Advance";
    next.addEventListener("click", () => { const index = ["Todo", "Doing", "Done"].indexOf(task.status); board.changeStatus(task.id, ["Todo", "Doing", "Done"][(index + 1) % 3]); render(); });
    const edit = document.createElement("button"); edit.type = "button"; edit.textContent = "Edit";
    edit.addEventListener("click", () => { const updated = prompt("Task title", task.title); if (updated !== null) { const updatedPriority = prompt("Priority (Low, Medium, or High)", task.priority); if (updatedPriority !== null) { try { board.editTask(task.id, { title: updated, priority: updatedPriority }); render(); } catch (problem) { error.textContent = problem.message; } } } });
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "delete"; remove.textContent = "Delete";
    remove.addEventListener("click", () => { board.deleteTask(task.id); render(); });
    actions.append(next, edit, remove); item.append(text, actions); list.append(item);
  }
  empty.hidden = board.visibleTasks().length > 0;
}

form.addEventListener("submit", (event) => { event.preventDefault(); try { board.createTask({ title: title.value, status: status.value, priority: priority.value }); form.reset(); status.value = "Todo"; priority.value = "Medium"; error.textContent = ""; render(); title.focus(); } catch (problem) { error.textContent = problem.message; } });
filter.addEventListener("change", () => { board.setFilter(filter.value); render(); });
render();
