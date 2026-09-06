import { TaskBoard, describeTask } from "./src/taskboard.js";

const board = new TaskBoard(localStorage);
const form = document.querySelector("#task-form"), title = document.querySelector("#task-title"), description = document.querySelector("#task-description"), status = document.querySelector("#task-status"), priority = document.querySelector("#task-priority"), search = document.querySelector("#search-filter"), statusFilter = document.querySelector("#status-filter"), priorityFilter = document.querySelector("#priority-filter"), clear = document.querySelector("#clear-filters"), list = document.querySelector("#task-list"), empty = document.querySelector("#empty-state"), error = document.querySelector("#form-error");

function render() {
  search.value = board.filters.query; statusFilter.value = board.filters.status; priorityFilter.value = board.filters.priority; list.replaceChildren();
  for (const task of board.visibleTasks()) {
    const item = document.createElement("li"), text = document.createElement("span"), edit = document.createElement("button"), remove = document.createElement("button");
    text.className = "task-title"; text.textContent = describeTask(task); edit.type = remove.type = "button"; edit.textContent = "Edit"; remove.textContent = "Delete";
    edit.addEventListener("click", () => { const nextTitle = prompt("Task title", task.title); if (nextTitle === null) return; const nextDescription = prompt("Description", task.description); if (nextDescription === null) return; const nextStatus = prompt("Status (Todo, Doing, or Done)", task.status); if (nextStatus === null) return; const nextPriority = prompt("Priority (Low, Medium, or High)", task.priority); if (nextPriority === null) return; try { board.editTask(task.id, { title: nextTitle, description: nextDescription, status: nextStatus, priority: nextPriority }); render(); } catch (problem) { error.textContent = problem.message; } });
    remove.addEventListener("click", () => { board.deleteTask(task.id); render(); }); item.append(text, edit, remove); list.append(item);
  }
  empty.hidden = board.visibleTasks().length > 0;
}
form.addEventListener("submit", (event) => { event.preventDefault(); try { board.createTask({ title: title.value, description: description.value, status: status.value, priority: priority.value }); form.reset(); render(); } catch (problem) { error.textContent = problem.message; } });
function updateFilters() { board.setFilters({ query: search.value, status: statusFilter.value, priority: priorityFilter.value }); render(); }
search.addEventListener("input", updateFilters); statusFilter.addEventListener("change", updateFilters); priorityFilter.addEventListener("change", updateFilters); clear.addEventListener("click", () => { board.clearFilters(); render(); }); render();
