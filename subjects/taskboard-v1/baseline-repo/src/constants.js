export const TASK_STORAGE_KEY = "taskboard.tasks.v1";
export const STATUSES = Object.freeze(["Todo", "Doing", "Done"]);

export function validStatus(value) {
  return STATUSES.includes(value);
}
