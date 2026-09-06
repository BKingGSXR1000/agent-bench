export const TASK_STORAGE_KEY = "taskboard.tasks.v1";
export const FILTER_STORAGE_KEY = "taskboard.filters.v1";
export const STATUSES = Object.freeze(["Todo", "Doing", "Done"]);
export const PRIORITIES = Object.freeze(["Low", "Medium", "High"]);
export function validStatus(value) { return STATUSES.includes(value); }
export function validPriority(value) { return PRIORITIES.includes(value); }
