export const TASK_STORAGE_KEY="taskboard.tasks.v1",FILTER_STORAGE_KEY="taskboard.filters.v1",PROJECT_STATE_KEY="taskboard.projects.v2";
export const STATUSES=Object.freeze(["Todo","Doing","Done"]),PRIORITIES=Object.freeze(["Low","Medium","High"]);
export const validStatus=value=>STATUSES.includes(value),validPriority=value=>PRIORITIES.includes(value);
