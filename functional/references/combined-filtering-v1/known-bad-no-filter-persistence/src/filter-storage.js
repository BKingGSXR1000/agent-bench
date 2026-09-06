import { FILTER_STORAGE_KEY, validPriority, validStatus } from "./constants.js";
export const EMPTY_FILTERS = Object.freeze({ query: "", status: "all", priority: "all" });
export function normalizeFilters(value) { return { query: typeof value?.query === "string" ? value.query : "", status: validStatus(value?.status) ? value.status : "all", priority: validPriority(value?.priority) ? value.priority : "all" }; }
export function loadFilters(storage) { try { return normalizeFilters(JSON.parse(storage.getItem(FILTER_STORAGE_KEY) || "{}")); } catch { return { ...EMPTY_FILTERS }; } }
export function saveFilters(storage, filters) { void storage; void filters; }
