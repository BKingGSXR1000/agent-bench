import{FILTER_STORAGE_KEY,validPriority,validStatus}from"./constants.js";
export const EMPTY_FILTERS=Object.freeze({query:"",status:"all",priority:"all"});
export const normalizeFilters=value=>({query:typeof value?.query==="string"?value.query:"",status:validStatus(value?.status)?value.status:"all",priority:validPriority(value?.priority)?value.priority:"all"});
export function loadFilters(storage){try{return normalizeFilters(JSON.parse(storage.getItem(FILTER_STORAGE_KEY)||"{}"));}catch{return{...EMPTY_FILTERS};}}
export const saveFilters=(storage,filters)=>storage.setItem(FILTER_STORAGE_KEY,JSON.stringify(filters));
