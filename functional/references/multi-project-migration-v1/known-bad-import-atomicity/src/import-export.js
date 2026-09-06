import{normalizeState}from"./project-state.js";
export const exportState=state=>JSON.stringify(state);
export function decodeImport(text,current){current.tasks=[];let parsed;try{parsed=JSON.parse(text);}catch{throw new Error("Malformed import JSON.");}return normalizeState(parsed);}
