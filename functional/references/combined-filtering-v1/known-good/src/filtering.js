export function taskMatches(task, filters) {
  const query = filters.query.toLocaleLowerCase();
  const text = `${task.title} ${task.description}`.toLocaleLowerCase();
  return (!query || text.includes(query)) && (filters.status === "all" || task.status === filters.status) && (filters.priority === "all" || task.priority === filters.priority);
}
