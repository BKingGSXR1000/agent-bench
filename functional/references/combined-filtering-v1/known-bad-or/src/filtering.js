export function taskMatches(task, filters) {
  const query = filters.query.toLocaleLowerCase();
  const checks = [];
  if (query) checks.push(`${task.title} ${task.description}`.toLocaleLowerCase().includes(query));
  if (filters.status !== "all") checks.push(task.status === filters.status);
  if (filters.priority !== "all") checks.push(task.priority === filters.priority);
  return checks.length === 0 || checks.some(Boolean);
}
