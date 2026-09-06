# Multi-project migration v1 — Complex scenario design

This Complex scenario uses `taskboard-filtering-v1`, a third frozen lineage:
`taskboard-v1` → `taskboard-priority-v1` → `taskboard-filtering-v1`. The last
baseline freezes validated priority and combined filtering so projects,
migration, and import/export are the only requested feature family.

The evaluator-owned reference spans seven source modules: constants, project
state/schema migration, project scoping, import/export decoding, filter matching,
task-board state operations, and browser wiring. Correct solutions need not use
this structure; it documents the baseline and reference change surface only.

The schema contract is version 2 project state with projects, project-owned
tasks, active project, and filters. Legacy `taskboard.tasks.v1` plus filters are
migrated once into deterministic `project-inbox`. Import validates the full
candidate state before replacing live state; archived projects remain exported
with their tasks. No restore/unarchive behavior is in scope.
