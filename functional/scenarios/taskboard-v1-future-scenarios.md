# Taskboard v1 future functional scenarios

These are evaluator-owned scenario designs, not tasks delivered to an agent yet.

## Medium — search-status-priority-filter-v1

Add case-insensitive task search plus combined status and priority filtering. The
complete filter state persists across reload without mutating tasks. Acceptance
categories: baseline CRUD/persistence regression, search matching and empty
results, combined-filter intersection, filter-state reload, and filter reset.
Hard gates: existing task persistence, combined intersection correctness, and
filter state never changes stored task records.

## Complex — projects-migration-archive-import-export-v1

Add multiple projects, migrate existing unprojected data safely, isolate tasks
between projects, archive projects, and support a full JSON export/import
round-trip. Acceptance categories: baseline regression, migration, project
isolation, archived-project behavior, invalid import safety, and exact
round-trip preservation. Hard gates: no loss of legacy tasks, project isolation,
and round-trip data equivalence.
