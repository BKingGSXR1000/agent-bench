# Combined filtering v1 — Medium scenario design

This scenario uses baseline strategy **B**: `taskboard-priority-v1`, a separate
frozen evaluator baseline derived from the validated Easy priority reference.
Priority is therefore established functionality and a regression dimension, not
part of the work requested from an agent. This prevents a Medium run from
measuring two feature implementations at once.

The requested behavior adds title/description search, status and priority
filters, immediate AND-combined visibility, persisted filter state, clear-all,
and correct behavior when editing or deleting under active filters. The hidden
acceptance suite checks baseline regression, search, individual/combined filters,
filter persistence, active-filter interactions, and recoverable zero results.

Self-validation fixtures are evaluator-owned: the untouched derived baseline,
a complete filtering reference, OR-semantics filtering, non-persistent filter
state, and a delete regression. Exact vectors and hard gates live in the YAML
scenario definition; no acceptance material is copied into the subject.
