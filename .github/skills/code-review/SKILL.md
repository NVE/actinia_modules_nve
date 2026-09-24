# GRASS / Actinia Module Review Instructions

You are reviewing code for NVE GRASS and Actinia modules.

The repository contains production modules used in automated geospatial processing
workflows. Reviews must prioritize correctness, maintainability, performance,
and consistency with GRASS and GRASS Addons practices.

## General Review Principles

Focus on:

- correctness
- robustness
- maintainability
- backward compatibility
- performance
- user-facing behaviour

Avoid suggesting purely stylistic changes unless they improve readability,
maintainability, correctness, or align with established GRASS conventions.

Only report issues that are actionable and sufficiently important.

---

## GRASS Conventions

When reviewing Python code:

### Module interface

Verify that:

- options and flags follow GRASS conventions
- option names are descriptive and consistent with existing modules
- required inputs are correctly declared
- defaults are reasonable
- descriptions are clear and user-oriented

Prefer consistency with existing modules from GRASS core and GRASS Addons.

### GRASS Python API

Prefer:

- grass.tools
- grass.script as gs
- grass.exceptions for GRASS-specific exceptions
- gs.run_command()
- gs.read_command()
- gs.parse_command()

Avoid:

- unnecessary subprocess calls to GRASS modules
- manual parsing of command output when a parser exists (use Tools() result)
- shell=True

### Temporary Data

Verify that:

- temporary maps use append_node_pid() or equivalent unique naming
- temporary maps are removed
- cleanup() handlers exist
- cleanup is registered early

Flag:

- leaked temporary raster maps
- leaked vector maps
- leaked files
- leaked database connections

### Computational Regions

Check whether modules:

- unexpectedly modify the current region
- fail to restore regions when needed
- rely on the current region without documenting it

Recommend use of:

- use_temp_region()
- del_temp_region()

where appropriate.

---

## Performance

Raster and vector datasets may be national-scale.

Flag:

- unnecessary loops over raster values
- repeated r.info calls on the same map
- repeated g.list calls
- repeated database queries
- avoidable temporary maps

Prefer:

- GRASS modules over Python loops
- vectorized NumPy operations
- single-pass processing
- cached metadata

Pay particular attention to:

- O(n²) algorithms
- repeated filesystem scans
- repeated calls to databases

---

## Parallel Processing

Check multiprocessing carefully.

Verify:

- worker count respects available CPUs
- race conditions cannot occur
- temporary names are unique per process
- mapset conflicts are avoided

Flag:

- shared temp map names
- concurrent writes to identical outputs
- unsafe global state

---

## Error Handling

Prefer:

- gs.fatal()
- gs.warning()
- gs.message()

over generic print statements.

Verify that:

- recoverable situations emit warnings
- fatal situations stop execution
- exceptions include useful context

Flag:

- bare except clauses
- swallowed exceptions
- silent failures

---

## Compatibility

The code should remain compatible with current GRASS development practices.

Check:

- Python compatibility
- deprecation risks
- GRASS API changes

Avoid recommending patterns known to be deprecated in GRASS.

---

## Documentation

Verify that:

### Module documentation

- purpose is clear
- parameters are documented
- examples are realistic

### Code documentation

- non-obvious logic is explained
- public functions have docstrings
- comments explain intent rather than implementation

Flag:

- misleading comments
- outdated documentation

---

## Testing

When tests are modified:

Verify:

- behaviour is tested rather than implementation details
- edge cases are covered
- error paths are tested

Prefer:

- pytest
- existing GRASS testing patterns

Flag:

- tests likely to be flaky
- reliance on execution order
- platform-specific assumptions

---

## Actinia-Specific Concerns

These modules may run in distributed Actinia environments.

Pay attention to:

- filesystem assumptions
- hardcoded paths
- local temporary files
- external dependencies
- container compatibility

Flag:

- assumptions about local state
- non-portable filesystem usage
- undocumented system dependencies

---

## Review Prioritization

Report findings only when they are:

1. Correctness issues
2. Potential data loss issues
3. Performance problems on large datasets
4. Concurrency issues
5. Maintainability concerns
6. Documentation omissions affecting users

Do not report cosmetic issues unless they significantly reduce readability or
are inconsistent with established GRASS GIS conventions.
