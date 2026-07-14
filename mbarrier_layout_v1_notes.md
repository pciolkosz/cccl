# mbarrier layout v1 barrier design

This note sketches an underscored layout-v1 barrier prototype:
`cuda::__barrier_with_status<cuda::thread_scope_block>`.

The goal is to expose mbarrier layout v1 status reporting without changing the existing
`cuda::barrier` layout-v0 behavior. The design should also avoid choices that would make a
future global-memory barrier unit (GBU) abstraction hard to add later, but this is not a GBU
design.

## Goals

- Keep the API close to `cuda::barrier<cuda::thread_scope_block>`.
- Add status reporting for layout-v1 waits.
- Do not provide a fake status fallback for host or non-shared-memory storage.
- Keep operation status, tokens, count policy, and storage lifetime separable so future GBU
  support is not blocked.
- Keep the interface underscored until the public API is settled.

## Changes from `cuda::barrier`

### Storage and lifetime

Current `cuda::barrier<thread_scope_block>` uses mbarrier layout v0 when the object is in local
shared memory, and falls back to `cuda::std::__barrier_base` otherwise. That fallback is appropriate
for `cuda::barrier` because it preserves the same logical barrier interface.

The layout-v1 prototype intentionally does not keep that split:

- Shared-memory block-scope barriers use `mbarrier.init.layout::v1.shared::cta.b64`.
- Host and non-shared-memory paths are unsupported because they cannot produce layout-v1 status
  reports.
- Cluster-shared storage is rejected by assertions.
- Destruction still invalidates shared-memory mbarriers.

If a caller does not use status-bearing operations, `cuda::barrier` remains the right interface.
Synthesizing successful no-report statuses from a fallback barrier would make
`__barrier_with_status` look valid in cases where the core feature is unavailable.

GBU extensibility: global barriers have very different storage and lifetime rules: 256B-aligned
global-memory storage, device-locality requirements, and explicit init/inval operations. The
layout-v1 constructor/destructor model should not be treated as the generic model for all future
barriers.

### Counts

Layout v0 supports expected/update counts up to `(1 << 20) - 1`. Layout v1 supports only
`(1 << 9) - 1`, so `max()`, construction checks, and `arrive(update)` checks use that smaller
limit.

GBU extensibility: GBU uses a 48-bit unified arrival + transaction count with a configurable bit
split. A future GBU design likely needs explicit count helpers or policies rather than a single
`max()` value.

### Tokens and phase

The prototype keeps an `arrival_token` shape compatible with the mbarrier state used by the
layout-v1 wait instructions.

GBU extensibility: GBU waits use a phase operand and have stricter phase-progression rules. Token
types should stay opaque and barrier-specific so a future GBU token can carry the right phase
information.

### Wait APIs

Layout-v0 waits only report completion. Layout-v1 waits can also report fabric operation status, so
status-bearing APIs return `operation_status`:

| Current API | Layout-v1 prototype |
| --- | --- |
| `void wait(token)` | `[[nodiscard]] operation_status wait(token)` |
| `void arrive_and_wait()` | `[[nodiscard]] operation_status arrive_and_wait()` |
| `bool try_wait_for(token, duration)` | `[[nodiscard]] operation_status try_wait_for(token, duration)` |
| `bool try_wait_until(token, time_point)` | `[[nodiscard]] operation_status try_wait_until(token, time_point)` |

The prototype also adds:

- `[[nodiscard]] operation_status try_wait(token)` for non-timed status-bearing polling.
- `[[nodiscard]] bool try_wait_no_status(token)` for callers that only need the completion
  predicate and intentionally do not want a status object.

`wait(token)` loops until `complete() == true`, so it returns a completed status that may still
carry an error report.

Token passing needs an explicit ownership rule because status-bearing waits can complete with an
error report. Blocking `wait(token)` can consume the token because it returns only after completion.
Non-blocking and timed waits may return an incomplete `operation_status`, so taking the token by
value keeps retry semantics simple. This differs from the current timed `cuda::barrier` API, which
takes `arrival_token&&` and ties token consumption to a `bool` success result.

The prototype also includes parity waits with the same result model:

- `wait_parity(bool) -> operation_status`
- `try_wait_parity_for(...) -> operation_status`
- `try_wait_parity_until(...) -> operation_status`

### `arrive_and_drop`

`arrive_and_drop()` has no wait status to return, so it can remain `void` unless future PTX/runtime
interfaces add reportable failure information. It still needs layout-v1 instruction coverage and
layout-v1 count checks.

## `operation_status`

`operation_status` is the unified result object for status-bearing waits. It stores:

- completion predicate
- report predicate
- report value
- an inspection flag

Public queries:

| API | Meaning |
| --- | --- |
| `complete()` | The wait operation completed. |
| `operator bool()` | Completed and no error report was present. |
| `has_error_report()` | An error report was present; also marks the report inspected. |
| `get_error_count()` | Decodes and returns the error count; also marks inspected. |
| `for_each_error(fn)` | Iterates decoded `cudaFabricOpStatusInfo` entries; also marks inspected. |

If a status object contains an error report and no inspection API is called, destroying or
overwriting it traps on device. This prevents silent loss of fabric errors.

GBU extensibility: GBU waits also produce completion/report predicates, but report lookup may be a
separate phase-based query. `operation_status` should remain a logical result API, not expose
layout-v1-specific payload storage.

## GBU constraints to preserve

This proposal should leave room for a later GBU design. Important constraints from the GBU
programming model:

- GBU storage is global memory, not shared memory, and has explicit locality and lifetime rules.
- GBU counts are unified arrival/transaction counts, not simple arrival counts.
- GBU phase ordering is stricter than the current `cuda::barrier` documentation.
- GBU has multiple wait-like operations with different progress guarantees, including local
  try-wait, host try-wait, remote try-wait, and possibly peek.
- GBU subscribe/unsubscribe and fence interactions are memory-model issues and should not be hidden
  inside `operation_status`.

## Open questions

- Final barrier type name: `barrier_with_status` is tentative.
- Final result name: `operation_status` is the current preference.
- Finalize token ownership rules for non-blocking and timed status-bearing waits.
- Which helpers should become common building blocks for future GBU support.
