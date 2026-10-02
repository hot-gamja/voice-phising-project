# Freeze provenance correction (recorded 2026-10-02)

`experiments/freeze.json:3` records `"git_commit": "d9b81b4"`, which is **not an
object in this repository**. The same value was copied into
`experiments/p09_test_results.md:1` and `README.md`.

`freeze.json` and `p09_test_results.md` are one-shot sealed records (`freeze.json`
rule: *"Nothing above changes after this point"*), so neither is edited. This file
is the correction, appended rather than applied in place.

## Verified, with the command used

| claim | command | result |
|---|---|---|
| `d9b81b4` is not an object | `git cat-file -t d9b81b4` | `fatal: Not a valid object name` |
| no loose object with that prefix | `ls .git/objects/d9/` | directory does not exist |
| nothing recoverable from lost objects | `git fsck` | 0 dangling objects |
| no rebase/amend rewrote history | author vs committer date, all 21 commits | identical on every commit |
| reflog cannot reach freeze time | `git reflog --all \| wc -l` | 9 entries, oldest 2026-09-29 17:23:58 +0900 |
| frozen code files are content-verifiable | recompute sha256 of each `code:` blob per commit | 5/5 match on every commit from `a3931f5` through HEAD |
| frozen code last changed at | `git log -1 -- <each code file>` | `threshold.py` → `a3931f5`; the other four are older |
| freeze instant matches a commit | `frozen_at` 2026-09-28T01:45:37+00:00 = 2026-09-28 10:45:37 +0900 | equals `ad1b77b` author and committer time to the second |
| artifacts predate the freeze | `ls -lT models/` | 10:24:18 – 10:37:05 +0900, all before 10:45:37 |

## Correct identifiers

- `scripts/freeze.py` ran at **`ad1b77b`** (2026-09-28T10:45:37+09:00) — the only
  commit whose timestamp equals `frozen_at` exactly.
- `freeze.json` and the Test results were committed in **`fe25496`**
  (2026-09-28T11:26:36+09:00).
- The frozen code state first existed at **`a3931f5`**
  (2026-09-28T10:35:51+09:00) and is byte-identical through HEAD.

Read `d9b81b4` as `ad1b77b` wherever it appears.

## Not determined

Why `d9b81b4` was written instead. `scripts/freeze.py:48` takes the value from
`git rev-parse --short HEAD`, which cannot emit a non-existent object, so the
recorded value has no reconstructible origin. The reflog does not reach back far
enough and `git fsck` finds nothing, so this is left unresolved rather than
guessed at.

## Does this block V2? No, but every V2 report must cite this file

The freeze stands: what broke is traceability, not integrity. Integrity is proven
by content — `scripts/verify_freeze.py` recomputes all 9 recorded hashes and they
match (9/9, 2026-10-02), so the artifacts V2 evaluates are the ones that produced
the Test numbers. The broken commit id only costs the ability to name that moment
in the code history, and that loss is itself repaired above by content hash: the
frozen code is byte-identical from `a3931f5` to HEAD.

Re-freezing V1 would be worse than the defect. Test is opened once per model
(plan §12) and it has already been opened, so a second freeze cannot restore
blindness — it would only overwrite a verifiable record with a newer one.

Condition: V2 results that rest on the frozen V1 cite this file alongside
`freeze.json`, because `freeze.json` alone contains a field that does not resolve.

Out of scope here: `freeze.json` records no hash of `data/processed/*.jsonl`, so
the training input is not cryptographically pinned. Separate defect, not fixed by
this correction.

