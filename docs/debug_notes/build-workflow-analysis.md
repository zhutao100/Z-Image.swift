# Build Workflow Analysis

## Scope

This note validates the repo's current build and test shape, records the concrete MLX-related failure mode that existed before the fix, and documents the workflow changes that were applied.

## Validated Repo Facts

- The repo is package-first. `Package.swift` is the source of truth and there is no checked-in Xcode project or workspace driving development.
- Release packaging is already Xcode-based. `.github/workflows/ci.yml` builds `ZImageCLI` with `xcodebuild`, then copies `default.metallib` out of `mlx-swift_Cmlx.bundle`.
- SwiftPM-based MLX execution still depends on the repo-local `scripts/build_mlx_metallib.sh` workaround because the SwiftPM path does not emit the default MLX metallib on its own.
- Before this change, some integration and E2E helpers knew how to colocate `mlx.metallib`, but the default `ZImageTests` target was not consistently doing that.

## Measured Failure Before Changes

On March 10, 2026, a clean `swift test` failed immediately in the default unit-test workflow:

```text
MLX error: Failed to load the default metallib.
```

The failure surfaced in `AIOCheckpointTests` before the rest of the suite could run. That invalidated the previous documentation claim that plain `swift test` was the default working verification path for this repo.

## Decision

The right split for this repo remains:

- Keep SwiftPM as the source of truth for the package graph, unit tests, and day-to-day development.
- Keep Xcode/xcodebuild as the release-artifact builder.
- Hide the SwiftPM metallib workaround behind stable repo workflows instead of requiring contributors to remember manual preparation steps.
- Add a release-path smoke check without trying to move the entire repo back to Xcode-first testing.

## Implemented Changes

- Hardened `Tests/ZImageTests` so MLX-backed unit tests now prepare and colocate the SwiftPM metallib automatically.
- Hardened the integration and E2E test helpers so they prepare `mlx.metallib` automatically when those suites are explicitly enabled.
- Hardened the E2E helper so it builds the SwiftPM `ZImageCLI` product automatically when an opt-in CLI test is requested.
- Updated `scripts/build.sh` to use the same non-interactive plugin flags as CI and to accept environment overrides such as `DERIVED_DATA_PATH` and `CONFIGURATION`.
- Evaluated and removed a temporary `scripts/test.sh` wrapper because it added no behavior beyond `swift test` when invoked from the repo root.
- Updated CI to:
  - run the SwiftPM verification job on pull requests
  - gate the nightly release job on that verification
  - keep Xcode as the release builder
  - smoke-run `ZImageCLI --help` from the packaged release directory before zipping the artifact

## Validation After Changes

Commands run and outcomes:

- `swift test`
  - passed
  - `210` tests executed, `31` skipped, `0` failures
- `ZIMAGE_RUN_E2E_TESTS=1 swift test --filter CLIEndToEndTests/testHelpCommand`
  - passed
  - verified the opt-in E2E path can build `ZImageCLI` and run without manual prep
- `DERIVED_DATA_PATH=./dist ./scripts/build.sh`
  - passed
- Release-directory smoke and package check
  - copied `default.metallib` next to `ZImageCLI`
  - `./ZImageCLI --help` succeeded
  - packaged zip contained exactly `ZImageCLI` and `default.metallib`

## Remaining Tradeoffs

- SwiftPM MLX execution still depends on the repo-local metallib generation script. That is acceptable for dev/test support, but it is still less elegant than the Xcode bundle path used for release artifacts.
- The release smoke check validates packaging and process startup, not full inference. That is intentional because full runtime coverage is already available through opt-in integration and E2E workflows that are much more expensive.
- The Xcode release build still emits pre-existing compiler warnings unrelated to this workflow change. They were observed during validation but not addressed here.
