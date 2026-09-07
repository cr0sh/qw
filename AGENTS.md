- DO NOT introduce backwards compatibility or versioning in serialization,
  programs, or other formats. This is because this product is early-stage so not
  yet released. Proper versioning will be introduced after initial release.
- NEVER add external dependencies to project/subprojects other than explicitly
  specified ones to install.

- Every subagent worktree under `worktrees/` MUST link Cargo's `target/` to the primary repository's `target/` immediately after worktree creation and before any Cargo, build, or benchmark command. For the direct `worktrees/<task>` layout, create `target -> ../../target` and verify that it resolves to the primary root `target/`. NEVER delete or replace a pre-existing real directory or unrelated symlink; stop and preserve its contents instead (an existing link is usable only after verifying its resolution).
- Sharing `target/` is not a universal race-free guarantee: agents MUST serialize concurrent Cargo commands that can write shared artifacts with a separate build lock. This does not replace explicit `./gpu-lock -- ...` serialization for Metal commands. Benchmark and other result files MUST use unique names to avoid output collisions.
