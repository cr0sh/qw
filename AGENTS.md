- DO NOT introduce backwards compatibility or versioning in serialization,
  programs, or other formats. This is because this product is early-stage so not
  yet released. Proper versioning will be introduced after initial release.
- NEVER add external dependencies to project/subprojects other than explicitly
  specified ones to install.
- Actively commit your works. Commits must include concise conventional commit
  titles.
  - Rule of thumb: Commits should be split enough, by features or specific
    changes. So the work history should be bisectable/diffable when a bug
    or breakage occurrs in the future.
  - Nontrivially complex commits may include descriptions along with the commit title.
  - Always append `Assisted-by: <agent-name>/<model-id>` to the commit
    description (for example, `Assisted-by: claude-code/sonnet`) to preserve LLM
    provenance.

- All agents(including the principal root agent) should not work on main to
  prevent collision. Delegate to subagents unless it's trivial fix unlikely to collide.
- Subagents MUST create a dedicated git worktree for each task under `worktrees/`, using a new branch based on `main`. On the worktree, symlink `mlx` to the repo root's `mlx` submodule(so `../../mlx`)
- Subagents MUST perform all task work in that worktree and MUST NOT modify the main worktree directly.
- After completing and committing the task, subagents MUST return to the main worktree and merge the worktree branch into `main` as their final action.

