// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Resolution and verification of the MLX upstream submodule commit.
//!
//! The pin is the checked-out commit of the `mlx` git submodule at the
//! repository root. Build scripts resolve it with `git rev-parse HEAD` instead
//! of restating a second SHA in CMake or Rust.
//!
//! The build script pulls this module in with `#[path]` and uses it to resolve
//! the submodule commit before the `_deps/` purge decision, the
//! `cargo:rustc-env=MLXCEL_MLX_COMMIT` export, the post-build verification and
//! the `_deps/.mlx-build-commit` marker write.
//!
//! The logic is deliberately dependency-free (`std` only): a build script that
//! needs a crates.io dependency to read its own submodule commit would be a
//! worse trade than the small git command below.

// `build.rs` includes this file as a private module, so not every helper is
// called in that compilation even though the tests cover all of them.
#![allow(dead_code)]

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Location of the MLX submodule, relative to `mlxcel-core`'s
/// `CARGO_MANIFEST_DIR`.
pub const MLX_SUBMODULE_RELATIVE_PATH: &str = "../../mlx";

/// Length of a full git object name in hex characters.
const FULL_SHA_LEN: usize = 40;

/// Why the MLX submodule commit could not be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinError {
    Unreadable { path: PathBuf, message: String },
    NotAFullCommitSha { path: PathBuf, value: String },
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreadable { path, message } => write!(
                f,
                "cannot read the MLX submodule commit from {} ({message})",
                path.display()
            ),
            Self::NotAFullCommitSha { path, value } => write!(
                f,
                "the MLX submodule commit in {} is {value:?}, which is not a \
                 {FULL_SHA_LEN}-character lowercase hex commit SHA",
                path.display()
            ),
        }
    }
}

impl std::error::Error for PinError {}

/// Whether a cached `_deps/` tree may be reused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheState {
    /// The marker records the pin currently in effect; the cache is reusable.
    Valid,
    /// The marker is absent or records a different commit; `_deps/` must go.
    Stale { cached: Option<String> },
}

/// Result of comparing an MLX checkout's HEAD against the expected commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeadCheck {
    /// HEAD is the expected commit.
    Match,
    /// HEAD is readable and is a different commit.
    Mismatch { found: String },
    /// HEAD could not be read.
    Unavailable { reason: String },
}

/// Read and validate the checked-out MLX submodule commit.
///
/// `manifest_dir` is `mlxcel-core`'s `CARGO_MANIFEST_DIR`.
pub fn read_pinned_commit(manifest_dir: &Path) -> Result<String, PinError> {
    let path = manifest_dir.join(MLX_SUBMODULE_RELATIVE_PATH);
    if !path.join(".git").exists() {
        return Err(PinError::Unreadable {
            path,
            message: "MLX submodule metadata is missing; run `git submodule update --init mlx`"
                .to_string(),
        });
    }

    let output = head_command(&path)
        .output()
        .map_err(|err| PinError::Unreadable {
            path: path.clone(),
            message: format!("could not run `git rev-parse HEAD`: {err}"),
        })?;
    if !output.status.success() {
        return Err(PinError::Unreadable {
            path,
            message: format!(
                "`git rev-parse HEAD` failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        });
    }

    let commit = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !is_full_commit_sha(&commit) {
        return Err(PinError::NotAFullCommitSha {
            path,
            value: commit,
        });
    }
    Ok(commit)
}

/// Decide whether a `_deps/` tree carrying `marker_contents` matches the pin.
///
/// `marker_contents` is `None` when `_deps/.mlx-build-commit` is missing or
/// unreadable, which is treated exactly like a wrong commit: nothing vouches
/// for the tree, so it does not get reused.
pub fn cache_state(marker_contents: Option<&str>, expected_commit: &str) -> CacheState {
    let cached = marker_contents.map(|raw| raw.trim().to_string());
    if cached.as_deref() == Some(expected_commit) {
        CacheState::Valid
    } else {
        CacheState::Stale { cached }
    }
}

/// Environment variables through which git's repository discovery can be
/// redirected at a different repository than the one named on the command line.
///
/// They are cleared for the HEAD probe. The probe exists to read the HEAD of
/// one specific directory, and any of these left in the ambient environment
/// silently makes `git -C <dir> rev-parse HEAD` answer for something else:
/// `GIT_DIR` alone is enough to turn a `Mismatch` into a `Match`, which would
/// let `mark_mlx_cache_valid` bless a `_deps/mlx-src` that is not the pinned
/// commit. This is not an exotic condition. Git exports `GIT_DIR` (and friends)
/// into the child environment of every hook, of `git rebase -x`, and of
/// `git bisect run`, so any `cargo build` reached through one of those inherits
/// it. The `.git` probe above defends against discovery walking *up*; this
/// defends against it being redirected *sideways*.
const GIT_DISCOVERY_ENV: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
];

/// Build the `git rev-parse HEAD` invocation the HEAD probe runs.
///
/// `mlx_src_dir` is passed as the argument of `-C`, which git consumes
/// unconditionally, so a directory whose name begins with a dash is read as a
/// path and never as an option. Nothing else on the command line is
/// path-derived.
fn head_command(mlx_src_dir: &Path) -> Command {
    let mut command = Command::new("git");
    for name in GIT_DISCOVERY_ENV {
        command.env_remove(name);
    }
    command
        .arg("-C")
        .arg(mlx_src_dir)
        .args(["rev-parse", "HEAD"]);
    command
}

/// Compare a git checkout's HEAD against the expected MLX submodule commit.
pub fn check_head(mlx_src_dir: &Path, expected_commit: &str) -> HeadCheck {
    if !mlx_src_dir.exists() {
        return HeadCheck::Unavailable {
            reason: format!("{} does not exist", mlx_src_dir.display()),
        };
    }

    // The `.git` probe is load-bearing, not a fast path. `git -C <dir>` walks
    // up to the nearest enclosing repository, and `_deps/` sits inside the
    // mlxcel checkout, so without this a source tree with no git metadata
    // would answer with *mlxcel's* HEAD and be reported as a pin mismatch.
    if !mlx_src_dir.join(".git").exists() {
        return HeadCheck::Unavailable {
            reason: format!(
                "{} has no .git metadata (vendored or exported source tree)",
                mlx_src_dir.display()
            ),
        };
    }

    let output = match head_command(mlx_src_dir).output() {
        Ok(output) => output,
        Err(err) => {
            return HeadCheck::Unavailable {
                reason: format!("could not run `git`: {err}"),
            };
        }
    };

    if !output.status.success() {
        return HeadCheck::Unavailable {
            reason: format!(
                "`git rev-parse HEAD` failed in {}: {}",
                mlx_src_dir.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        };
    }

    let found = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if found == expected_commit {
        HeadCheck::Match
    } else {
        HeadCheck::Mismatch { found }
    }
}

/// A full git object name: exactly 40 lowercase hex characters.
///
/// Uppercase is rejected on purpose. `git rev-parse HEAD` prints lowercase and
/// the `_deps/.mlx-build-commit` marker is compared byte for byte, so an
/// uppercase pin would make every cache look stale forever.
fn is_full_commit_sha(value: &str) -> bool {
    value.len() == FULL_SHA_LEN
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const PIN: &str = "2c46b953db88965c4270cc7306eda6887a3247f2";
    const OTHER_PIN: &str = "b7c3dd6d27f45b5365b08a840310187dc503f1db";

    fn git_available() -> bool {
        Command::new("git")
            .arg("--version")
            .output()
            .is_ok_and(|out| out.status.success())
    }

    fn init_repo_with_one_commit(dir: &Path) -> String {
        let run = |args: &[&str]| {
            let out = Command::new("git")
                .arg("-C")
                .arg(dir)
                .args(args)
                .output()
                .expect("git should run");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        run(&["init", "-q"]);
        run(&[
            "-c",
            "user.name=mlxcel test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "pin fixture",
        ]);
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git should run");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    #[test]
    fn the_real_mlx_submodule_resolves_to_a_full_sha() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let commit = read_pinned_commit(manifest_dir).expect("in-tree MLX submodule must resolve");
        assert!(
            is_full_commit_sha(&commit),
            "resolved commit {commit:?} is not a full lowercase hex SHA"
        );
    }

    #[test]
    fn a_missing_submodule_is_an_error() {
        let dir = TempDir::new().unwrap();
        let err = read_pinned_commit(dir.path()).unwrap_err();
        assert!(matches!(err, PinError::Unreadable { .. }));
        assert!(err.to_string().contains("mlx"), "{err}");
    }

    #[test]
    fn marker_whitespace_is_ignored() {
        assert_eq!(
            cache_state(Some(&format!("{PIN}\n")), PIN),
            CacheState::Valid
        );
    }

    #[test]
    fn a_marker_from_another_pin_makes_the_cache_stale() {
        assert_eq!(
            cache_state(Some(OTHER_PIN), PIN),
            CacheState::Stale {
                cached: Some(OTHER_PIN.to_string()),
            }
        );
    }

    #[test]
    fn an_absent_marker_makes_the_cache_stale() {
        assert_eq!(cache_state(None, PIN), CacheState::Stale { cached: None });
    }

    #[test]
    fn head_matching_the_pin_passes() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let dir = TempDir::new().unwrap();
        let head = init_repo_with_one_commit(dir.path());
        assert_eq!(check_head(dir.path(), &head), HeadCheck::Match);
    }

    #[test]
    fn head_differing_from_the_pin_is_a_mismatch() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let dir = TempDir::new().unwrap();
        let head = init_repo_with_one_commit(dir.path());
        assert_eq!(
            check_head(dir.path(), PIN),
            HeadCheck::Mismatch { found: head }
        );
    }

    #[test]
    fn a_tree_without_git_metadata_is_unavailable_not_a_mismatch() {
        let dir = TempDir::new().unwrap();
        assert!(matches!(
            check_head(dir.path(), PIN),
            HeadCheck::Unavailable { .. }
        ));
    }

    #[test]
    fn a_missing_directory_is_unavailable() {
        let dir = TempDir::new().unwrap();
        assert!(matches!(
            check_head(&dir.path().join("mlx-src"), PIN),
            HeadCheck::Unavailable { .. }
        ));
    }

    #[test]
    fn the_check_does_not_walk_up_into_an_enclosing_repository() {
        if !git_available() {
            eprintln!("skipping: git is not on PATH");
            return;
        }
        let outer = TempDir::new().unwrap();
        let outer_head = init_repo_with_one_commit(outer.path());
        let inner = outer.path().join("build/_deps/mlx-src");
        std::fs::create_dir_all(&inner).unwrap();

        match check_head(&inner, PIN) {
            HeadCheck::Unavailable { .. } => {}
            other => panic!("expected Unavailable, got {other:?} (outer HEAD is {outer_head})"),
        }
    }

    #[test]
    fn the_head_probe_clears_git_repository_redirection_variables() {
        let command = head_command(Path::new("/out/build/_deps/mlx-src"));
        let cleared: Vec<String> = command
            .get_envs()
            .filter(|(_, value)| value.is_none())
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect();
        for name in GIT_DISCOVERY_ENV {
            assert!(
                cleared.iter().any(|entry| entry == name),
                "{name} must be cleared, cleared: {cleared:?}"
            );
        }
    }
}
