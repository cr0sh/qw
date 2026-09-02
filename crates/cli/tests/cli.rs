use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use qw_runtime::{PINNED_REPOSITORY, PINNED_REVISION, pinned_model_dir};

static NEXT_TEMP_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

fn temp_home(label: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "qw-cli-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create temporary HOME");
    path
}

fn run_stats(home: &std::path::Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_qw"))
        .arg("stats")
        .env("HOME", home)
        .env_remove("QW_MODEL_PATH")
        .output()
        .expect("run qw stats")
}

#[test]
fn generate_rejects_model_path_overrides_before_loading() {
    let output = Command::new(env!("CARGO_BIN_EXE_qw"))
        .args(["generate", "--model", "/tmp/model", "--prompt", "hello"])
        .output()
        .expect("run qw");

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains("unexpected argument '--model'"), "{stderr}");
}

#[test]
fn stats_reports_fixed_pinned_model_presence() {
    let home = temp_home("stats");
    let model_path = pinned_model_dir(&home);
    std::fs::create_dir_all(&model_path).expect("create cached model");
    std::fs::write(model_path.join("fixture.bin"), vec![0_u8; 16_000])
        .expect("write cached model file");
    let checkpoint_path = home.join(".cache/qw/checkpoint/session");
    std::fs::create_dir_all(&checkpoint_path).expect("create checkpoint cache");
    std::fs::write(checkpoint_path.join("prefix.bin"), vec![0_u8; 3_500])
        .expect("write checkpoint cache file");
    std::fs::write(checkpoint_path.join("metadata.json"), vec![0_u8; 500])
        .expect("write checkpoint metadata");

    let output = run_stats(&home);

    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    assert_eq!(
        stdout,
        format!(
            "Cache root: {}\n\
             Cache usage: 3 files, 20000 bytes (20.00K)\n\
             Model cache usage: 1 files, 16000 bytes (16.00K)\n\
             Checkpoint cache usage: 2 files, 4000 bytes (4.00K)\n\
             Pinned model: {PINNED_REPOSITORY}@{PINNED_REVISION}\n\
             Model path: {}\n\
             Model exists locally: yes\n",
            home.join(".cache/qw").display(),
            model_path.display()
        )
    );

    std::fs::remove_dir_all(home).expect("remove temporary HOME");
}

#[test]
fn stats_ignores_legacy_model_override_environment() {
    let home = temp_home("stats-env");
    let output = Command::new(env!("CARGO_BIN_EXE_qw"))
        .arg("stats")
        .env("HOME", &home)
        .env("QW_MODEL_PATH", home.join("ignored"))
        .output()
        .expect("run qw stats");

    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    assert!(
        stdout.contains(&format!(
            "Model path: {}",
            pinned_model_dir(&home).display()
        )),
        "{stdout}"
    );
    assert!(!stdout.contains("ignored"), "{stdout}");
    std::fs::remove_dir_all(home).expect("remove temporary HOME");
}
