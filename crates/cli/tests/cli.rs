use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use qw_runtime::{DEFAULT_MODEL_IDENTIFIER, model_cache_path};

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

fn run_stats(
    home: &std::path::Path,
    model_override: Option<&std::path::Path>,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qw"));
    command.arg("stats").env("HOME", home);
    match model_override {
        Some(path) => {
            command.env("QW_MODEL_PATH", path);
        }
        None => {
            command.env_remove("QW_MODEL_PATH");
        }
    }
    command.output().expect("run qw stats")
}

#[test]
fn nonexistent_local_model_exits_nonzero_with_its_path() {
    let missing = std::env::temp_dir().join(format!("qw-missing-model-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_qw"))
        .args([
            "generate",
            "--model",
            missing.to_str().expect("UTF-8 test path"),
            "--prompt",
            "hello",
        ])
        .output()
        .expect("run qw");

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains(&missing.display().to_string()), "{stderr}");
    assert!(stderr.contains("model directory"), "{stderr}");
}

#[test]
fn stats_reports_cache_category_usage_and_default_model_presence() {
    let home = temp_home("stats-default");
    let model_path =
        model_cache_path(&home, DEFAULT_MODEL_IDENTIFIER).expect("default model cache path");
    std::fs::create_dir_all(&model_path).expect("create cached model");
    std::fs::write(model_path.join("config.json"), vec![0_u8; 16_000])
        .expect("write cached model file");
    let checkpoint_path = home.join(".cache/qw/checkpoint/session");
    std::fs::create_dir_all(&checkpoint_path).expect("create checkpoint cache");
    std::fs::write(checkpoint_path.join("prefix.bin"), vec![0_u8; 3_500])
        .expect("write checkpoint cache file");
    std::fs::write(checkpoint_path.join("metadata.json"), vec![0_u8; 500])
        .expect("write checkpoint metadata");

    let output = run_stats(&home, None);

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
             Selected model: {DEFAULT_MODEL_IDENTIFIER}\n\
             Model path: {}\n\
             Model exists locally: yes\n",
            home.join(".cache/qw").display(),
            model_path.display()
        )
    );

    std::fs::remove_dir_all(home).expect("remove temporary HOME");
}

#[test]
fn stats_reports_missing_model_selected_by_environment() {
    let home = temp_home("stats-override");
    let missing_model = home.join("missing-model");

    let output = run_stats(&home, Some(&missing_model));

    assert!(output.status.success(), "{output:?}");
    assert!(output.stderr.is_empty(), "{output:?}");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 stdout");
    assert_eq!(
        stdout,
        format!(
            "Cache root: {}\n\
             Cache usage: 0 files, 0 bytes (0.00B)\n\
             Model cache usage: 0 files, 0 bytes (0.00B)\n\
             Checkpoint cache usage: 0 files, 0 bytes (0.00B)\n\
             Selected model: {}\n\
             Model path: {}\n\
             Model exists locally: no\n",
            home.join(".cache/qw").display(),
            missing_model.display(),
            missing_model.display()
        )
    );

    std::fs::remove_dir_all(home).expect("remove temporary HOME");
}
