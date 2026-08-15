use std::process::Command;

#[test]
fn nonexistent_local_model_exits_nonzero_with_its_path() {
    let missing = std::env::temp_dir().join(format!(
        "qwr-missing-model-{}",
        std::process::id()
    ));
    let output = Command::new(env!("CARGO_BIN_EXE_qwr"))
        .args([
            "generate",
            "--model",
            missing.to_str().expect("UTF-8 test path"),
            "--prompt",
            "hello",
        ])
        .output()
        .expect("run qwr");

    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).expect("UTF-8 stderr");
    assert!(stderr.contains(&missing.display().to_string()), "{stderr}");
    assert!(stderr.contains("model directory"), "{stderr}");
}
