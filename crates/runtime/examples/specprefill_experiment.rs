use anyhow::{Context, Result, ensure};
use qw_runtime::{
    ChatMessage, ChatMessageContent, KVCacheMode, PrefillMode, Qwen35Provider, SpecPrefillConfig,
    resolve_model_path,
};

const AUTHORITATIVE_CODE: &str = "AUTH-7XK9-PROD-4421";
const DECOY_CODES: [&str; 3] = [
    "AUTH-1AAA-TEST-1000",
    "AUTH-4MNO-STAGE-2330",
    "AUTH-9ZZZ-OLD-9009",
];
const INITIAL_INSTRUCTION: &str = "You are reviewing an operational authorization ledger. Only the record marked AUTHORITATIVE CHANGE CONTROL is binding. Ignore superseded, staging, test, and historical records. At the end, answer with only the exact authorization code from the authoritative record.\n\n";

fn build_prompt(record_count: usize) -> String {
    let mut prompt = String::from(INITIAL_INSTRUCTION);
    for index in 0..record_count {
        if index == record_count / 2 {
            prompt.push_str(&format!(
                "RECORD CC-88421 | AUTHORITATIVE CHANGE CONTROL | region=prod-us-west | service=payments-ledger | approved_by=incident-commander | authorization_code={AUTHORITATIVE_CODE} | status=ACTIVE AND BINDING | note=Use this exact code.\n"
            ));
        }
        let (environment, status, code) = match index % 3 {
            0 => ("test", "NON-PRODUCTION", DECOY_CODES[0]),
            1 => ("staging", "SUPERSEDED", DECOY_CODES[1]),
            _ => ("prod-archive", "EXPIRED HISTORICAL", DECOY_CODES[2]),
        };
        prompt.push_str(&format!(
            "RECORD OPS-{index:06} | timestamp=2026-08-{:02}T{:02}:{:02}:00Z | environment={environment} | subsystem={} | authorization_code={code} | status={status} | operator=rotation-{:03} | evidence=health checks stable, rollback rehearsed, approval window closed.\n",
            1 + index % 20,
            index % 24,
            index % 60,
            index % 17,
            index % 113,
        ));
    }
    prompt.push_str(
        "\nFinal instruction: Return only the exact authorization code from AUTHORITATIVE CHANGE CONTROL. Do not return any test, staging, superseded, expired, or historical code.",
    );
    prompt
}

fn main() -> Result<()> {
    let target_path = resolve_model_path(None).context("resolve QW_MODEL_PATH/default target")?;
    let mut provider = Qwen35Provider::load(&target_path, KVCacheMode::Fp16)
        .context("load target and default/overridden SpecPrefill draft")?;

    let mut record_count = 128;
    let (prompt, rendered_prompt, prompt_ids) = loop {
        let prompt = build_prompt(record_count);
        let message = ChatMessage {
            role: "user".to_string(),
            name: None,
            content: Some(ChatMessageContent::Text(prompt.clone())),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        };
        let rendered_prompt = provider.render_messages(&[message], &[], None, false)?;
        let encoding = provider
            .tokenizer()
            .encode(rendered_prompt.as_str(), true)
            .map_err(anyhow::Error::msg)?;
        if encoding.len() > 6_000 {
            let ids = encoding
                .get_ids()
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<i32>>();
            break (prompt, rendered_prompt, ids);
        }
        record_count += 64;
    };
    let instruction_start = rendered_prompt
        .find(INITIAL_INSTRUCTION)
        .context("rendered chat prompt omitted the user instruction")?;
    let instruction_end = instruction_start + INITIAL_INSTRUCTION.len();
    let protected_prefix_tokens = provider
        .tokenizer()
        .encode(&rendered_prompt[..instruction_end], true)
        .map_err(anyhow::Error::msg)?
        .len();
    ensure!(
        prompt.contains(AUTHORITATIVE_CODE),
        "authoritative record missing"
    );

    let sampling = provider.baseline_sampling(
        true,
        qw_runtime::SamplingOptions {
            temperature: Some(0.0),
            top_p: Some(1.0),
            seed: Some(0),
            ..Default::default()
        },
    );
    let dense = provider.generate_baseline_streaming(
        &prompt_ids,
        32,
        &sampling,
        None,
        None,
        &[],
        PrefillMode::Dense,
        |_| true,
    )?;
    let sparse = provider.generate_baseline_streaming(
        &prompt_ids,
        32,
        &sampling,
        None,
        None,
        &[],
        PrefillMode::SpecPrefill(SpecPrefillConfig {
            min_tokens: 5_000,
            keep_rate: 0.30,
            protected_prefix_tokens,
            ..Default::default()
        }),
        |_| true,
    )?;
    let stats = sparse
        .specprefill_stats
        .as_ref()
        .context("SpecPrefill admission did not activate")?;
    let keep_ratio = stats.selected_target_tokens as f64 / stats.eligible_target_tokens as f64;

    println!("prompt_tokens={}", prompt_ids.len());
    println!("scored_tokens={}", stats.draft_tokens);
    println!("eligible_target_tokens={}", stats.eligible_target_tokens);
    println!("selected_target_tokens={}", stats.selected_target_tokens);
    println!("keep_ratio={keep_ratio:.4}");
    println!(
        "draft_scoring_seconds={:.3}",
        stats.draft_scoring_time.as_secs_f64()
    );
    println!(
        "target_prefill_seconds={:.3}",
        stats.target_prefill_time.as_secs_f64()
    );
    println!(
        "dense_prefill_seconds={:.3}",
        dense.prefill_time.as_secs_f64()
    );
    println!(
        "dense_decode_seconds={:.3}",
        dense.decode_time.as_secs_f64()
    );
    println!(
        "specprefill_decode_seconds={:.3}",
        sparse.decode_time.as_secs_f64()
    );
    println!("dense_output={:?}", dense.text);
    println!("specprefill_output={:?}", sparse.text);

    ensure!(
        stats.selected_target_tokens > 0
            && stats.selected_target_tokens < stats.eligible_target_tokens,
        "SpecPrefill must select a strict nonempty subset"
    );
    for (route, output) in [("dense", &dense.text), ("specprefill", &sparse.text)] {
        ensure!(!output.trim().is_empty(), "{route} output is empty");
        ensure!(
            output.contains(AUTHORITATIVE_CODE),
            "{route} output omitted authoritative code: {output:?}"
        );
        ensure!(
            DECOY_CODES.iter().all(|code| !output.contains(code)),
            "{route} output contained a decoy code: {output:?}"
        );
    }
    Ok(())
}
