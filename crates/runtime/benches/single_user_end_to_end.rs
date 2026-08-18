mod support;

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use qw_runtime::provider::Qwen35GenerationMode;
use support::{MTP_BLOCK_SIZE, prepare_decode_fixture};

fn single_user_end_to_end(criterion: &mut Criterion) {
    let mut provider = support::load_provider();
    let decode_fixture = prepare_decode_fixture(&mut provider);
    let decode_request = decode_fixture.request;
    let baseline_token_ids = decode_fixture.baseline_token_ids;
    let mtp_token_ids = decode_fixture.mtp_token_ids;
    let baseline_tokens = baseline_token_ids.len() as u64;
    let mtp_tokens = decode_fixture.mtp_decode_tokens as u64;

    let mut group = criterion.benchmark_group("single_user_end_to_end");
    group.throughput(Throughput::Elements(baseline_tokens));
    group.bench_function("baseline", |bencher| {
        bencher.iter(|| {
            let output = provider
                .generate_streaming_in_mode(
                    &decode_request,
                    Qwen35GenerationMode::Baseline,
                    |delta| {
                        black_box(delta);
                        true
                    },
                )
                .expect("benchmark baseline single-user end-to-end")
                .0;
            assert_eq!(
                &output.token_ids, &baseline_token_ids,
                "deterministic baseline token IDs changed"
            );
            black_box(output);
        });
    });
    group.throughput(Throughput::Elements(mtp_tokens));
    group.bench_function("mtp_k3_e2e", |bencher| {
        bencher.iter(|| {
            let output = provider
                .generate_streaming_in_mode(
                    &decode_request,
                    Qwen35GenerationMode::Mtp,
                    |delta| {
                        black_box(delta);
                        true
                    },
                )
                .unwrap_or_else(|error| panic!("benchmark MTP k={MTP_BLOCK_SIZE}: {error:#}"))
                .0;
            assert_eq!(
                &output.token_ids, &mtp_token_ids,
                "deterministic bundled-MTP k={MTP_BLOCK_SIZE} greedy token IDs changed"
            );
            black_box(output);
        });
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));
    targets = single_user_end_to_end
}
criterion_main!(benches);
