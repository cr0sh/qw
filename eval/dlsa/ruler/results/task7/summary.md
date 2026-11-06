### RULER 4096: task scores

| task | dense (n) | dlsa (n) | paired n | dense paired | dlsa paired | Δ dlsa − dense |
|---|---|---|---|---|---|---|
| niah_single_1 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_single_2 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_single_3 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_multikey_1 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_multikey_2 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_multikey_3 | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_multiquery | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| niah_multivalue | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| vt | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| cwe | 0.945 (20) | 0.900 (2) | 2 | 0.900 | 0.900 | +0.000 |
| fwe | 1.000 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| qa_squad | 0.900 (20) | 1.000 (2) | 2 | 1.000 | 1.000 | +0.000 |
| qa_hotpot | 0.650 (20) | 0.500 (2) | 2 | 0.500 | 0.500 | +0.000 |
| **mean** (tasks: own 13/13, paired 13) | **0.961** | **0.954** |  | **0.954** | **0.954** | **+0.000** |

vs upstream D-LSA (Flat) bf16, README.md:180 (paper 0.975 / 0.966): 0.974 at 4096
- dense: own mean -0.013, paired mean -0.020
- dlsa: own mean -0.020, paired mean -0.020

### RULER 32768: task scores

| task | dense (n) | dlsa (n) | paired n | dense paired | dlsa paired | Δ dlsa − dense |
|---|---|---|---|---|---|---|
| niah_single_1 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_single_2 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_single_3 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_multikey_1 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_multikey_2 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_multikey_3 | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_multiquery | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| niah_multivalue | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| vt | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| cwe | 0.980 (10) | 0.990 (10) | 10 | 0.980 | 0.990 | +0.010 |
| fwe | 1.000 (10) | 1.000 (10) | 10 | 1.000 | 1.000 | +0.000 |
| qa_squad | 0.800 (10) | 0.800 (10) | 10 | 0.800 | 0.800 | +0.000 |
| qa_hotpot | 0.800 (10) | 0.800 (10) | 10 | 0.800 | 0.800 | +0.000 |
| **mean** (tasks: own 13/13, paired 13) | **0.968** | **0.968** |  | **0.968** | **0.968** | **+0.001** |

vs upstream D-LSA (Flat) bf16, README.md:180 (paper 0.975 / 0.966): 0.968 at 32768
- dense: own mean -0.000, paired mean -0.000
- dlsa: own mean +0.000, paired mean +0.000

### Timing

| config | length | rows | errors | mean s | median s | mean prompt tok | mean completion tok | mean cached tok | sum of latencies h | first start → last finish h | est. h for 100 rows × 13 tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 4096 | 260 | 0 | 20.1 | 16.2 | 3716 | 284.2 | 15 | 1.45 | 1.45 | 7.3 |
| dlsa | 4096 | 26 | 0 | 21.0 | 16.3 | 3734 | 322.5 | 0 | 0.15 | 0.15 | 7.6 |
| dense | 32768 | 130 | 0 | 148.2 | 144.7 | 32026 | 286.6 | 0 | 5.35 | 5.35 | 53.5 |
| dlsa | 32768 | 130 | 0 | 141.7 | 137.5 | 32026 | 334.6 | 0 | 5.12 | 5.12 | 51.2 |

The estimate is N × the sum over the 13 tasks of each task's mean latency (a task not run yet takes the
length's mean), one request at a time.

### Errors and replies cut at max_tokens

| config | length | task | index | issue | completion tok | score |
|---|---|---|---|---|---|---|
| dense | 4096 | cwe | 0 | finish_reason=length | 4096 | 0.800 |
| dense | 4096 | cwe | 3 | finish_reason=length | 4096 | 0.800 |
| dense | 4096 | cwe | 4 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 5 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 6 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 7 | finish_reason=length | 4096 | 0.700 |
| dense | 4096 | cwe | 9 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 10 | finish_reason=length | 4096 | 0.600 |
| dense | 4096 | cwe | 14 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 17 | finish_reason=length | 4096 | 1.000 |
| dense | 4096 | cwe | 18 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 1 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 3 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 4 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 5 | finish_reason=length | 4096 | 0.800 |
| dense | 32768 | cwe | 6 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 7 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 8 | finish_reason=length | 4096 | 1.000 |
| dense | 32768 | cwe | 9 | finish_reason=length | 4096 | 1.000 |
| dlsa | 4096 | cwe | 0 | finish_reason=length | 4096 | 0.800 |
| dlsa | 32768 | cwe | 0 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 2 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 3 | finish_reason=length | 4096 | 0.900 |
| dlsa | 32768 | cwe | 4 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 6 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 7 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 8 | finish_reason=length | 4096 | 1.000 |
| dlsa | 32768 | cwe | 9 | finish_reason=length | 4096 | 1.000 |

### Agreement 4096: dlsa vs dense (rows both finished)

| task | paired n | score differs | dlsa lower (indices) | dlsa higher (indices) | reply text differs |
|---|---|---|---|---|---|
| niah_single_1 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_single_2 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_single_3 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_multikey_1 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_multikey_2 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_multikey_3 | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_multiquery | 2 | 0 | 0 [] | 0 [] | 0 |
| niah_multivalue | 2 | 0 | 0 [] | 0 [] | 0 |
| vt | 2 | 0 | 0 [] | 0 [] | 0 |
| cwe | 2 | 0 | 0 [] | 0 [] | 0 |
| fwe | 2 | 0 | 0 [] | 0 [] | 0 |
| qa_squad | 2 | 0 | 0 [] | 0 [] | 0 |
| qa_hotpot | 2 | 0 | 0 [] | 0 [] | 0 |

### Agreement 32768: dlsa vs dense (rows both finished)

| task | paired n | score differs | dlsa lower (indices) | dlsa higher (indices) | reply text differs |
|---|---|---|---|---|---|
| niah_single_1 | 10 | 0 | 0 [] | 0 [] | 0 |
| niah_single_2 | 10 | 0 | 0 [] | 0 [] | 1 |
| niah_single_3 | 10 | 0 | 0 [] | 0 [] | 3 |
| niah_multikey_1 | 10 | 0 | 0 [] | 0 [] | 0 |
| niah_multikey_2 | 10 | 0 | 0 [] | 0 [] | 3 |
| niah_multikey_3 | 10 | 0 | 0 [] | 0 [] | 4 |
| niah_multiquery | 10 | 0 | 0 [] | 0 [] | 1 |
| niah_multivalue | 10 | 0 | 0 [] | 0 [] | 0 |
| vt | 10 | 0 | 0 [] | 0 [] | 2 |
| cwe | 10 | 2 | 1 [3] | 1 [5] | 10 |
| fwe | 10 | 0 | 0 [] | 0 [] | 0 |
| qa_squad | 10 | 0 | 0 [] | 0 [] | 0 |
| qa_hotpot | 10 | 0 | 0 [] | 0 [] | 0 |
