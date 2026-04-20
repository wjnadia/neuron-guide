---
hidden: true
---

# TensorRT-LLM 추론 성능 벤치마크

본 페이지는 KISTI 뉴론 시스템에서 측정한 NVIDIA TensorRT-LLM 프레임워크의 LLM 추론 성능 벤치마크 결과를 제공합니다.

A100 SXM 80GB를 포함한 다양한 NVIDIA GPU 환경에서 Llama-3.1-8B-Instruct 등 주요 오픈소스 모델을 대상으로, FP16·FP8·NVFP4 등 정밀도 설정과 다양한 입출력 시퀀스 길이(ISL/OSL) 조합에 따른 처리량(tokens/sec)을 측정 하였습니다.&#x20;

결과는 아래 웹페이지에서 인터랙티브 필터를 통해 GPU, 모델, 정밀도, 시퀀스 길이 조합별 성능을 직접 비교해 볼 수 있습니다.

{% embed url="https://vitduck.github.io/KISTI-llmbench" %}

***

## Methods

### Dependencies

```bash
pip install pyyaml tabulate rich
```

### Workflows

| Workflow  | Description             | Supported Steps                                            |
| --------- | ----------------------- | ---------------------------------------------------------- |
| `legacy`  | Compiled TRT-LLM engine | download, convert, build, data, throughput, summary, clean |
| `pytorch` | HF-based inference      | download, data, throughput, summary, clean                 |

### Steps

| Step         | Description                                          | Output                                                       |
| ------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| `download`   | Fetch HF checkpoint                                  | `hf-models/<model>/`                                         |
| `convert`    | Convert HF → TRT-LLM checkpoint (legacy only)        | `ckpts/<model>/<quant>/pp<P>-tp<T>/`                         |
| `build`      | Compile TRT-LLM engine (legacy only)                 | `engines/<model>/<quant>/pp<P>-tp<T>-sl<S>-tk<T>-bs<B>/`     |
| `data`       | Generate synthetic datasets                          | `datasets/synthetic-il<N>-ol<N>-rq<N>.txt`                   |
| `throughput` | Run `trtllm-bench`, sweep over all cases             | `throughput/<model>/<quant>/pp<P>-tp<T>/output-...-<ts>.log` |
| `summary`    | Print throughput table; scan all historical logs     | `logs/summary-<config>-<workflow>-<model>-<ts>.log`          |
| `clean`      | Delete artifacts scoped to exact config combinations | —                                                            |

### Config

```yaml
workflow: pytorch
sif: tensorrt_llm_v1.2.0.sif
model: nvidia/Llama-3.1-70B-Instruct-FP8

# parallelism
pp_sizes: [1]
tp_sizes: [1, 2, 4, 8]

# engine parameters
max_seq_len: 4096
max_num_tokens: [8192]
max_batch_sizes: [2048]

# throughput
workloads:
  - {input_mean: 128,  output_mean: 128,  num_requests: 30000}
  - {input_mean: 128,  output_mean: 2048, num_requests: 3000}
  - {input_mean: 2048, output_mean: 128,  num_requests: 3000}

kv_cache_fraction: 0.95
```

### Usage

```bash
./llama3.py --config config.yaml <steps>
```

Steps are comma-separated with no spaces:

```bash
# pytorch — full run
./llama3.py --config config.yaml download,data,throughput,summary

# legacy — full run
./llama3.py --config config.yaml download,convert,build,data,throughput,summary

# summary only (scans all historical logs)
./llama3.py --config config.yaml summary

# clean artifacts for this config
./llama3.py --config config.yaml clean
```

### Field Codes

| Code | Meaning            |
| ---- | ------------------ |
| `il` | input\_mean        |
| `ol` | output\_mean       |
| `rq` | num\_requests      |
| `pp` | pipeline\_parallel |
| `tp` | tensor\_parallel   |
| `sl` | max\_seq\_len      |
| `tk` | max\_num\_tokens   |
| `bs` | max\_batch\_size   |
| `ts` | run timestamp      |

### Result: H200 SXM 141 GB

```
  pipeline    tensor    input len    output len    requests    max tokens    batch size    run timestamp    Throughput (tok/s)
----------  --------  -----------  ------------  ----------  ------------  ------------  ---------------  --------------------
         1         1          128           128       30000          8192          2048  20260415-114432                  3644
         1         1          128          2048        3000          8192          2048  20260415-114432                  4086
         1         1         2048           128        3000          8192          2048  20260415-114432                   461
         1         2          128           128       30000          8192          2048  20260415-114432                  6667
         1         2          128          2048        3000          8192          2048  20260415-114432                  6274
         1         2         2048           128        3000          8192          2048  20260415-114432                   789
         1         4          128           128       30000          8192          2048  20260415-114432                 10698
         1         4          128          2048        3000          8192          2048  20260415-114432                 11674
         1         4         2048           128        3000          8192          2048  20260415-114432                  1283
         1         8          128           128       30000          8192          2048  20260415-114432                 15487
         1         8          128          2048        3000          8192          2048  20260415-114432                 24066
         1         8         2048           128        3000          8192          2048  20260415-114432                  1884
```
