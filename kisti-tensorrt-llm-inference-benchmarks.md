---
hidden: true
---

# KISTI TensorRT-LLM Inference Benchmarks

## KISTI TensorRT-LLM Inference Benchmarks

Throughput (tokens/sec) · higher is better

<details>

<summary>Legend</summary>

Testing was performed on models with quantized weights from NVIDIA's [Model Optimizer HuggingFace Collection](https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer).

* **PP**: Pipeline parallelism (multi-node inference)
* **TP**: Tensor parallelism (multi-GPU inference)
* **ISL**: Benchmark input sequence length
* **OSL**: Benchmark output sequence length
* **Requests**: The number of requests to generate for dataset generation
  * For shorter (ISL/OSL), a larger number of messages were used to guarantee that the system hit a steady state because requests enter and exit the system at a much faster rate
  * For longer (ISL/OSL), requests remain in the system longer and therefore require less requests to achieve steady state

</details>

| GPU            | Model                   | Precision | PP | TP | ISL    | OSL   | Requests | Throughput (tok/s) | Version |
| -------------- | ----------------------- | --------- | -- | -- | ------ | ----- | -------- | ------------------ | ------- |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 128    | 128   | 30,000   | 6,627              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 128    | 2,048 | 3,000    | 5,256              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 500    | 2,000 | 3,000    | 4,277              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 1,000  | 1,000 | 3,000    | 3,787              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 128    | 4,096 | 1,500    | 3,454              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 2,048  | 2,048 | 1,500    | 2,424              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 5,000  | 500   | 1,500    | 825                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 2,048  | 128   | 3,000    | 800                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP16      | 1  | 1  | 20,000 | 2,000 | 1,000    | 330                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 128    | 128   | 30,000   | 1,314              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 128    | 2,048 | 3,000    | 531                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 1,000  | 1,000 | 3,000    | 473                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 500    | 2,000 | 3,000    | 439                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 2,048  | 2,048 | 1,500    | 261                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 2,048  | 128   | 3,000    | 149                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 2  | 5,000  | 500   | 1,500    | 121                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 128    | 2,048 | 3,000    | 2,811              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 128    | 128   | 30,000   | 2,733              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 500    | 2,000 | 3,000    | 2,276              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 128    | 4,096 | 1,500    | 1,976              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 1,000  | 1,000 | 3,000    | 1,922              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 2,048  | 2,048 | 1,500    | 1,394              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 5,000  | 500   | 1,500    | 393                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 2,048  | 128   | 3,000    | 325                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 4  | 20,000 | 2,000 | 1,000    | 212                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 128    | 2,048 | 3,000    | 5,242              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 128    | 128   | 30,000   | 4,718              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 500    | 2,000 | 3,000    | 4,445              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 128    | 4,096 | 1,500    | 3,725              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 1,000  | 1,000 | 3,000    | 3,320              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 2,048  | 2,048 | 1,500    | 2,554              | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 5,000  | 500   | 1,500    | 696                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 2,048  | 128   | 3,000    | 542                | v0.17.0 |
| A100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP16      | 1  | 8  | 20,000 | 2,000 | 1,000    | 413                | v0.17.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 128   | 30,000   | 26,401             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 2,048 | 3,000    | 21,413             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 500    | 2,000 | 3,000    | 17,571             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 1,000 | 3,000    | 14,992             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 4,096 | 1,500    | 13,542             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 2,000 | 1,500    | 13,505             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,024  | 2,048 | 1,500    | 13,166             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 2,048 | 1,500    | 9,462              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 128   | 3,000    | 3,276              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 5,000  | 500   | 1,500    | 3,276              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 20,000 | 2,000 | 1,000    | 1,341              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 128    | 128   | 30,000   | 3,191              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 128    | 2,048 | 3,000    | 745                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,000  | 1,000 | 3,000    | 735                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,000  | 2,000 | 1,500    | 526                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,024  | 2,048 | 1,500    | 525                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 2,048  | 2,048 | 1,500    | 358                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 2,048  | 128   | 3,000    | 316                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 5,000  | 500   | 1,500    | 203                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 128   | 30,000   | 6,183              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 2,048 | 3,000    | 5,822              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 500    | 2,000 | 3,000    | 4,704              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 1,000 | 3,000    | 4,191              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 2,000 | 1,500    | 3,920              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,024  | 2,048 | 1,500    | 3,896              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 4,096 | 1,500    | 3,715              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 2,048 | 1,500    | 2,733              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 5,000  | 500   | 1,500    | 867                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 128   | 3,000    | 748                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 20,000 | 2,000 | 1,000    | 408                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 2,048 | 3,000    | 11,442             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 500    | 2,000 | 3,000    | 10,278             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 128   | 30,000   | 10,261             | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 4,096 | 1,500    | 8,210              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,000  | 2,000 | 1,500    | 7,590              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,024  | 2,048 | 1,500    | 7,557              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,000  | 1,000 | 3,000    | 7,427              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 2,048  | 2,048 | 1,500    | 5,640              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 5,000  | 500   | 1,500    | 1,572              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 2,048  | 128   | 3,000    | 1,240              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 20,000 | 2,000 | 1,000    | 911                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 2,048 | 3,000    | 4,572              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 128   | 30,000   | 3,732              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 500    | 2,000 | 3,000    | 3,662              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,000  | 2,000 | 1,500    | 3,253              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,024  | 2,048 | 1,500    | 3,089              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,000  | 1,000 | 3,000    | 2,963              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 4,096 | 1,500    | 2,911              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 2,048  | 2,048 | 1,500    | 2,140              | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 5,000  | 500   | 1,500    | 579                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 2,048  | 128   | 3,000    | 449                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 20,000 | 2,000 | 1,000    | 370                | v0.19.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 128   | 30,000   | 6,092              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 2,048 | 3,000    | 5,893              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 500    | 2,000 | 3,000    | 4,655              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 1,000 | 3,000    | 4,181              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 4,096 | 1,500    | 3,828              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,024  | 2,048 | 1,500    | 3,785              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 2,000 | 1,500    | 3,709              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 2,048 | 1,500    | 2,786              | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 5,000  | 500   | 1,500    | 866                | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 128   | 3,000    | 723                | v0.21.0 |
| H100 SXM 80GB  | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 20,000 | 2,000 | 1,000    | 412                | v0.21.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 128   | 30,000   | 27,028             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 2,048 | 3,000    | 23,102             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 500    | 2,000 | 3,000    | 19,759             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 4,096 | 1,500    | 17,397             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 1,000 | 3,000    | 17,162             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 2,000 | 1,500    | 16,227             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,024  | 2,048 | 1,500    | 16,058             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 2,048 | 1,500    | 11,822             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 5,000  | 500   | 1,500    | 3,758              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 128   | 3,000    | 3,391              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 20,000 | 2,000 | 1,000    | 1,706              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 128    | 2,048 | 3,000    | 4,351              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 128    | 128   | 30,000   | 3,658              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 500    | 2,000 | 3,000    | 3,476              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,000  | 2,000 | 1,500    | 2,914              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,024  | 2,048 | 1,500    | 2,893              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 1,000  | 1,000 | 3,000    | 2,727              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 128    | 4,096 | 1,500    | 2,697              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 2,048  | 2,048 | 1,500    | 1,990              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 5,000  | 500   | 1,500    | 544                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 2,048  | 128   | 3,000    | 433                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 1  | 20,000 | 2,000 | 1,000    | 277                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 2,048 | 3,000    | 8,450              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 500    | 2,000 | 3,000    | 6,712              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 128   | 30,000   | 6,478              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 2,000 | 1,500    | 5,841              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 128    | 4,096 | 1,500    | 5,599              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,024  | 2,048 | 1,500    | 5,565              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 1,000 | 3,000    | 5,097              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 2,048 | 1,500    | 3,823              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 5,000  | 500   | 1,500    | 1,006              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 128   | 3,000    | 773                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 2  | 20,000 | 2,000 | 1,000    | 618                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 2,048 | 3,000    | 13,439             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 500    | 2,000 | 3,000    | 12,332             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 4,096 | 1,500    | 11,525             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 128    | 128   | 30,000   | 10,466             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,024  | 2,048 | 1,500    | 9,018              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,000  | 2,000 | 1,500    | 9,016              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 1,000  | 1,000 | 3,000    | 8,698              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 2,048  | 2,048 | 1,500    | 7,069              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 5,000  | 500   | 1,500    | 1,715              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 2,048  | 128   | 3,000    | 1,278              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 4  | 20,000 | 2,000 | 1,000    | 1,175              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 128    | 2,048 | 3,000    | 20,751             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 500    | 2,000 | 3,000    | 17,311             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 128    | 4,096 | 1,500    | 16,635             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 128    | 128   | 30,000   | 15,555             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 1,000  | 2,000 | 1,500    | 13,175             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 1,024  | 2,048 | 1,500    | 13,117             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 1,000  | 1,000 | 3,000    | 12,795             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 2,048  | 2,048 | 1,500    | 10,529             | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 5,000  | 500   | 1,500    | 2,683              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 20,000 | 2,000 | 1,000    | 2,021              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-70B-Instruct  | FP8       | 1  | 8  | 2,048  | 128   | 3,000    | 1,947              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 2,048 | 3,000    | 5,661              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 4,096 | 1,500    | 5,167              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 500    | 2,000 | 3,000    | 4,854              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 128    | 128   | 30,000   | 3,800              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,024  | 2,048 | 1,500    | 3,686              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,000  | 2,000 | 1,500    | 3,682              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 1,000  | 1,000 | 3,000    | 3,332              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 2,048  | 2,048 | 1,500    | 3,056              | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 5,000  | 500   | 1,500    | 656                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 20,000 | 2,000 | 1,000    | 514                | v0.19.0 |
| H200 SXM 141GB | Llama-3.1-405B-Instruct | FP8       | 1  | 8  | 2,048  | 128   | 3,000    | 453                | v0.19.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 2,048 | 3,000    | 7,467              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 500    | 2,000 | 3,000    | 6,639              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 128   | 30,000   | 6,328              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 2,000 | 1,500    | 5,790              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 128    | 4,096 | 1,500    | 5,526              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,024  | 2,048 | 1,500    | 5,480              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 1,000  | 1,000 | 3,000    | 4,773              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 2,048 | 1,500    | 3,776              | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 5,000  | 500   | 1,500    | 978                | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 2,048  | 128   | 3,000    | 748                | v0.21.0 |
| H200 SXM 141GB | Llama-3.3-70B-Instruct  | FP8       | 1  | 2  | 20,000 | 2,000 | 1,000    | 609                | v0.21.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 128   | 30,000   | 27,304             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 2,048 | 3,000    | 24,046             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 500    | 2,000 | 3,000    | 20,124             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 1,000 | 3,000    | 16,353             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,024  | 2,048 | 1,500    | 16,103             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 1,000  | 2,000 | 1,500    | 15,706             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 128    | 4,096 | 1,500    | 15,410             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 2,048 | 1,500    | 10,767             | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 5,000  | 500   | 1,500    | 3,585              | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 2,048  | 128   | 3,000    | 3,574              | v0.19.0 |
| GH200 96GB     | Llama-3.1-8B-Instruct   | FP8       | 1  | 1  | 20,000 | 2,000 | 1,000    | 1,393              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 128    | 2,048 | 3,000    | 10,387             | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 500    | 2,000 | 3,000    | 9,242              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 128    | 128   | 30,000   | 9,185              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 128    | 4,096 | 1,500    | 8,742              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 1,000  | 2,000 | 1,500    | 7,697              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 1,024  | 2,048 | 1,500    | 7,569              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 1,000  | 1,000 | 3,000    | 7,566              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 2,048  | 2,048 | 1,500    | 6,092              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 5,000  | 500   | 1,500    | 1,332              | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 20,000 | 2,000 | 1,000    | 962                | v0.19.0 |
| B200 180GB     | Llama-3.1-405B-Instruct | NVFP4     | 1  | 8  | 2,048  | 128   | 3,000    | 954                | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 128   | 30,000   | 11,253             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 2,048 | 3,000    | 9,925              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 500    | 2,000 | 3,000    | 7,560              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,000  | 1,000 | 3,000    | 6,867              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,000  | 2,000 | 1,500    | 6,737              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,024  | 2,048 | 1,500    | 6,581              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 4,096 | 1,500    | 6,319              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 2,048  | 2,048 | 1,500    | 4,545              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 5,000  | 500   | 1,500    | 1,488              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 2,048  | 128   | 3,000    | 1,375              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 20,000 | 2,000 | 1,000    | 581                | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 128    | 128   | 30,000   | 17,868             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 128    | 2,048 | 3,000    | 15,460             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 1,000  | 1,000 | 3,000    | 10,838             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 500    | 2,000 | 3,000    | 10,602             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 1,000  | 2,000 | 1,500    | 9,132              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 1,024  | 2,048 | 1,500    | 8,767              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 128    | 4,096 | 1,500    | 8,712              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 2,048  | 2,048 | 1,500    | 6,956              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 5,000  | 500   | 1,500    | 2,380              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 2,048  | 128   | 3,000    | 1,611              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 2  | 20,000 | 2,000 | 1,000    | 1,044              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 128    | 128   | 30,000   | 24,944             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 128    | 2,048 | 3,000    | 23,609             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 500    | 2,000 | 3,000    | 20,910             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 128    | 4,096 | 1,500    | 17,660             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 1,000  | 1,000 | 3,000    | 16,568             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 1,000  | 2,000 | 1,500    | 15,737             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 1,024  | 2,048 | 1,500    | 15,723             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 2,048  | 2,048 | 1,500    | 12,292             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 5,000  | 500   | 1,500    | 3,588              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 2,048  | 128   | 3,000    | 2,708              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 4  | 20,000 | 2,000 | 1,000    | 1,958              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 128    | 2,048 | 3,000    | 30,743             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 500    | 2,000 | 3,000    | 28,182             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 128    | 128   | 30,000   | 27,471             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 128    | 4,096 | 1,500    | 24,947             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 1,000  | 2,000 | 1,500    | 20,518             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 1,024  | 2,048 | 1,500    | 20,438             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 1,000  | 1,000 | 3,000    | 19,992             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 2,048  | 2,048 | 1,500    | 15,661             | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 5,000  | 500   | 1,500    | 4,810              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 2,048  | 128   | 3,000    | 3,718              | v0.19.0 |
| B200 180GB     | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 8  | 20,000 | 2,000 | 1,000    | 3,167              | v0.19.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 128    | 2,048 | 3,000    | 7,497              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 128    | 128   | 30,000   | 6,599              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 500    | 2,000 | 3,000    | 6,198              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 128    | 4,096 | 1,500    | 5,898              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 1,000  | 1,000 | 3,000    | 5,243              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 1,000  | 2,000 | 1,500    | 4,906              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 1,024  | 2,048 | 1,500    | 4,686              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 2,048  | 2,048 | 1,500    | 4,327              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 5,000  | 500   | 1,500    | 1,079              | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 2,048  | 128   | 3,000    | 762                | v0.21.0 |
| GB200 196GB    | Llama-3.1-405B-Instruct | NVFP4     | 1  | 4  | 20,000 | 2,000 | 1,000    | 650                | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 128   | 30,000   | 11,101             | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 2,048 | 3,000    | 10,276             | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 500    | 2,000 | 3,000    | 8,194              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,024  | 2,048 | 1,500    | 7,923              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,000  | 1,000 | 3,000    | 7,402              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 128    | 4,096 | 1,500    | 7,351              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 1,000  | 2,000 | 1,500    | 6,479              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 2,048  | 2,048 | 1,500    | 5,327              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 5,000  | 500   | 1,500    | 1,502              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 2,048  | 128   | 3,000    | 1,418              | v0.21.0 |
| GB200 196GB    | Llama-3.3-70B-Instruct  | NVFP4     | 1  | 1  | 20,000 | 2,000 | 1,000    | 732                | v0.21.0 |

<details>

<summary>Legend</summary>

Benchmarks run on Neuron cluster at KISTI using TensorRT-LLM.

* **PP**: Pipeline parallelism (multi-node inference)
* **TP**: Tensor parallelism (multi-GPU inference)
* **ISL**: Benchmark input sequence length
* **OSL**: Benchmark output sequence length
* **Requests**: The number of requests to generate for dataset generation
  * For shorter (ISL/OSL), a larger number of messages were used to guarantee that the system hit a steady state because requests enter and exit the system at a much faster rate
  * For longer (ISL/OSL), requests remain in the system longer and therefore require less requests to achieve steady state

</details>

| GPU            | Model                  | Precision | PP | TP | ISL   | OSL   | Requests | Throughput (tok/s) | Version |
| -------------- | ---------------------- | --------- | -- | -- | ----- | ----- | -------- | ------------------ | ------- |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 128   | 30,000   | 1,918              | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 2,048 | 3,000    | 642                | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 2,048 | 128   | 3,000    | 204                | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 128   | 30,000   | 2,206              | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 2,048 | 3,000    | 1,357              | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 2,048 | 128   | 3,000    | 313                | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 2,048 | 3,000    | 2,896              | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 128   | 30,000   | 2,841              | v0.14.0 |
| V100 PCIE 32GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 2,048 | 128   | 3,000    | 360                | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 128   | 30,000   | 2,006              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 2,048 | 3,000    | 675                | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 2,048 | 128   | 3,000    | 210                | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 128   | 30,000   | 3,164              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 2,048 | 3,000    | 1,430              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 2,048 | 128   | 3,000    | 369                | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 128   | 30,000   | 6,788              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 2,048 | 3,000    | 4,495              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 2,048 | 128   | 3,000    | 834                | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 128   | 30,000   | 10,607             | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 2,048 | 3,000    | 9,062              | v0.14.0 |
| V100 SXM 32GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 2,048 | 128   | 3,000    | 1,140              | v0.14.0 |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 128   | 30,000   | 6,658              | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 2,048 | 3,000    | 5,165              | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 2,048 | 128   | 3,000    | 793                | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 128   | 30,000   | 11,049             | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 2,048 | 3,000    | 9,893              | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 2,048 | 128   | 3,000    | 1,404              | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 2,048 | 3,000    | 17,737             | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 128   | 30,000   | 17,062             | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 2,048 | 128   | 3,000    | 2,261              | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 2,048 | 3,000    | 25,740             | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 128   | 30,000   | 22,786             | v1.2.0  |
| A100 SXM 80GB  | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 2,048 | 128   | 3,000    | 3,161              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 128   | 30,000   | 16,660             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 128   | 2,048 | 3,000    | 13,467             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 1  | 2,048 | 128   | 3,000    | 2,125              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 128   | 30,000   | 27,713             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 128   | 2,048 | 3,000    | 24,324             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 2  | 2,048 | 128   | 3,000    | 3,587              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 2,048 | 3,000    | 45,311             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 128   | 128   | 30,000   | 43,538             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 4  | 2,048 | 128   | 3,000    | 5,877              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 2,048 | 3,000    | 70,256             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 128   | 128   | 30,000   | 61,139             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP16      | 1  | 8  | 2,048 | 128   | 3,000    | 8,682              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 128   | 128   | 30,000   | 29,103             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 128   | 2,048 | 3,000    | 25,640             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 2,048 | 128   | 3,000    | 3,643              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 1  | 128   | 2,048 | 3,000    | 4,086              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 1  | 128   | 128   | 30,000   | 3,644              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 1  | 2,048 | 128   | 3,000    | 461                | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 2  | 128   | 128   | 30,000   | 6,667              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 2  | 128   | 2,048 | 3,000    | 6,274              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 2  | 2,048 | 128   | 3,000    | 789                | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 4  | 128   | 2,048 | 3,000    | 11,674             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 4  | 128   | 128   | 30,000   | 10,698             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 4  | 2,048 | 128   | 3,000    | 1,283              | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 8  | 128   | 2,048 | 3,000    | 24,066             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 8  | 128   | 128   | 30,000   | 15,487             | v1.2.0  |
| H200 SXM 141GB | Llama-3.1-70B-Instruct | FP8       | 1  | 8  | 2,048 | 128   | 3,000    | 1,884              | v1.2.0  |
| GH200 96GB     | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 128   | 128   | 30,000   | 32,233             | v1.2.0  |
| GH200 96GB     | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 128   | 2,048 | 3,000    | 21,081             | v1.2.0  |
| GH200 96GB     | Llama-3.1-8B-Instruct  | FP8       | 1  | 1  | 2,048 | 128   | 3,000    | 3,964              | v1.2.0  |

### Release Notes

{% updates format="full" %}
{% update date="2026-03-10" %}
## v1.2.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.0)

#### Key Features & Enhancements

* Added beta support for K-EXAONE, Nemotron Nano V3, Qwen3-Next and Qwen3-VL.
* Validated models and precision formats:
  * GPT-OSS-20B, GPT-OSS-120B (MXFP4)
  * Llama-3.1-8B-Instruct (FP16/FP8/NVFP4)
  * Llama-3.3-70B-Instruct (FP8/NVFP4)
  * Qwen3-8B, Qwen3-14B (FP16/FP8/NVFP4)
  * Qwen3-32B (FP16/NVFP4)
  * Qwen3-30B-A3B (FP16/NVFP4)
  * NVIDIA-Nemotron-Nano-9B-v2 (FP4)
  * Llama-3.3-Nemotron-Super-49B-v1.5 (FP8)
  * Phi-4-multimodal-instruct (FP16/FP8/NVFP4)
  * Phi-4-reasoning-plus (FP16/FP8/NVFP4)
* Added NUMA-aware CPU affinity automatic configuration.

#### Known Issues

* A hang may occur in disaggregated serving with context pipeline parallelism and generation tensor parallelism configurations.
{% endupdate %}

{% update date="2025-12-12" %}
## v1.1.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.1.0)

#### Key Features & Enhancements

* Added support for B300/GB300.
* Added dedicated performance tests for disaggregated serving scenarios.
* Added specific performance test cases for NIM (NVIDIA Inference Microservices) integration.
* Enhanced reporting to include KV cache size metrics in benchmark results.

#### Known Issues

* Support for GB300 in multi-node configurations is currently in beta and not fully validated in this release.
{% endupdate %}

{% update date="2025-09-24" %}
## v1.0.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.0.0)

#### Key Features & Enhancements

* PyTorch is now the default LLM backend.
* Added support for GB10 (sm121).
* Added EXAONE 4.0 model support.
* Huge page mapping for host accessible memory on GB200.
* Added latency support for trtllm-bench.

#### Known Issues

* When using disaggregated serving with pipeline parallelism and KV cache reuse, a hang can occur.
* Running multi-node cases where each node has just a single GPU is known to fail.
{% endupdate %}

{% update date="2025-08-01" %}
## v0.21.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.21.0)

#### Key Features & Enhancements

* Added DeepSeek R1 FP8 support on Blackwell.
* Validated Llama 3.1 models on H200 NVL.
* Added all\_reduce.py benchmark script for testing.
{% endupdate %}

{% update date="2025-06-18" %}
## v0.20.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.20.0)

#### Key Features & Enhancements

* Added Qwen3 support.
* Supported DeepSeek-R1 W4A8 on Hopper.
* Added RTX Pro 6000 support on single GPU.

#### Infrastructure Changes

* TRT-LLM team formally releases Docker image on NGC.

#### Known Issues

* Multi-GPU model support on RTX Pro 6000.
{% endupdate %}

{% update date="2025-03-09" %}
## v0.19.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.19.0)

#### Key Features & Enhancements

* The C++ runtime is now open sourced.
{% endupdate %}

{% update date="2025-02-11" %}
## v0.17.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.17.0)

#### Key Features & Enhancements

* Added support for B200 (GB200 NVL is not fully supported).
* Added benchmark script to measure performance benefits of KV cache host offload with expected runtime improvements from GH200.
* PyTorch workflow experimental support on H100/H200/B200.
{% endupdate %}

{% update date="2024-12-24" %}
## v0.16.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.16.0)

#### Key Features & Enhancements

* Breaking Removed NVIDIA V100 GPU support.

#### Known Issues

* Known AllReduce performance issue on AMD-based CPU platforms on NCCL 2.23.4 — workaround: `export NCCL_P2P_LEVEL=SYS`.
{% endupdate %}

{% update date="2024-12-04" %}
## v0.15.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.15.0)

#### Key Features & Enhancements

* Added functional support for GH200 systems.
* Breaking NVIDIA Volta GPU support has been removed in this and all future releases.
{% endupdate %}

{% update date="2024-11-01" %}
## v0.14.0

[↗ GitHub](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v0.14.0)

#### Key Features & Enhancements

* NVIDIA Volta GPU support is deprecated and will be removed in a future release.
{% endupdate %}

{% update date="2024-09-30" %}
## v0.13.0

#### Infrastructure Changes

* TensorRT updated to **10.4.0**.
* CUDA updated to **12.5.1**.
* PyTorch updated to **2.4.0**.
* ModelOpt updated to **v0.15**.
{% endupdate %}
{% endupdates %}

