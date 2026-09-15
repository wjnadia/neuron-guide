---
hidden: true
---

# (6호기 한강) 컨테이너 활용 가이드(작성중)

### 1. 개요

6호기 한강  시스템은 컨테이너 이미지 준비와 계산 작업 실행을 분리합니다. Podman은 OCI 이미지를 빌드하고 관리하는 도구입니다. 계산 작업은 `kisti-container`를 통해 Singularity, Apptainer, Enroot 또는 Pyxis로 실행합니다.

`kisti-container`는 다음 항목을 작업 특성에 맞게 구성합니다.

* 호스트 GPU와 컨테이너 연결
* Slurm rank와 torchrun rank 구성
* 단일 노드와 다중 노드의 NCCL 전송 방식 선택
* AWS OFI NCCL, Cray Libfabric 및 CXI 경로 연결
* 체크포인트 디렉터리의 쓰기 가능 마운트

[뉴론 시스템의 기존 컨테이너  활용 가이드](../)와 마찬가지로 Podman은 빌드·관리, `.sif`와 `.sqsh`는 계산 작업 실행에 사용합니다. 한강 시스템에서는 `kisti-container`가 런타임 별 옵션을 통일한다는 점이 추가됩니다.

### 2. 도구 선택

| 목적                 | 권장 도구       | 이미지 형식               | `kisti-container` 역할       |
| ------------------ | ----------- | -------------------- | -------------------------- |
| 이미지 빌드·수정·레지스트리 전송 | Podman      | OCI 이미지              | 실행 백엔드가 아님                 |
| 기존 HPC 이미지 실행      | Singularity | `.sif`               | GPU, MPI, NCCL 환경 구성       |
| Singularity 호환 실행  | Apptainer   | `.sif`               | GPU, MPI, NCCL 환경 구성       |
| GH200 작업의 직접 실행    | Enroot      | `.sqsh`              | Enroot 명령과 마운트 생성          |
| Slurm 통합 Enroot 실행 | Pyxis       | `.sqsh` 또는 Pyxis URI | `srun --container-*` 명령 생성 |

처음 사용하는 경우에는 기존 `.sif` 이미지가 있으면 Singularity/Apptainer를, GH200용 `.sqsh` 이미지가 있으면 Pyxis를 권장합니다. 장애 분석이나 Enroot 자체 동작 확인에는 `--runtime enroot`가 유용합니다.

### 3. 아키텍처와 이미지 호환성

컨테이너 이미지의 CPU 아키텍처는 실행 노드와 같아야 합니다.

| 6호기 프로파일     | CPU 아키텍처  | GPU          | Wrapper 옵션                  | 이미지 예시                                        |
| ------------ | --------- | ------------ | --------------------------- | --------------------------------------------- |
| GH200 GPU 노드 | `aarch64` | NVIDIA GH200 | `--platform gh200`          | `pytorch-aarch64.sif`, `pytorch-aarch64.sqsh` |
| AMD CPU 노드   | `x86_64`  | 없음           | `--platform amd --gpu none` | `ubuntu-x86_64.sif`                           |

현재 노드 아키텍처 확인:

```bash
uname -m
```

Podman 이미지 아키텍처 확인:

```bash
podman image inspect IMAGE:TAG --format '{{.Architecture}}'
```

`exec format error`가 발생하면 이미지와 계산 노드의 아키텍처를 먼저 비교합니다. Enroot/Pyxis는 `.sqsh`를 실행하기 위해 컨테이너를 시작하지 않고 아키텍처를 완전히 판별하기 어렵기 때문에 파일명에 `aarch64` 또는 `x86_64`를 넣는 것을 권장합니다.

### 4. 기본 확인

```bash
KISTI_CONTAINER=/apps/common/kisti-container/bin/kisti-container

$KISTI_CONTAINER --version
$KISTI_CONTAINER --help
srun --help | grep -- --container-image
```

마지막 명령에 `--container-image`가 표시되면 Pyxis가 Slurm에 등록된 상태입니다.

사이트 기본 설정은 다음 파일에 있습니다.

```
/apps/common/kisti-container/conf/kisti-container.conf
```

사용자  별 설정은 필요할 때만 다음 파일에 작성합니다.

```
$HOME/.config/kisti-container.conf
```

별도 설정 파일을 지정하려면 다음 환경 변수를 사용합니다.

```bash
export KISTI_CONTAINER_CONFIG=/absolute/path/kisti-container.conf
```

### 5. 기본 명령 형식

```bash
kisti-container [Wrapper 옵션] IMAGE COMMAND [ARG ...]
```

주요 옵션:

| 옵션                  | 값                                                 | 설명                        |
| ------------------- | ------------------------------------------------- | ------------------------- |
| `--runtime`         | `singularity`, `apptainer`, `enroot`, `pyxis`     | 실행 백엔드                    |
| `--platform`        | `gh200`, `amd`                                    | 노드 아키텍처와 사이트 프로파일         |
| `--workload`        | `generic`, `pytorch-ddp`, `pytorch-fsdp2`, `nemo` | 워크로드별 검증 규칙               |
| `--launcher`        | `srun-native`, `torchrun`                         | 분산 프로세스 생성 방식             |
| `--local-processes` | 양의 정수                                             | 노드별 torchrun worker 수     |
| `--mpi`             | `none`, `cray`, `openmpi`, `auto`                 | MPI 구성                    |
| `--nccl`            | `none`, `native`, `ofi`, `auto`                   | NCCL 전송 구성                |
| `--checkpoint-dir`  | 절대 경로                                             | 쓰기 가능한 체크포인트 경로           |
| `-B`, `--bind`      | `SRC:DST:MODE`                                    | 추가 마운트                    |
| `--env`             | `NAME=VALUE`                                      | 컨테이너에 전달할 환경변수            |
| `--diagnose`        | 값 없음                                              | rank 0에서 결정된 설정 출력        |
| `--doctor`          | 값 없음                                              | GPU, CXI, OFI 라이브러리 사전 점검 |
| `--dry-run`         | 값 없음                                              | 실행하지 않고 최종 명령 출력          |

### 6. 이미지 준비

#### 6.1 Podman 빌드

뉴론 가이드와 동일하게 Podman을 이용해 Dockerfile 기반 이미지를 준비할 수 있습니다. 6호기에서 Podman 사용 전 별도의 사이트 활성화 절차가 있다면 해당 운영 지침을 먼저 적용합니다.

```bash
cd /apps/common/kisti-container/examples/00-image-build
cp -a . /scratch/$USER/kisti-image-build
cd /scratch/$USER/kisti-image-build

./build-with-podman.sh
```

레지스트리에서 직접 가져오는 예:

```bash
podman pull nvcr.io/nvidia/pytorch:25.03-py3
podman images
```

#### 6.2 `.sqsh`와 `.sif` 변환

```bash
PODMAN_IMAGE=localhost/kisti-pytorch:tutorial \
OUTPUT_DIR=/scratch/$USER/container-images \
./convert-images.sh
```

직접 실행할 경우의 기본 명령은 다음과 같습니다.

```bash
enroot import -o pytorch-aarch64.sqsh podman://localhost/kisti-pytorch:tutorial

podman save localhost/kisti-pytorch:tutorial -o pytorch-aarch64.tar
singularity build --fakeroot pytorch-aarch64.sif \
  docker-archive://pytorch-aarch64.tar
```

로그인/계산 노드의 로컬 컨테이너 저장소는 영구 보관 대상으로 가정하지 않습니다. 완성한 이미지는 `/scratch` 등의 승인된 공유 경로 또는 [KISTI 내부 레지스트리](appendix-12-how-to-use-containers.md#id-3)에 보관해야합니다.

### 7. 배치 작업 실행 방식

#### 7.1 SingularityCE, Apptainer, Enroot

이 세 백엔드에서 `kisti-container`는 Slurm task 안에서 실행되는 Wrapper입니다. 따라서 배치 파일에 `srun`을 사용합니다.

```bash
srun --mpi=none \
  /apps/common/kisti-container/bin/kisti-container \
    --runtime enroot \
    --platform gh200 \
    IMAGE.sqsh python3 train.py
```

#### 7.2 Pyxis

Pyxis에서 `kisti-container`는 `srun` 명령을 생성하는 프론트엔드입니다. 배치 스크립트에서 직접 호출합니다.

```bash
/apps/common/kisti-container/bin/kisti-container \
  --runtime pyxis \
  --platform gh200 \
  IMAGE.sqsh python3 train.py
```

다음 형식은 사용하지 않습니다.

```bash
srun kisti-container --runtime pyxis ...
```

이 형식은 이미 생성된 Slurm step 안에서 Pyxis용 새 `srun`을 만들려고 하므로 Wrapper가 오류로 중단합니다.

### 8. 워크로드 프로파일

#### 8.1 일반 CPU/GPU 작업

`--workload generic`을 사용합니다. 단일 GPU에서는 `--nccl native` 또는 `--nccl none`을 선택할 수 있습니다.

```bash
sbatch --export=ALL,IMAGE=/absolute/path/pytorch-arm64.sqsh \
  /apps/common/kisti-container/examples/02-gpu-smoke/run-enroot.sbatch
```

성공 로그:

```
GPU_SMOKE_PASS host=... gpu=NVIDIA GH200 120GB ...
```

#### 8.2 PyTorch DDP

2노드 8GPU 예제는 노드당 4개의 Slurm rank를 생성합니다. `--nccl auto`는 2노드에서 사이트의 AWS OFI NCCL, Cray Libfabric, CXI 프로파일을 선택합니다.

```bash
sbatch --export=ALL,IMAGE=/absolute/path/pytorch-arm64.sqsh \
  /apps/common/kisti-container/examples/03-pytorch-ddp/run-pyxis.sbatch
```

결과 파일:

```
$SLURM_SUBMIT_DIR/checkpoints/ddp-pyxis-JOBID/ddp-smoke-result.json
```

#### 8.3 PyTorch FSDP2

FSDP2 예제는 노드당 Slurm task 1개를 만들고, 각 task 안에서 torchrun worker 4개를 실행합니다. 전체 `WORLD_SIZE`는 8입니다.

```bash
sbatch --export=ALL,IMAGE=/absolute/path/pytorch-arm64.sqsh \
  /apps/common/kisti-container/examples/04-pytorch-fsdp2/run-enroot.sbatch
```

성공 여부는 종료 코드와 `RESULT ... correctness=True` 로그로 확인합니다. 이 프로그램은 합성 데이터 기반 통신·학습 smoke test이며 실제 모델 성능 기준은 아닙니다.

#### 8.4 NeMo/Megatron TP=4, DP=2

NeMo 예제는 GH200 2노드에서 노드 내부 TP=4, 노드 간 DP=2를 확인합니다.

```bash
sbatch --export=ALL,IMAGE=/absolute/path/nemo-aarch64.sqsh \
  /apps/common/kisti-container/examples/05-nemo-megatron/run-pyxis.sbatch
```

컨테이너에는 NeMo, Megatron Core, Transformer Engine 및 호환 PyTorch/CUDA가 설치되어 있어야 합니다. 제공 smoke test의 검증 환경은 NeMo 2.3.0rc5, Megatron Core 0.12.0rc4, Transformer Engine 2.2.0 개발 버전 계열이었습니다.

### 9. 체크포인트와 데이터 마운트

체크포인트에는 계산 노드 모두에서 접근 가능한 절대 경로를 사용합니다.

```bash
CHECKPOINT_DIR=/scratch/$USER/checkpoints/run01
mkdir -p "$CHECKPOINT_DIR"
```

배치 제출 시 지정:

```bash
sbatch --export=ALL,IMAGE=/path/image.sqsh,CHECKPOINT_DIR=$CHECKPOINT_DIR \
  run-pyxis.sbatch
```

`CHECKPOINT_DIR`를 생략한 예제는 다음 형식으로 작업별 경로를 만듭니다.

```
${SLURM_SUBMIT_DIR}/checkpoints/WORKLOAD-RUNTIME-${SLURM_JOB_ID}
```

기존 체크포인트에서 재시작할 때는 동일한 `CHECKPOINT_DIR`를 명시합니다.

추가 데이터 마운트:

```bash
kisti-container ... \
  -B /scratch/$USER/dataset:/workspace/dataset:ro \
  -B /scratch/$USER/output:/workspace/output:rw \
  IMAGE COMMAND
```

### 10. 다중 노드 NCCL-OFI/CXI

2노드 이상 GH200 작업은 다음 모듈을 배치 스크립트에서 로드합니다.

```bash
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
```

사전 점검:

```bash
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --gpus-per-node=4 --gpu-bind=none --gres-flags=allow-task-sharing \
  --mpi=none \
  /apps/common/kisti-container/bin/kisti-container \
    --runtime enroot --platform gh200 --mpi none --nccl ofi \
    --gpu nvidia --doctor IMAGE.sqsh
```

정상 상태에서는 각 노드에서 다음 항목이 `PASS`로 표시됩니다.

```
NVIDIA=PASS
CXI=PASS
NCCL_PLUGIN=PASS
LIBFABRIC=PASS
LIBCXI=PASS
DOCTOR_PASS
```

### 11. 문제 해결

| 오류 또는 현상                                  | 확인 항목                                      | 조치                                           |
| ----------------------------------------- | ------------------------------------------ | -------------------------------------------- |
| `Command not found: nvidia-container-cli` | 계산 노드의 libnvidia-container 설치              | 관리자에게 노드 패키지/Enroot hook 확인 요청               |
| `libfuse3.so.4` 없음                        | `ldd $(command -v squashfuse)`             | 사이트의 SquashFUSE/FUSE3 설치 확인                  |
| Zstd `.sqsh` 마운트 실패                       | `ldd squashfuse`의 `libzstd.so.1`           | Zstd 지원 SquashFUSE 사용                        |
| `Pyxis is not registered`                 | `srun --help`의 `--container-image`         | Slurm SPANK 설정과 계산 노드 플러그인 확인                |
| `invoke kisti-container directly`         | Pyxis 앞에 외부 `srun` 사용 여부                   | 외부 `srun` 제거                                 |
| `network AWS Libfabric not found`         | OFI plugin과 libfabric 로딩                   | 두 모듈 로드, `--doctor` 실행                       |
| `Couldn't create domain ... -38`          | 컨테이너에서 CXI 환경과 라이브러리 중복                    | Cray libfabric 우선순위와 `/opt/amazon/efa` 혼입 확인 |
| TCPStore 접속 시간 초과                         | `MASTER_ADDR`, `MASTER_PORT`, rank 0 생존 여부 | 최초 rank 오류를 먼저 확인                            |
| `EADDRINUSE`                              | 동일 포트를 여러 rank 0이 사용                       | Wrapper의 Slurm rank 전달과 중복 launcher 확인       |
| 체크포인트가 생성되지 않음                            | 프로그램의 저장 구현, 공유 경로                         | `HPC_CHECKPOINT_DIR` 사용 여부와 쓰기 권한 확인         |

작업 상태와 종료 코드:

```bash
sacct -j JOBID --format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList%30
```

핵심 로그 검색:

```bash
grep -hE 'PASS|FAIL|NCCL|Libfabric|CXI|Traceback|BATCH_ERROR' \
  JOBNAME-JOBID.out JOBNAME-JOBID.err
```

### 12. 예제 파일 배포 및 사용

관리자 배포 위치:

```
/apps/common/kisti-container/examples
```

사용자는 예제를 자신의 작업 공간으로 복사합니다.

```bash
mkdir -p /scratch/$USER/kisti-container-tutorial
cp -a /apps/common/kisti-container/examples/. \
  /scratch/$USER/kisti-container-tutorial/
cd /scratch/$USER/kisti-container-tutorial
```

배치 파일의 `IMAGE`, 파티션, 시간, CPU/GPU 수 등을 확인한 뒤 제출합니다.&#x20;

###

### 13. 참고 자료

* [뉴론 컨테이너 활용 가이드](https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-12-how-to-use-containers)
* [NVIDIA Enroot](https://github.com/NVIDIA/enroot)
* [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis)
* [Podman build 문서](https://docs.podman.io/en/latest/markdown/podman-build.1.html)
* [Apptainer GPU 지원](https://apptainer.org/docs/user/main/gpu.html)
