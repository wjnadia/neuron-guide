---
description: 수정 중이므로 10월 19일(월) 이후 사용 가능함
hidden: true
---

# (6호기 한강) 컨테이너 활용 가이드

### 1. 개요

6호기 한강  시스템에서 컨테이너 활용 환경은  [뉴론 시스템의 기존 컨테이너  활용 가이드](appendix-12-how-to-use-containers.md)와 마찬가지로 이미지 준비와 계산 작업 실행으로 구분하고 있습니다.&#x20;

Podman은 OCI 이미지를 빌드하고 관리하는 도구이고, 계산  작업을 실행하기 위해서는 Singularity, Apptainer, Enroot 또는 Pyxis와 같은 컨테이너 런타임을 선택적으로 사용할 수 있습니다.

또한, 병렬 연산 및 분산  학습 작업에서 한강 시스템의 고성능 네트워크(Slingshot) 기반의 MPI 및 NCCL 환경 구성을 지원하는 컨테이너 런타임 Wrapper 프로그램(kisti-container)을 제공하고 있습니다.   &#x20;

### 2. 도구 선택

<table data-header-hidden><thead><tr><th width="258.115966796875"></th><th width="118.81158447265625"></th><th width="121.5072021484375"></th><th></th></tr></thead><tbody><tr><td>목적</td><td>권장 도구</td><td>이미지 형식</td><td><code>kisti-container</code> 역할</td></tr><tr><td>이미지 빌드·수정·레지스트리 전송</td><td>Podman 4.9.5 </td><td>OCI 이미지</td><td>실행 백엔드가 아님</td></tr><tr><td>기존 Singularity 이미지 실행</td><td>Singularity 4.5.0</td><td rowspan="2"><code>.sif 파일</code></td><td rowspan="4">GPU, MPI, NCCL 환경 구성</td></tr><tr><td>Singularity 호환 실행</td><td>Apptainer 1.4.5</td></tr><tr><td>GH200 작업의 직접 실행</td><td>Enroot 4.2.0</td><td rowspan="2"><code>.sqsh 파일</code></td></tr><tr><td>Slurm 통합 Enroot 실행</td><td>Pyxis 0.24.0.1</td></tr></tbody></table>

처음 사용하는 경우에는 기존 `.sif` 이미지가 있으면 Singularity/Apptainer를, GH200용 `.sqsh` 이미지가 있으면 Pyxis를 권장합니다.&#x20;

{% hint style="info" %}
**도구 별 사용법 참고 자료**

* [뉴론 컨테이너 활용 가이드](https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-12-how-to-use-containers) (Podman, Singularity, Enroot, Pyxis)  &#x20;
* [뉴론 Singularity 컨테이너](https://docs-ksc.gitbook.io/neuron-user-guide/~/revisions/KYLAXrLpht5xTVDD1Y1G/appendix/appendix-3-how-to-use-singularity-container)
* [Podman build 문서](https://docs.podman.io/en/latest/markdown/podman-build.1.html)
* [Apptainer GPU 지원](https://apptainer.org/docs/user/main/gpu.html)
* [NVIDIA Enroot](https://github.com/NVIDIA/enroot)
* [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis)
{% endhint %}



### 3. 아키텍처와 이미지 호환성

컨테이너 이미지의 CPU 아키텍처는 계산 노드와 같아야 합니다.

<table><thead><tr><th width="98.94195556640625">노드 구분</th><th width="169.82598876953125">CPU 아키텍처</th><th width="87.53619384765625">GPU</th><th>이미지 파일 예시</th></tr></thead><tbody><tr><td>GPU 노드</td><td><code>aarch64/arm64</code><br><code>(Grace CPU)</code></td><td>H100</td><td><p><code>pytorch-aarch64.sif</code>, </p><p><code>pytorch-aarch64.sqsh</code></p></td></tr><tr><td>CPU 노드</td><td><code>x86_64</code><br><code>(AMD EPYC CPU)</code></td><td>없음</td><td><code>ubuntu-x86_64.sif</code></td></tr></tbody></table>

1\) 현재 노드 아키텍처 확인

```bash
$ uname -m
aarch64
```



2\) Podman 이미지 아키텍처 확인

<pre class="language-bash"><code class="lang-bash"><strong>$ podman image inspect myhub.ksc.re.kr/wjnadia/pytorch:25.03-py3-aarch64 --format '{{.Architecture}}'
</strong>arm64
</code></pre>

`exec format error`가 발생하면 이미지와 계산 노드의 아키텍처를 먼저 비교합니다.&#x20;



&#x20;3\) apptainer/singularity 이미지 아키텍처 확인

```bash
$ singularity inspect --labels pytorch-25.03-py3-aarch64.sif | grep build-arch
org.label-schema.build-arch: arm64
```

`exec format error`가 발생하면 이미지와 계산 노드의 아키텍처를 먼저 비교합니다.&#x20;

Enroot/Pyxis는 `.sqsh`를 실행하기 위해 컨테이너를 시작하지 않고 아키텍처를 완전히 판별하기 어렵기 때문에 파일명에 `aarch64` 또는 `x86_64`를 넣는 것을 권장합니다.



### 4. 이미지 준비

#### 4.1 Podman 이미지  빌드

[뉴론 컨테이너 활용 가이드](appendix-12-how-to-use-containers.md#id-2.-podman)와 동일하게 Podman을 이용해 Dockerfile 기반 이미지를 빌드할 수 있습니다.&#x20;

```bash
# Podman 사용 환경 설정을 위해서는 먼저 사용자 홈 디렉터리에 .usepodman 이라는 파일을 생성해야 합니다. 
# 한 번만 생성하면 되고 로그아웃 후 다시 로그인 하면 바로 적용 됩니다.
$ cd ~                # 사용자홈 디렉터리(/home01/[ID])로 이동             
$ touch .usepodman    
$ ls -la .usepodman
-rw-r--r-- 1 test test 0  3월  3 11:19 .usepodman
$ exit                # 로그아웃 후 다시 로그인
```

<pre class="language-bash"><code class="lang-bash">$ cat Dockerfile.pytorch   # 이미지 빌드를 위한 Dockerfile 생성
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="kisti-container tutorial image"
WORKDIR /workspace

RUN umask 022
RUN python3 -m pip install --no-cache-dir deepspeed==0.18.9
RUN chmod -R a+rX /usr/local /opt

CMD ["python3", "--version"]

# 이미지 빌드 및 아키텍처 확인
$ podman build -f Dockerfile.pytorch -t localhost/kisti-pytorch:tutorial .
$ podman image inspect localhost/kisti-pytorch:tutorial --format 'image={{.RepoTags}} arch={{.Architecture}} os={{.Os}}'
<strong>  image=[localhost/kisti-pytorch:tutorial] arch=arm64 os=linux
</strong></code></pre>

레지스트리에서 직접 가져오는 예시:

```bash
$ podman pull nvcr.io/nvidia/pytorch:25.03-py3
$ podman images
```

#### 4.2 Singularity/Enroot 이미지로 변환

```bash
## Enroot
enroot import -o pytorch-aarch64.sqsh podman://localhost/kisti-pytorch:tutorial

## Singularity
podman save localhost/kisti-pytorch:tutorial -o pytorch-aarch64.tar
singularity build --fakeroot pytorch-aarch64.sif \
  docker-archive://pytorch-aarch64.tar
```

빌드된 이미지는 홈/스크래치 디렉터리 또는 [KISTI 내부 레지스트리](appendix-12-how-to-use-containers.md#id-3)에 보관해야 합니다.



컨테이너 이미지 배포 위치

```bash
/apps/common/kisti-container/images
```

배치 파일의 `IMAGE`, 파티션, 시간, CPU/GPU 수 등을 확인한 뒤 제출합니다.

###

### 5. kisti-container 실행 환경&#x20;

```bash
KISTI_CONTAINER=/apps/common/kisti-container/bin/kisti-container

$ KISTI_CONTAINER --version
$ KISTI_CONTAINER --help
$ srun --help | grep -- --container-image
```

마지막 명령에 `--container-image`가 표시되면 Pyxis가 Slurm에 등록된 상태입니다.

사이트 기본 설정은 다음 파일에 존재합니다.

```
/apps/common/kisti-container/conf/kisti-container.conf
```

사용자 별 설정은 필요할 때만 다음 파일에 작성합니다.

```
$HOME/.config/kisti-container.conf
```

별도 설정 파일을 지정하려면 다음 환경 변수를 사용합니다.

```bash
export KISTI_CONTAINER_CONFIG=/absolute/path/kisti-container.conf
```



### 6. kisti-container 기본 명령 형식

```bash
kisti-container [Wrapper 옵션] IMAGE COMMAND [ARG ...]
```

주요 옵션:

<table><thead><tr><th width="178.07244873046875">옵션</th><th>값</th><th>설명</th></tr></thead><tbody><tr><td><code>--runtime</code></td><td><code>singularity</code>, <code>apptainer</code>, <code>enroot</code>, <code>pyxis</code></td><td>실행 백엔드</td></tr><tr><td><code>--workload</code></td><td><code>generic</code>, <code>pytorch-dist*</code></td><td>워크로드 별 검증 규칙</td></tr><tr><td><code>--launcher</code></td><td><code>srun-native</code>, <code>torchrun</code></td><td>분산 프로세스 생성 방식</td></tr><tr><td><code>--local-processes</code></td><td>양의 정수</td><td>노드별 torchrun worker 수</td></tr><tr><td><code>--mpi</code></td><td><code>none</code>, <code>cray</code>, <code>openmpi</code>, <code>auto</code></td><td>MPI 구성</td></tr><tr><td><code>--nccl</code></td><td><code>none</code>, <code>native</code>, <code>ofi</code>, <code>auto</code></td><td>NCCL 전송 구성</td></tr><tr><td><code>--checkpoint-dir</code></td><td>절대 경로</td><td>쓰기 가능한 체크포인트 경로</td></tr><tr><td><code>-B</code>, <code>--bind</code></td><td><code>SRC:DST:MODE</code></td><td>추가 마운트</td></tr><tr><td><code>--env</code></td><td><code>NAME=VALUE</code></td><td>컨테이너에 전달할 환경 변수</td></tr><tr><td><code>--diagnose</code></td><td>값 없음</td><td>rank 0에서 결정된 설정 출력</td></tr><tr><td><code>--doctor</code></td><td>값 없음</td><td>GPU, CXI, OFI 라이브러리 사전 점검</td></tr><tr><td><code>--dry-run</code></td><td>값 없음</td><td>실행하지 않고 최종 명령 출력</td></tr></tbody></table>

\*pytorch-dist : DDP, FDSP, DeepSpeed, Nemo/Megatron 등 기반의 분산학습 워크로드&#x20;



### 7. 배치 작업 실행 방식

#### 7.1 Singularity, Apptainer, Enroot

컨테이너 백엔드에서 `kisti-container`는 Slurm task 안에서 실행되는 Wrapper입니다. 따라서 배치 파일에 `srun`을 사용합니다.

```bash
srun --mpi=none \
  /apps/common/kisti-container/bin/kisti-container \
    --runtime singularity \
    IMAGE.sqsh python3 train.py
```

#### 7.2 Pyxis

Pyxis에서 `kisti-container`는 `srun` 명령을 생성하는 프론트엔드입니다. 배치 스크립트에서 직접 호출합니다.

```bash
/apps/common/kisti-container/bin/kisti-container \
  --runtime pyxis \
  IMAGE.sqsh python3 train.py
```

다음 형식은 사용하지 않습니다.

```bash
srun kisti-container --runtime pyxis ...
```

이 형식은 이미 생성된 Slurm step 안에서 Pyxis용 새 `srun`을 만들려고 하므로 Wrapper가 오류로 중단합니다.



### 8. 작업 유형 별 예제

예제  파일 배포 위치:

```
/apps/common/kisti-container/examples
```

사용자는 예제를 자신의 작업 공간으로 복사합니다.

```bash
mkdir -p /scratch/$USER/kisti-container
cp -a /apps/common/kisti-container/examples/. \
  /scratch/$USER/kisti-container/
cd /scratch/$USER/kisti-container
```

배치 파일의 `IMAGE`, 파티션, 시간, CPU/GPU 수 등을 확인한 뒤 제출합니다.&#x20;

#### 8.1 일반  CPU/GPU 작업

&#x20;일반 CPU/GPU 작업에서 kisti-container의 옵션은 `--workload generic` 을 사용합니다. 단일 GPU에서는 `--nccl native` 또는 `--nccl none`을 선택할 수 있습니다.

```bash
## (pyxis) 일반 GPU 작업 스크립트 예제
## /apps/common/kisti-container/examples/02-gpu-smoke/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=gpu-pyxis
#SBATCH --partition=gpu
#SBATCH --comment="field=misc;appl=pytorch"
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT=/scratch/$USER/kisti-container
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE=$ROOT/images/pytorch-25.03-py3-aarch64.sqsh
PROGRAM=$ROOT/examples/02-gpu-smoke/gpu_smoke.py

module purge
module load enroot/4.2.0

"$KC" \
  --runtime pyxis --workload generic \
  "$IMAGE" python3 "$PROGRAM"
```

작업제출:

<pre class="language-bash"><code class="lang-bash"><strong>$ sbatch run-pyxis.sbatch
</strong></code></pre>

결과파일:

```bash
$ cat gpu-pyxis-20590.out
GPU_SMOKE_PASS host=gpu0014 gpu=NVIDIA GH200 120GB shape=(2048, 2048) mean=0.011777
```

#### 8.2  MPI 작업

&#x20;MPI 작업은 `--workload generic`을 사용하며, MPI 유형에 따른 관련 옵션은 아래와 같습니다.&#x20;

|    구분   |       srun 옵션      | kisti-container 옵션 |
| :-----: | :----------------: | :----------------: |
| CrayMPI | --mpi=cray\_shasta |     --mpi cray     |
| OpenMPI |     --mpi=pmix     |    --mpi openmpi   |

```bash
## (Singularity) MPI OMB 작업 스크립트 예제
## /apps/common/kisti-container/examples/03-mpi-omb/run-gh200-singularity.sbatch

#!/bin/bash
#SBATCH --job-name=gh200-omb-sing
#SBATCH --comment="field=misc;appl=misc"
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE="/scratch/$USER/kisti-container/images/omb-7.5.2-aarch64.sif"

OMB="/opt/omb/cray-mpich/7.5.2/pt2pt"

module purge
module load cray-mpich/9.0.1
module load libfabric/2.3.1
module load singularity/4.5.0

echo "===== [CrayMPI] GH200 OMB SINGULARITY LATENCY ====="
srun \
  --mpi=cray_shasta --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=1 \
  --distribution=block:block --cpu-bind=cores \
  "$KC" \
    --runtime singularity --workload generic --mpi cray \
    --provider cxi --diagnose \
    "$IMAGE" "$OMB/osu_multi_lat"

echo "===== [CrayMPI] GH200 OMB SINGULARITY BANDWIDTH ====="
srun \
  --mpi=cray_shasta --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=1 \
  --distribution=block:block --cpu-bind=cores \
  "$KC" \
    --runtime singularity --workload generic --mpi cray \
    --provider cxi --diagnose \
    "$IMAGE" "$OMB/osu_mbw_mr"

OMB="/opt/omb/openmpi/7.5.2/pt2pt"

module purge
module load openmpi/5.0.10
module load libfabric/2.3.1
module load singularity/4.5.0

echo "===== [OpenMPI] GH200 OMB SINGULARITY LATENCY ====="
srun \
  --mpi=pmix --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=1 \
  --distribution=block:block --cpu-bind=cores \
  "$KC" \
    --runtime singularity --workload generic --mpi openmpi \
    --provider cxi --diagnose \
    "$IMAGE" "$OMB/osu_multi_lat"

echo "===== [OpenMPI] GH200 OMB SINGULARITY BANDWIDTH ====="
srun \
  --mpi=pmix --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=1 \
  --distribution=block:block --cpu-bind=cores \
  "$KC" \
    --runtime singularity --workload generic --mpi openmpi \
    --provider cxi --diagnose \
    "$IMAGE" "$OMB/osu_mbw_mr"
```

작업제출:

<pre class="language-bash"><code class="lang-bash"><strong>$ sbatch run-gh200-singularity.sbatch
</strong></code></pre>

결과파일:

<pre class="language-bash"><code class="lang-bash"><strong>$ cat gh200-omb-sing-20548.out
</strong>===== [CrayMPI] GH200 OMB SINGULARITY BANDWIDTH =====
# OSU MPI Multiple Bandwidth / Message Rate Test v7.5.2
# [ pairs: 4 ] [ window size: 64 ]
# Datatype: MPI_CHAR.
# Size                  MB/s        Messages/s
1                       8.36        8361985.11
2                      16.57        8284756.13
4                      33.39        8348405.71
8                      66.63        8328298.94
16                    132.68        8292748.94
32                    264.68        8271214.41
64                    539.18        8424707.26
128                  1093.55        8543336.87
256                  2361.93        9226298.30
512                  4762.16        9301087.75
1024                 9532.32        9308903.88
2048                19069.25        9311156.29
4096                38041.09        9287376.18
8192                75100.02        9167483.30
16384              138962.42        8481592.79
32768              151199.58        4614244.85
65536              169302.95        2583358.08
131072             180156.90        1374488.08
262144             185277.89         706779.06
524288             190569.95         363483.34
1048576            192158.82         183256.93
2097152            193140.30          92096.47
4194304            193711.29          46184.37
===== [OpenMPI] GH200 OMB SINGULARITY BANDWIDTH =====
# OSU MPI Multiple Bandwidth / Message Rate Test v7.5.2
# [ pairs: 4 ] [ window size: 64 ]
# Datatype: MPI_CHAR.
# Size                  MB/s        Messages/s
1                       5.71        5711864.23
2                      11.63        5816973.47
4                      23.18        5793892.90
8                      46.28        5785194.05
16                     93.43        5839613.20
32                    185.89        5809137.39
64                    376.66        5885318.59
128                   746.86        5834845.61
256                  1505.84        5882194.13
512                  3043.19        5943724.08
1024                 6050.44        5908636.67
2048                12102.72        5909531.76
4096                23954.69        5848312.59
8192                48628.59        5936107.31
16384               95052.14        5801522.22
32768              148685.54        4537522.54
65536              168811.67        2575861.71
131072             179137.31        1366709.25
262144             184658.59         704416.62
524288             190285.40         362940.59
1048576            191985.38         183091.53
2097152            193126.48          92089.88
4194304            193713.41          46184.88
</code></pre>

#### 8.3 PyTorch DDP

DDP 2노드 8GPU 예제는 노드당 4개의 Slurm rank를 생성합니다. `--nccl auto`는 2노드에서 사이트의 AWS OFI NCCL, Cray Libfabric, CXI 프로파일을 선택합니다.<br>

{% hint style="info" %}
**DDP :  Distributed Data Parallel**

&#x20;**Pytorch**의 **기본 분산 학습 방식**으로 각 GPU에 모델 전체를 복제하고, 서로 다른 데이터로 계산한 그래디언트를 동기화해 동일한 모델 상태를 유지합니다.
{% endhint %}

```bash
## (Pyxis) Pytorch DDP 작업 스크립트 예제
## /apps/common/kisti-container/examples/04-pytorch-ddp/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=ddp-pyxis
#SBATCH --partition=gpu
#SBATCH --comment="field=sciai;appl=pytorch-ddp"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT=/scratch/$USER/kisti-container
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE=$ROOT/images/pytorch-25.03-py3-aarch64.sqsh
PROGRAM=$ROOT/examples/04-pytorch-ddp/ddp_smoke.py

CHECKPOINT_DIR=$SLURM_SUBMIT_DIR/checkpoints/ddp-pyxis-$SLURM_JOB_ID
mkdir -p "$CHECKPOINT_DIR"

module purge
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
module load enroot/4.2.0

"$KC" \
  --runtime pyxis --workload pytorch-dist \
  --launcher srun-native --nccl ofi \
  --checkpoint-dir "$CHECKPOINT_DIR" --diagnose \
  "$IMAGE" python3 "$PROGRAM"
```



작업제출:

<pre class="language-bash"><code class="lang-bash"><strong>$ sbatch run-pyxis.sbatch
</strong></code></pre>



결과 파일:

```bash
$ cat ddp-pyxis-20445.out
--[중략]--
DDP_RANK_PASS rank=7 host=gpu0015 gpu=3
DDP_RANK_PASS rank=0 host=gpu0014 gpu=0
DDP_RANK_PASS rank=3 host=gpu0014 gpu=3
DDP_RANK_PASS rank=4 host=gpu0015 gpu=0
DDP_RANK_PASS rank=2 host=gpu0014 gpu=2
DDP_RANK_PASS rank=1 host=gpu0014 gpu=1
DDP_RANK_PASS rank=6 host=gpu0015 gpu=2
DDP_RANK_PASS rank=5 host=gpu0015 gpu=1

$ cat checkpoints/ddp-pyxis-20445/ddp-smoke-result.json
{
  "status": "PASS",
  "world_size": 8,
  "collective_sum": 36.0,
  "nodes": [
    "gpu0014",
    "gpu0015"
  ],
  "ranks": [
    {
      "rank": 0,
      "local_rank": 0,
      "host": "gpu0014",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 1,
      "local_rank": 1,
      "host": "gpu0014",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 2,
      "local_rank": 2,
      "host": "gpu0014",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 3,
      "local_rank": 3,
      "host": "gpu0014",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 4,
      "local_rank": 0,
      "host": "gpu0015",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 5,
      "local_rank": 1,
      "host": "gpu0015",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 6,
      "local_rank": 2,
      "host": "gpu0015",
      "gpu": "NVIDIA GH200 120GB"
    },
    {
      "rank": 7,
      "local_rank": 3,
      "host": "gpu0015",
      "gpu": "NVIDIA GH200 120GB"
    }
  ],
  "nccl_net": "Libfabric",
  "fi_provider": "cxi"
}
```

#### 8.3 PyTorch FSDP2

FSDP2 예제는 노드당 Slurm task 1개를 만들고, 각 task 안에서 torchrun worker 4개를 실행합니다. 전체 `WORLD_SIZE`는 8입니다.

{% hint style="info" %}
**FSDP2**: **Fully Sharded Data Parallel, version 2**\
PyTorch의 **완전 분할 데이터 병렬 학습 방식 2세대**입니다. 모델 파라미터, 그래디언트, 옵티마이저 상태를 GPU 간에 분산해 GPU당 메모리 부담을 줄입니다.
{% endhint %}

```bash
## (Pyxis) Pytorch FSDP2 작업 스크립트 예제
## /apps/common/kisti-container/examples/05-pytorch-fsdp2/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=fsdp-pyxis
#SBATCH --partition=gpu
#SBATCH --comment="field=misc;appl=pytorch-fsdp"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE="$ROOT/images/pytorch-25.03-py3-aarch64.sqsh"
TRAIN_SCRIPT="$ROOT/examples/05-pytorch-fsdp2/fsdp2_train_smoke.py"

module purge
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
module load enroot/4.2.0

"$KC" \
  --runtime pyxis --workload pytorch-dist \
  --launcher srun-native --nccl ofi \
  --diagnose \
  "$IMAGE" "$TRAIN_SCRIPT"
```

작업제출:

```bash
$ sbatch run-pyxis.sbatch
```

결과 파일:\
성공 여부는 종료 코드와 `RESULT ... correctness=True` 로그로 확인합니다. 이 프로그램은 합성 데이터 기반 통신·학습 smoke test 용도이며 실제 모델 훈련의 실행을 목적으로 하지 않습니다.

```bash
$ cat fsdp-pyxis-20459.out
--[중략]--
step=1/25 loss=1.136039 ms=401.150
step=2/25 loss=0.601430 ms=28.652
step=3/25 loss=0.480413 ms=24.631
step=4/25 loss=0.272872 ms=22.903
--[중략]--
step=22/25 loss=54.140591 ms=22.411
step=23/25 loss=35.631592 ms=22.266
step=24/25 loss=24.416914 ms=22.344
step=25/25 loss=54.308086 ms=22.022
RESULT backend=nccl workload=fsdp2 world_size=8 precision=bf16 mean_step_ms=22.373 median_step_ms=22.333 final_check=36.0 expected=36.0 correctness=True
```



#### 8.4 NeMo/Megatron&#x20;

TP·DP 등 모델 병렬과 데이터 병렬을 조합하는 대규모 모델 학습을 위한 NeMo 예제는 GH200 2노드에서 노드 내부 TP=4, 노드 간 DP=2를 확인합니다.

컨테이너에는 NeMo, Megatron Core, Transformer Engine 및 호환 PyTorch/CUDA가 설치되어 있어야 합니다. 제공 smoke test의 검증 환경은 NeMo 2.3.0rc5, Megatron Core 0.12.0rc4, Transformer Engine 2.2.0 개발 버전 계열입니다.

{% hint style="info" %}
**NeMo**: **Neural Modules**에서 유래한 이름\
NVIDIA의 AI 모델 개발·학습 프레임워크입니다. 현재 예제의 **NeMo/Megatron**은 NeMo와 Megatron의 Tensor Parallel 등 모델 병렬과 데이터 병렬을 조합한 분산 학습 기능을 활용하는 구성을 뜻합니다.
{% endhint %}

```bash
## (Pyxis) NeMO Megatron 작업 스크립트 예제
## /apps/common/kisti-container/examples/06-nemo-megatron/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=nemo-pyxis
#SBATCH --partition=gpu
#SBATCH --comment="field=misc;appl=megatron-lm"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE="$ROOT/images/nemo-25.04.00-aarch64.sqsh"
TRAIN_SCRIPT="$ROOT/examples/06-nemo-megatron/nemo_megatron_tp4_dp2_smoke.py"

CHECKPOINT_DIR=${SLURM_SUBMIT_DIR}/checkpoints/nemo-pyxis-${SLURM_JOB_ID}
mkdir -p "$CHECKPOINT_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

module purge
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
module load enroot/4.2.0

"$KC" \
  --runtime pyxis --workload pytorch-dist \
  --launcher torchrun --local-processes 4 --nccl ofi \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --env "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS" \
  --diagnose \
  "$IMAGE" "$TRAIN_SCRIPT" --steps 5 --tp-size 4 --expected-dp-size 2 \
  --checkpoint-dir "$CHECKPOINT_DIR"
```

작업제출:

```bash
sbatch run-pyxis.sbatch
```

결과 파일:

```bash
$ cat nemo-pyxis-20467.out
--[중략]--
NEMO_MCORE_TPDP_RESULT status=PASS world_size=8 tp_size=4 dp_size=2 steps=5 last_dp_loss_avg=8.43100882 mean_step_ms=395.784 steady_mean_step_ms=49.037 collective=28.0 expected=28.0
```



#### 8.5 DeepSpeed&#x20;

ZeRO 최적화로 옵티마이저 상태와 그래디언트를 GPU들에 나눠 저장하는 데이터 병렬 학습 예제입니다. GH200 2노드에서 노드당 GPU 4장, 총 8개 rank로 ZeRO-2 학습이 동작하는지 확인합니다.

컨테이너에는 DeepSpeed와 호환 PyTorch/CUDA가 설치되어 있어야 합니다. 제공 예제의 검증 환경은 NGC PyTorch 25.03(PyTorch 2.7.0a0, CUDA 12.8)에 DeepSpeed 0.18.9를 설치한 이미지입니다.&#x20;

{% hint style="info" %}
**DeepSpeed**: Microsoft가 개발한 대규모 모델 학습 최적화 라이브러리입니다. 현재 예제의 ZeRO-2(Zero Redundancy Optimizer stage 2)는 각 GPU가 모델 전체를 가지되, 옵티마이저 상태와 그래디언트는 GPU 수만큼 나눠 저장해 GPU당 메모리 사용량을 줄이는 방식입니다.
{% endhint %}

```bash
## (Pyxis) DeepSpeed 작업 스크립트 예제
## /apps/common/kisti-container/examples/07-pytorch-deepspeed/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=ds-pyxis
#SBATCH --partition=gpu
#SBATCH --comment="field=misc;appl=deepspeed"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE=$ROOT/images/pytorch-25.03-py3-aarch64.sqsh
TRAIN_SCRIPT=$ROOT/examples/07-pytorch-deepspeed/train_deepspeed.py

module purge
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
module load enroot/4.2.0

USER=`id -un`
export CACHE_ROOT=/scratch/$USER/.cache/job-$SLURM_JOB_ID
mkdir -p $CACHE_ROOT

export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export XDG_CACHE_HOME=$CACHE_ROOT/xdg

"${KC}" \
    --runtime pyxis --workload pytorch-dist \
    --launcher torchrun --local-processes 4 --nccl ofi \
    --diagnose \
    "$IMAGE" "$TRAIN_SCRIPT" --deepspeed --deepspeed_config ${SLURM_SUBMIT_DIR}/ds_config.json --epochs 2
```

작업제출:

```bash
sbatch run-pyxis.sbatch
```

결과 파일:

```bash
$ cat ds-pyxis-23707.out
--[중략]--
  epoch 0 step    0  loss=10.5625
  epoch 0 step   20  loss=10.5000
  epoch 0 step   40  loss=10.4375
  epoch 0 step   60  loss=10.4375
  epoch 0 step   80  loss=10.4375
  epoch 0 step  100  loss=10.3750
  epoch 0 step  120  loss=10.3750
  epoch 0 step  140  loss=10.4375
  epoch 0 done  avg_loss=10.4387  time=8.5s  steps=157
  epoch 1 step    0  loss=10.4375
  epoch 1 step   20  loss=10.3750
  epoch 1 step   40  loss=10.3750
  epoch 1 step   60  loss=10.3750
  epoch 1 step   80  loss=10.3750
  epoch 1 step  100  loss=10.3750
  epoch 1 step  120  loss=10.3750
  epoch 1 step  140  loss=10.3750
  epoch 1 done  avg_loss=10.3838  time=8.7s  steps=157
학습 완료
```



### 9. 체크포인트  및 마운트

체크포인트에는 계산 노드 모두에서 접근 가능한 절대 경로를 사용합니다.

```bash
CHECKPOINT_DIR=/scratch/$USER/checkpoints/run01
mkdir -p "$CHECKPOINT_DIR"
```

`CHECKPOINT_DIR`를 생략한 예제는 다음 형식으로 작업 별 경로를 만듭니다.

```
CHECKPOINT_DIR=${SLURM_SUBMIT_DIR}/checkpoints/WORKLOAD-RUNTIME-${SLURM_JOB_ID}
```

기존 체크포인트에서 재시작 할 때는 동일한 `CHECKPOINT_DIR`를 명시합니다.

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
    --runtime enroot --nccl ofi --doctor IMAGE.sqsh
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
| `libfuse3.so.4` 없음                        | `ldd $(command -v squashfuse)`             | 관리자에게 SquashFUSE/FUSE3 설치 확인 요청              |
| zstd `.sqsh` 마운트 실패                       | `ldd squashfuse`의 `libzstd.so.1`           | zstd 지원 SquashFUSE 사용                        |
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



### 12. 참고 자료

#### 12.1 kisti-container 실행 워크플로우

사용자 옵션과 Slurm 환경을 해석하여 컨테이너 실행 명령을 구성하는 Bash Wrapper

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>



#### 12.2 kisti-container 런타임별 실행과 통신 환경

Slurm 작업에서 직접 런타임은 task 안에서 실행하고, Pyxis는 Wrapper가 srun을 생성

<figure><img src="../.gitbook/assets/image (21).png" alt=""><figcaption></figcaption></figure>

#### 12.3 한강 시스템 컨테이너 OMB 통신 성능 비교<br>

* OMB 7.5.2
* Native / Singularity
* GH200 노드 : Slingshot NIC 4ea
* AMD CPU 노드 : Slingshot NIC 1ea
* GH200 노드 대역폭(2노드 · 4 Pair · 4 MiB)<br>

<figure><img src="../.gitbook/assets/image (18).png" alt=""><figcaption></figcaption></figure>

