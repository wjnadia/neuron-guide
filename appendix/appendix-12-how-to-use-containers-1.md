---
description: 수정 중
---

# (6호기 한강) 컨테이너 활용 가이드

### 1. 개요

한강 시스템은 HPC 응용 및 AI 학습/추론 등에서 복잡한 소프트웨어 의존성을 해결하고 다양한 시스템에서 일관된 작업 환경을  유지할 수 있도록 최적화된 컨테이너 활용 환경을 제공합니다.        &#x20;

사용자는 자신의 작업 단계(빌드, 실행)에 맞춰 적절한 도구를 선택하여 사용할 수 있습니다.

Podman은 OCI 이미지를 빌드하고 관리하는 도구이고, 계산  작업을 실행하기 위해서는 Singularity, Apptainer, Enroot과 같은 컨테이너 런타임을 선택적으로 사용할 수 있습니다.

또한, 병렬 연산 및 분산  학습 작업에서 고성능 네트워크(Slingshot) 기반의 MPI 및 NCCL 환경 구성을 지원하는 컨테이너 런타임 Wrapper 프로그램(kisti-container)을 제공하고 있습니다.   &#x20;

{% hint style="info" %}
* **Podman** : 일반 사용자 권한 기반의 이미지 빌드 및 관리 도구입니다. Docker를 대체하여 이미지를 생성할 때 사용합니다.
* **Enroot** : NVIDIA에서 개발한 HPC 전용 런타임입니다. Nvidia GPU에 최적화 되어 있으며, Pyxis 플러그인을 통해 Slurm 스케줄러와 유기적으로 연동됩니다.
* **Singularity/Apptainer** : HPC 환경에  최적화된 컨테이너 도구로 HPC 시스템에서 보편적으로 활용되고 있습니다.  Apptainer는 Singularity에서 분리되어 발전한 오픈소스 컨테이너 런타임으로, 대부분의 이미지 형식과 사용 방식이 호환됩니다.
{% endhint %}



### 2. 컨테이너 도구 선택 가이드

사용자의 목적 및 실행 노드 특성에 따라 최적의 도구를 선택할 수 있습니다.

<table data-header-hidden><thead><tr><th width="104.800048828125" align="center"></th><th align="center"></th><th align="center"></th><th align="center"></th><th></th></tr></thead><tbody><tr><td align="center"><strong>구분</strong></td><td align="center"><strong>Podman</strong><br><strong>(빌드/관리)</strong></td><td align="center"><strong>Enroot</strong><br><strong>(GPU 실행)</strong></td><td align="center"><p><strong>Singularity</strong></p><p><strong>/Apptainer</strong><br><strong>(범용 실행)</strong></p></td><td><strong>kisti-container</strong></td></tr><tr><td align="center">주요 역할</td><td align="center"><p>이미지 생성 </p><p>및 커스터마이징</p></td><td align="center"><p>GPU 연산 가속 </p><p>(권장)</p></td><td align="center">시스템 간<br>호환성 유지</td><td>컨테이너 런타임 Wrapper 프로그램</td></tr><tr><td align="center">저장 형식</td><td align="center"><p>OCI 레이어 </p><p>(디렉터리)</p></td><td align="center"><p><code>.sqsh</code> </p><p>(단일 압축 파일)</p></td><td align="center"><p><code>.sif</code> </p><p>(단일 이미지 파일)</p></td><td>컨테이너 런타임 별 이미지 형식</td></tr><tr><td align="center">특징</td><td align="center">Docker 명령어 호환</td><td align="center">빠른 로딩, <br>Nvidia GPU 최적화<br>Slurm 스케줄러 연동 플러그인 제공(Pyxis)</td><td align="center">기존  누리온/뉴론 컨테이너 환경 유지</td><td>컨테이너에서 계산 노드에 최적화된 MPI, NCCL 환경 구성 지원</td></tr></tbody></table>



{% hint style="info" %}
**도구 별 사용법 참고 자료**

* [뉴론 컨테이너 활용 가이드](https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-12-how-to-use-containers) (Podman, Singularity, Enroot, Pyxis)  &#x20;
* [뉴론 Singularity 컨테이너](https://docs-ksc.gitbook.io/neuron-user-guide/~/revisions/KYLAXrLpht5xTVDD1Y1G/appendix/appendix-3-how-to-use-singularity-container)
* [Podman build 문서](https://docs.podman.io/en/latest/markdown/podman-build.1.html)
* [Apptainer GPU 지원](https://apptainer.org/docs/user/main/gpu.html)
* [NVIDIA Enroot](https://github.com/NVIDIA/enroot)
* [NVIDIA Pyxis](https://github.com/NVIDIA/pyxis)
{% endhint %}



### 3. 이미지 빌드 (Podman)

로그인 노드 또는 계산 노드에서 Podman을 사용하여 컨테이너 이미지를 빌드합니다.&#x20;

{% hint style="info" %}
Singularity 컨테이너 이미지를 직접 빌드 하기 위해서는 뉴론 컨테이너 활용 가이드의  [**가. 컨테이너 이미지 빌드하기**](https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-3-how-to-use-singularity-container#id) 를 참조하시기 바랍니다.&#x20;
{% endhint %}

#### 3.1 Podman 사용 설정

Podman 사용 환경 설정을 위해서는 먼저 사용자 홈 디렉터리에 .usepodman 이라는 파일을 생성해야 합니다. 한 번만 생성하면 되고 로그아웃 후 다시 로그인 하면 바로 적용 됩니다.

```bash
$ cd ~                # 사용자홈 디렉터리(/home01/[ID])로 이동             
$ touch .usepodman    
$ ls -la .usepodman
-rw-r--r-- 1 test test 0  3월  3 11:19 .usepodman
$ exit                # 로그아웃 후 다시 로그인
```

#### 3.2 이미지 빌드하기

사용자 소스 코드나 특정 라이브러리를 포함한 커스텀 이미지를 생성합니다.

```bash
# 현재 디렉터리(.)의 Dockerfile.pytorch로 'pytorch:deepspeed-aarch64' 이미지 빌드
$ podman build -t pytorch:deepspeed-aarch64 -f Dockerfile.pytorch .
$ podman images
REPOSITORY                       TAG                IMAGE ID      CREATED         SIZE
localhost/pytorch                deepspeed-aarch64  5e7a4783117b  21 minutes ago  22.1 GB
$ podman image inspect localhost/pytorch:deepspeed-aarch64 --format 'image={{.RepoTags}} arch={{.Architecture}} os={{.Os}}'
image=[localhost/pytorch:deepspeed-aarch64] arch=arm64 os=linux
```

```bash
[Dockerfile 예시  : Dockerfile.pytorch ]
# 베이스 이미지 지정 (NVIDIA GPU Cloud 제공 이미지)
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.03-py3
FROM ${BASE_IMAGE}

# 메타 데이터 설정 (`docker image inspect` 등으로 이미지의 용도 확인)
LABEL org.opencontainers.image.title="kisti-container tutorial image"

# 이후 명령의 기본 작업 디렉터리 및 컨테이너 실행 기본 위치 설정
WORKDIR /workspace

# 추가 시스템 패키지 설치 (필요시)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
    
# Python 라이브러리 추가 설치 (pip 활용)
RUN python3 -m pip install --no-cache-dir deepspeed==0.18.9

# 컨테이너 실행 명령을 별도로 지정하지 않았을 때 실행할 기본 명령어
CMD ["python3", "--version"]
```



레지스트리에서 직접 가져오는 예시:

```bash
$ podman pull nvcr.io/nvidia/pytorch:25.03-py3
Trying to pull nvcr.io/nvidia/pytorch:25.03-py3...
Getting image source signatures
Copying blob 69fbe38eb239 done   |
Copying blob 5b17151e9710 done   |
Copying blob 58f537c310dd done   |
--[중략]--
Copying config b9d9faba96 done   |
Writing manifest to image destination
b9d9faba9659ded5e9c72b694879edb79ef4a6e8786d367a2c22f83837536253

$ podman images
REPOSITORY                                 TAG                     IMAGE ID      CREATED        SIZE
nvcr.io/nvidia/pytorch                     25.03-py3               b9d9faba9659  19 months ago  22 GB
```



{% hint style="info" %}
**이미지 빌드 명령어**
{% endhint %}

```
$ podman build [옵션] -t [이미지명]:[태그] [Dockerfile경로]
-t, --tag: 이미지 이름과 버전(태그) 지정(예: my_env:v1.0)
-f, --file: 기본 파일명(Dockerfile)이 아닌 다른 이름의 파일을 사용할 때 지정
--no-cache: 캐시를 사용하지 않고 모든 단계를 새로 빌드(라이브러리 업데이트 시 유용)
-v, --volume: 빌드 과정 중 호스트 디렉토리를 마운트해야 할 때 사용

# 현재 디렉터리의 Dockerfile을 사용하여 빌드하기
$ podman build -t my_pytorch:deepspeed-aarch64 .
```



{% hint style="info" %}
**이미지 관리 명령어 요약**
{% endhint %}

<table data-header-hidden><thead><tr><th width="111.66668701171875" align="center"></th><th width="235.60009765625"></th><th></th></tr></thead><tbody><tr><td align="center"><strong>기능</strong></td><td><strong>명령어</strong></td><td><strong>설명</strong></td></tr><tr><td align="center">목록 확인</td><td><sub>$ podman images</sub></td><td>로컬에 저장된 이미지 리스트 출력</td></tr><tr><td align="center">이미지 삭제</td><td><sub>$ podman rmi [이미지ID]</sub></td><td>불필요한 이미지 제거</td></tr><tr><td align="center">상세 정보</td><td><sub>$ podman inspect [이미지ID]</sub></td><td>이미지 레이어, 환경변수 등 상세 정보 확인</td></tr><tr><td align="center">태그 변경</td><td><sub>$ podman tag [기존이름] [새이름]</sub></td><td>이미지에 새로운 이름/태그 부여</td></tr></tbody></table>



### 3. 이미지 업로드

Podman으로 빌드한 이미지를 **외부 저장소인 Docker Hub** 또는 **KISTI 내부 저장소인 myhub**에 업로드하여 관리 및 공유 할 수 있습니다. 특히, 로그인  또는 계산 노드의 로컬 파일시스템에 저장된 이미지는 영구 보관되지  않기  때문에,  외부 또는 내부 저장소에 이미지를 업로드하는 것을 권장합니다.  &#x20;

#### 3.1  Docker Hub 또는 myhub 로그인

먼저 Podman을 통해 Docker Hub 또는 myhub 계정으로 인증합니다.

**1) Docker Hub**

```
$ podman login docker.io
# Username과 Password(또는 Access Token)를 입력합니다.
```

**2) myhub**

{% code expandable="true" %}
```
$ podman login myhub.ksc.re.kr
# Username(슈퍼컴퓨터 계정 ID)와 Password(CLI secert)을 입력합니다.
```
{% endcode %}

{% hint style="info" %}
myhub 를 사용하기 위해서는 먼저  웹  브라우저에서 [**https://myhub.ksc.re.kr**](https://myhub.ksc.re.kr/)에 슈퍼컴퓨터 계정으로 로그인하여 **사용자 프로젝트**를 생성하고  CLI secret을  복사해 와야 합니다.&#x20;

<i class="fa-linktree">:linktree:</i> 자세한 사용 방법은 [**myhub 사용법**](https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-12-how-to-use-containers#myhub)을 참조하시기 바랍니다.&#x20;
{% endhint %}

#### 3.2 이미지 태그(Tag) 설정

업로드를 위해서는 이미지 이름을 **\[레지스트리주소]/\[Username 또는 프로젝트명]/\[이미지명]:\[버전태그]** 형식으로 지정해야 합니다.

**1) Docker Hub**

```
# 로컬 이미지(pytorch:deepspeed-aarch64)를 Docker Hub 형식으로 태그
$ podman tag localhost/pytorch:deepspeed-aarch64 docker.io/[Username]/pytorch:deepspeed-aarch64
```

**2) myhub**

```
# 로컬 이미지(pytorch:deepspeed-aarch64)를 myhub 형식으로 태그
$ podman tag localhost/pytorch:deepspeed-aarch64 myhub.ksc.re.kr/[프로젝트명]/pytorch:deepspeed-aarch64
```

#### 3.3. 이미지 업로드(Push)

태그가 완료된 이미지를 Docker Hub 또는 myhub로 전송합니다.

**1) Docker Hub**

```
$ podman push docker.io/[Username]/pytorch:deepspeed-aarch64
```

**2) myhub**

```
$ podman push myhub.ksc.re.kr/[프로젝트명]/pytorch:deepspeed-aarch64
```

###

### 4. 이미지 변환&#x20;

Podman으로 준비한 이미지는 컨테이너 실행 도구에 맞는 변환 과정을 거쳐야  합니다.

#### 4.1 Enroot&#x20;

이미지를 SquashFS 포맷으로 변환하여 로딩 속도를 높이고 GPU 연산 효율을 최적화 합니다.

<pre><code># sqsh 이미지 파일 생성(read-only, 계산 노드에서 작업 실행에 적합)
$ module load enroot/1.4.2
<strong>$ enroot import -o pytorch-deepspeed-aarch64.sqsh podman://localhost/pytorch:deepspeed-aarch64
</strong>
# sqsh 이미지 파일로부터 컨테이너 rootfs(unpacked 디렉터리) 생성 
# write 가능하며 이미지 빌드 및 시험에 적합 
$ enroot create -n pytorch-deepspeed-aarch64 pytorch-deepspeed-aarch64.sqsh
</code></pre>



{% hint style="info" %}
**Enroot 이미지 관련 명령어**&#x20;
{% endhint %}

<table data-header-hidden><thead><tr><th width="86.5999755859375" align="center"></th><th width="142.13336181640625" align="center"></th><th></th></tr></thead><tbody><tr><td align="center"><strong>단계</strong></td><td align="center"><strong>작업 내용</strong></td><td><strong>명령어 / 설정 예시</strong></td></tr><tr><td align="center">이미지 가져오기</td><td align="center"><sub>Docker Hub 및 myhub 등  외부 레지스트리에서 직접 가져오기</sub></td><td><p><sub>$ enroot import -o my_pytorch-v1.sqsh docker://[Username]/my_pytorch:v1</sub></p><p></p><p><sub>$ enroot import -o my_pytorch-v1.sqsh docker://myhub.ksc.re.kr/[프로젝트명]/ubuntu:latest</sub></p></td></tr><tr><td align="center">컨테이너 생성</td><td align="center"><sub>rootfs(unpacked 디렉터리) 생성하기</sub></td><td><p><sub>$ enroot create -n my_pytorch-v1</sub><sup><sub>*</sub></sup><sub> my_pytorch-v1.sqsh</sub>     </p><p><sup><sub>*</sub></sup><sub> 컨테이너(rootfs)는 /tmp/enroot_[UID</sub><sup><sub>**</sub></sup><sub>]/data/my_pytorch-v1 디렉터리에 생성됨</sub>   </p><p><sup><sub>**</sub></sup><sub>  id -u 출력값(숫자)</sub> </p></td></tr><tr><td align="center">컨테이너 <br>리스트</td><td align="center"><sub>enroot create 명령어로 생성한 컨테이너 목록 출력</sub></td><td><p><sub>$ enroot list</sub><sup><sub>*</sub></sup> </p><p><sup><sub>*</sub></sup><sub>컨테이너를 생성한 노드에서만 출력됨</sub></p></td></tr><tr><td align="center">컨테이너 삭제</td><td align="center"><sub>enroot create 명령어로 생성한 컨테이너 제거</sub></td><td><p><sub>$ enroot remove my_pytorch-v1</sub><sup><sub>*</sub></sup></p><p><sup><sub>*</sub></sup><sub>컨테이너를 생성한 노드에서만 제거할 수 있음</sub></p></td></tr></tbody></table>

#### 4.2 Singularity/Apptainer

기존 작업 방식 유지 또는 타 시스템과의 이미지 공유가 필요한 경우 활용합니다.

<pre class="language-bash"><code class="lang-bash">## Podman 이미지를 tar로 내보낸 후 .sif 파일로 빌드
$ podman save localhost/pytorch:deepspeed-aarch64 -o pytorch-deepspeed-aarch64.tar

# Singularity
<strong>$ module load singularity/4.5.0
</strong>$ singularity build --fakeroot pytorch-deepspeed-aarch64.tar.sif docker-archive://pytorch-deepspeed-aarch64.tar

# Apptainer
$ module load apptainer/1.4.5
$ apptainer build --fakeroot pytorch-deepspeed-aarch64.tar.sif docker-archive://pytorch-deepspeed-aarch64.tar
</code></pre>

{% hint style="info" %}
**KISTI에서 빌드하여  배포하는 Enroot(.sqsh), Singularity/Apptainer(.sif) 이미지 파일 경로**&#x20;

* _**기본 이미지  :  /apps/common/kisti-container/images**_

```
mlperf-llama31_8b-gh200-aarch64.sqsh  
nemo-25.04.00-aarch64.sqsh  
omb-7.5.2-x86_64.sif           
pytorch-25.03-py3-aarch64.sqsh  
pytorch-25.03-py3-x86_64.sqsh
mlperf-llama31_8b-gh200-aarch64.sif  
nemo-25.04.00-aarch64.sif             
omb-7.5.2-aarch64.sif       
pytorch-25.03-py3-aarch64.sif  
pytorch-25.03-py3-x86_64.sif    
pytorch-25.09-py3-aarch64.sqsh
```



* **Slingshot 고속 네트워크 지원 이미지  : /apps/common/kisti-container/images/slingshot**

```
kisti-pytorch-25.03-gh200-r1-aarch64.sif  
kisti-pytorch-25.03-gh200-r1-aarch64.sqsh
kisti-nemo-25.04-gh200-r1-aarch64.sif
kisti-nemo-25.04-gh200-r1-aarch64.sqsh
```
{% endhint %}



#### 4.3 아키텍처와 이미지 호환성

컨테이너 이미지의 CPU 아키텍처는 로그인/계산 노드와 같아야 합니다.

<table><thead><tr><th width="220.10137939453125" align="center">노드 구분(Slurm 파티션)</th><th width="177.94183349609375" align="center">CPU 아키텍처</th><th width="127.53631591796875" align="center">GPU</th><th align="center">이미지 파일 예시</th></tr></thead><tbody><tr><td align="center">GPU 노드<br>(normal)</td><td align="center"><code>aarch64(arm64)</code><br><code>(Grace CPU)</code></td><td align="center">Hopper GPU</td><td align="center"><p><code>pytorch-aarch64.sif</code>, </p><p><code>pytorch-aarch64.sqsh</code></p></td></tr><tr><td align="center">CPU 노드<br>(normal_cpu, bigsmp)</td><td align="center"><code>x86_64</code><br><code>(AMD EPYC CPU)</code></td><td align="center">없음</td><td align="center"><code>ubuntu-x86_64.sif</code></td></tr></tbody></table>

**1) 현재 노드 아키텍처 확인**

```bash
$ uname -m
aarch64
```

**2) 이미지 아키텍처 확인**

아키텍처 판별이 용이하도록  이미지 태그(Podman) 또는 파일명에 `aarch64(arm64)` 또는 `x86_64(amd64)`를 넣는 것을 권장합니다.

<pre class="language-bash"><code class="lang-bash"><strong>## Podman
</strong>## exec format error가 발생하면 이미지와 계산 노드의 아키텍처를 먼저 비교합니다. 
<strong>$ podman image inspect myhub.ksc.re.kr/wjnadia/pytorch:25.03-py3-aarch64 --format '{{.Architecture}}'
</strong>arm64

## Singularity
$ module load singularity/4.5.0
$ singularity inspect --labels pytorch-25.03-py3-aarch64.sif | grep build-arch
org.label-schema.build-arch: arm64

## Apptainer
$ module load apptainer/1.4.5
$ singularity inspect --labels pytorch-25.03-py3-x86_64.sif | grep build-arch
org.label-schema.build-arch: amd64   

## Enroot/Pyxis
$ module load enroot/4.2.0
$ enroot start pytorch-25.03-py3-aarch64.sif uname -m
aarch64
</code></pre>



### 5. 이미지 실행

생성된 이미지는  Enroot 또는 Singularity/Apptainer 환경에 배포하여 실행할 수 있습니다.

Slurm 배치 작업으로 실행 시 Enroot 의Pyxis 사용을 권장합니다.

#### 5.1 Enroot&#x20;

```bash
# GPU 계산 노드에서 squashFS 이미지 파일을 로드하고 실행
# GPU 가속 연동 옵션 필요 없음(자동 연동됨)
# 공유 이미지 경로 : /apps/common/kisti-container/images
$ enroot start pytorch-25.03-py3-aarch64.sqsh nvidia-smi
$ enroot start pytorch-25.03-py3-aarch64.sif python train.py
```

#### 5.2 Singularity/Apptainer

```bash
# GPU 계산노드에서 Singularity 이미지를 로드하여 실행
# --nv: GPU 가속 연동 옵션 필수
# 공유 이미지 경로 : /apps/common/kisti-container/images
$ singularity exec --nv pytorch-25.03-py3-aarch64.sif nvidia-smi
$ singularity exec --nv pytorch-25.03-py3-aarch64.sif python train.py
```

{% hint style="info" %}
#### [Podman에서 ](#user-content-fn-1)[^1]이미지  실행 방법

```shellscript
# 모든 GPU 사용
$ podman run --rm --device nvidia.com/gpu=all \
    nvcr.io/nvidia/pytorch  nvidia-smi

# 여러 장치 중 특정 GPU(예: 0번)만 사용
$ podman run --rm --device nvidia.com/gpu=0 \
    nvcr.io/nvidia/pytorch  nvidia-smi
```
{% endhint %}



#### 5.3 kisti-container&#x20;

kisti-container는 컨테이너 런타임을 감싸는 실행 래퍼입니다. MPI나 분산 학습 작업을 컨테이너에서 실행할 때 필요한 설정 — Slingshot 네트워크 라이브러리 연결, NCCL 통신 플러그인 지정, 분산 환경변수 주입 — 을 자동으로 처리합니다.

Singularity/Apptainer, Enroot(Pyxis)를 모두 지원하며, 노드의 CPU 아키텍처와 로드된 모듈을 보고 필요한 설정을 자동으로 선택합니다.



**1)  기본 명령 형식**

<pre class="language-bash"><code class="lang-bash">$ export PATH=$PATH:/apps/common/kisti-container/bin
<strong>$ kisti-container [Wrapper 옵션] IMAGE COMMAND [ARG ...]
</strong></code></pre>



**2) 주요 옵션**

<table><thead><tr><th width="178.07244873046875">옵션</th><th>값</th><th>설명</th></tr></thead><tbody><tr><td><code>--runtime</code></td><td><code>singularity</code>, <code>apptainer</code>, <code>enroot</code>, <code>pyxis</code></td><td>실행 백엔드</td></tr><tr><td><code>--workload</code></td><td><code>generic</code>, <code>pytorch-dist*</code></td><td>워크로드 별 검증 규칙</td></tr><tr><td><code>--launcher</code></td><td><code>srun-native</code>, <code>torchrun</code></td><td>분산 프로세스 생성 방식**</td></tr><tr><td><code>--local-processes</code></td><td>양의 정수</td><td>노드별 torchrun worker 수</td></tr><tr><td><code>--mpi</code></td><td><code>none</code>, <code>cray</code>, <code>openmpi</code>, <code>auto</code></td><td>MPI 구성</td></tr><tr><td><code>--nccl</code></td><td><code>none</code>, <code>native</code>, <code>ofi</code>, <code>auto</code></td><td>NCCL 전송 구성</td></tr><tr><td><code>--provider</code></td><td></td><td></td></tr><tr><td><code>-B</code>, <code>--bind</code></td><td><code>SRC:DST:MODE</code></td><td>추가 마운트</td></tr><tr><td><code>--env</code></td><td><code>NAME=VALUE</code></td><td>컨테이너에 전달할 환경 변수</td></tr><tr><td><code>--diagnose</code></td><td>값 없음</td><td>rank 0에서 결정된 설정 출력</td></tr><tr><td><code>--doctor</code></td><td>값 없음</td><td>GPU, CXI, OFI 라이브러리 사전 점검</td></tr><tr><td><code>--dry-run</code></td><td>값 없음</td><td>실행하지 않고 최종 명령 출력</td></tr></tbody></table>

\***pytorch-dist** : torch.distributed 기반 분산 학습 (DDP, FSDP, DeepSpeed, NeMo/Megatron 등)

\*\***srun-native**는 Slurm이 GPU마다 학습 워커 프로세스를 직접  구동하고, **torchrun**은 노드당 torchrun 에이전트 하나가 그 안에서 학습 워커 프로세스를 구동합니다.



**3) 컨테이너 런타임 별 실행 방법**&#x20;

&#x20;   \- Singularity/Apptainer, Enroot

컨테이너 백엔드에서 `kisti-container`는 Slurm task 안에서 실행되는 Wrapper입니다. 따라서 배치 파일에 `srun`을 사용합니다.

```bash
srun --mpi=none \
  /apps/common/kisti-container/bin/kisti-container \
    --runtime singularity \
    IMAGE.sqsh python3 train.py
```

&#x20;   \- Pyxis

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



**4) 다중 노드 NCCL-OFI/CXI**

2노드 이상 GH200 작업은 다음 모듈을 배치 스크립트에서 로드합니다.

<pre class="language-bash"><code class="lang-bash"><strong>module load aws-ofi-nccl/1.20.0
</strong>module load libfabric/2.3.1
</code></pre>

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

### 6. 스케줄러(SLURM)를 통한 작업 실행

스케줄러(Slurm)를 통해 컨테이너 작업을 제출하는 방법입니다. 먼저 컨테이너  런타임에 따라 스크립트를 작성합니다.

#### 6.1. 작업 유형 별 예제

사용자는 관심 있는 작업 유형의 예제 디렉터리와 예제 실행에 필요한 이미지 파일을 자신의 작업 공간으로 복사합니다.

<pre class="language-bash"><code class="lang-bash"><strong>## 예제 디렉터리
</strong><strong>$ ls -l /apps/common/kisti-container/examples
</strong>01-cpu-smoke  
02-gpu-smoke  
03-mpi-omb  
04-pytorch-ddp  
05-pytorch-fsdp2  
06-nemo-megatron  
07-pytorch-deepspeed  

## 이미지 파일 디렉터리 : Enroot(.sqsh), Singularity/Apptainer(.sif)
- 기본 이미지 : /apps/common/kisti-container/images  
- Slingshot 고속 네트워크 지원 이미지 : /apps/common/kisti-container/images/slingshot 
</code></pre>

```bash
# 예제 디렉터리 복사 예시
$ mkdir -p /scratch/$USER/kisti-container/examples

$ cp -a /apps/common/kisti-container/examples/04-pytorch-ddp. \
  /scratch/$USER/kisti-container/examples/

# 이미지 파일 복사 예시
$ mkdir -p /scratch/$USER/kisti-container/images
cp -a /apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sif \
  /scratch/$USER/kisti-container/images/
cp -a /apps/common/kisti-container/images/pytorch-25.03-py3-aarch64.sqsh \
  /scratch/$USER/kisti-container/images/
```



**1)  MPI 작업 (kisti-container 사용)**

MPI 작업은 컨테이너 안의 MPI가 호스트의 Slingshot 네트워크를 통해 다른 노드와 통신해야 합니다. kisti-container는 `--mpi` 옵션으로 지정한 MPI 유형에 맞춰 **호스트의 MPI 및 통신 라이브러리를 컨테이너에서 사용할 수 있도록 연결**하고 관련 환경 변수를 설정합니다.&#x20;

kisti-container를 사용하는 MPI 작업은 --workload=generic 옵션(기본 설정값 이라서 생략 가능)을 사용하며, MPI 유형에 따른 관련 옵션은 아래와 같습니다.&#x20;

|    구분   |       srun 옵션      | kisti-container 옵션 |
| :-----: | :----------------: | :----------------: |
| CrayMPI | --mpi=cray\_shasta |     --mpi cray     |
| OpenMPI |     --mpi=pmix     |    --mpi openmpi   |

아래 예제는 2노드 8프로세스 환경에서 노드 간 MPI 통신 성능을 측정합니다. OMB(OSU Micro-Benchmarks)의 두 가지 측정을 수행하는데, `osu_multi_lat`은 여러 프로세스 쌍이 동시에 메시지를 주고받을 때의 지연시간을, `osu_mbw_mr`은 같은 조건에서의 대역폭과 메시지 처리율을 측정합니다.

```bash
## (kisti-container Singularity) MPI OMB(통신성능측정)작업 스크립트 예제
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
--[중략]--
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
--[중략]--
524288             190285.40         362940.59
1048576            191985.38         183091.53
2097152            193126.48          92089.88
4194304            193713.41          46184.88
</code></pre>



**2) PyTorch DDP**

이 예제는 2노드 8GPU 환경에서 DDP 통신이 정상 동작하는지 확인하는 간단한 테스트 입니다. 노드당 4개의 Slurm rank(GPU당 1개)를 생성하고, 각 rank가 자신의 rank 번호를 all-reduce로 합산해 결과가 예상값과 일치하는지 검증합니다. 모든 rank가 서로 다른 두 노드에 걸쳐 배치되었는지도 함께 확인하므로, 노드 간 통신까지 실제로 이루어졌음을 보장합니다.

정상 종료되면 rank별 호스트명과 GPU 정보, NCCL 설정값이 JSON으로 출력됩니다.

{% hint style="info" %}
**DDP :  Distributed Data Parallel**

&#x20;**Pytorch**의 **기본 분산 학습 방식**으로 각 GPU에 모델 전체를 복제하고, 서로 다른 데이터로 계산한 그래디언트를 동기화해 동일한 모델 상태를 유지합니다. 모델이 GPU 메모리에 올라가는 규모에 적합합니다. 모델 자체가 한 GPU에 들어가지 않으면 FSDP나 텐서 병렬(Megatron) 방식을 사용합니다.
{% endhint %}

**a. Pyxis 작업 스크립트 예제**

컨테이너 런타임을 직접 사용하는 방식으로, 분산 학습에 필요한 설정을 스크립트에서 직접 지정합니다. `--ntasks-per-node=4`로 Slurm이 GPU마다 학습 워커를 하나씩 띄우고, 워커들이 서로를 찾을 접속 지점(`MASTER_ADDR`)과 포트를 지정합니다. 포트는 작업 ID로 계산해 다른 작업과 겹치지 않게 합니다.

통신 설정을 확인하려면 주석 처리된 `NCCL_DEBUG` 두 줄을 해제하고 실행한 뒤, 출력에서 `Selected provider is cxi`가 나오는지 보시면 됩니다. `Using network Socket`이면 Slingshot 대신 일반 TCP로 동작하는 상태라 성능이 크게 떨어집니다.

```bash
grep -E 'Selected provider|Using network' ddp-pyxis-*.out | sort -u
```

```bash
## Pyxis 작업 스크립트 예제
## /apps/common/kisti-container/examples/04-pytorch-ddp/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=ddp-pyxis
#SBATCH --partition=normal
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
IMAGE=$ROOT/images/slingshot/kisti-pytorch-25.03-gh200-r1-aarch64.sqsh 
PROGRAM=$ROOT/examples/04-pytorch-ddp/ddp_smoke.py

CHECKPOINT_DIR=$SLURM_SUBMIT_DIR/checkpoints/ddp-pyxis-$SLURM_JOB_ID
mkdir -p "$CHECKPOINT_DIR"

module purge
module load enroot/4.2.0

# DDP rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER=$MASTER_ADDR:$MASTER_PORT WORLD_SIZE=$WORLD_SIZE"

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET

srun --mpi=none --kill-on-bad-exit=1 --distribution=block:block \
  --gpu-bind=none --gres-flags=allow-task-sharing \
  --container-image="$IMAGE" \
  python3 "$PROGRAM"
```



**b. kisti-container Pyxis 작업 스크립트 예제**

같은 작업을 kisti-container로 실행합니다. 위에서 직접 지정했던 설정을 래퍼가 대신 처리하므로 `MASTER_ADDR`과 `MASTER_PORT` 계산이 필요 없습니다.

`--nccl ofi`는 NCCL이 Slingshot을 쓰도록 통신 플러그인과 라이브러리를 연결합니다. 이 예제가 Slingshot 스택이 없는 일반 이미지를 쓰면서도 동작하는 이유이며, 호스트의 `aws-ofi-nccl`과 `libfabric` 모듈을 로드하는 것도 이 때문입니다. `--diagnose`를 붙이면 적용된 설정이 실행 전에 출력됩니다.

```bash
## kisti-container Pyxis 작업 스크립트 예제
## /apps/common/kisti-container/examples/04-pytorch-ddp/run-kc-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=kc-ddp-pyxis
#SBATCH --partition=normal
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

CHECKPOINT_DIR=$SLURM_SUBMIT_DIR/checkpoints/kc-ddp-pyxis-$SLURM_JOB_ID
mkdir -p "$CHECKPOINT_DIR"

module purge
module load aws-ofi-nccl/1.20.0
module load libfabric/2.3.1
module load enroot/4.2.0

"$KC" \
  --runtime pyxis --workload pytorch-dist \
  --launcher srun-native --nccl ofi --debug-nccl --diagnose \
  "$IMAGE" python3 "$PROGRAM"
```



작업제출:

<pre class="language-bash"><code class="lang-bash"><strong>$ sbatch run-pyxis.sbatch
</strong><strong>$ sbatch run-kc-pyxis.sbatch (kisti-container 사용) 
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
    --[중략]--
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



**3) NeMo/Megatron**&#x20;

이 예제는 GH200 2노드에서 **노드 내부 TP=4, 노드 간 DP=2**로 구성해 총 8개 GPU를 사용합니다. 통신이 잦은 TP는 노드 안에, 비교적 느슨한 DP는 노드 사이에 배치한 형태로, 대규모 모델 학습에서 일반적으로 쓰이는 구성입니다.

{% hint style="info" %}
**NeMo (Neural Modules)**\
NVIDIA의 AI 모델 개발·학습 프레임워크입니다. 현재 예제의 **NeMo/Megatron**은 NeMo가 제공하는 학습 환경 위에서 Megatron-LM의 모델 병렬 기능을 함께 사용하는 방식을 뜻합니다.

대규모 모델은 한 GPU의 메모리에 올라가지 않기 때문에, 모델 자체를 여러 GPU에 나누는 **모델 병렬(TP)**&#xACFC; 데이터를 나누어 처리하는 **데이터 병렬(DP)**&#xC744; 조합합니다.

\*\*TP (Tensor Parallel)\*\*는 모델의 각 레이어를 여러 GPU에 쪼개 한 번의 계산을 나눠 수행하는 방식입니다. 계산 중간에 GPU 간 통신이 자주 일어나므로, 노드 내부에서 NVLink로 연결된 GPU끼리 묶는 것이 성능에 유리합니다.

\*\*DP (Data Parallel)\*\*는 같은 모델 복제본이 서로 다른 데이터를 학습하고 그래디언트를 동기화하는 방식입니다. 통신 빈도가 상대적으로 낮아 노드를 넘어가도 부담이 적습니다.
{% endhint %}

**a. Pyxis 작업 스크립트 예제**

`--ntasks-per-node=1`로 노드당 torchrun 에이전트를 하나 띄우고, 그 에이전트가 GPU별 학습 워커 4개를 생성합니다. 총 8개 워커가 TP=4, DP=2를 구성합니다. 워커들이 서로를 찾는 과정은 torchrun의 rendezvous가 담당하며, `--max-restarts=0`은 Slurm의 종료 처리와 충돌하지 않도록 자동 재시작을 끕니다.

```bash
## Pyxis 작업 스크립트 예제
## /apps/common/kisti-container/examples/06-nemo-megatron/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=nemo-pyxis
#SBATCH --partition=normal
#SBATCH --comment="field=misc;appl=megatron-lm"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
IMAGE="$ROOT/images/slingshot/kisti-nemo-25.04-gh200-r1-aarch64.sqsh"
TRAIN_SCRIPT="$ROOT/examples/06-nemo-megatron/nemo_megatron_tp4_dp2_smoke.py"

CHECKPOINT_DIR=${SLURM_SUBMIT_DIR}/checkpoints/nemo-pyxis-${SLURM_JOB_ID}
mkdir -p "$CHECKPOINT_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

module purge
module load enroot/4.2.0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET

echo "MASTER=$MASTER_ADDR:$MASTER_PORT NODES=$SLURM_NNODES"

srun --mpi=none --kill-on-bad-exit=1 --distribution=block:block \
  --gpu-bind=none --gres-flags=allow-task-sharing \
  --container-image="$IMAGE" \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --max-restarts=0 \
    "$TRAIN_SCRIPT" --steps 5 --tp-size 4 --expected-dp-size 2 \
    --checkpoint-dir "$CHECKPOINT_DIR"
```

**b. kisti-container Pyxis 작업 스크립트 예제**

`--launcher torchrun --local-processes 4`만 지정하면 torchrun 실행과 rendezvous 설정이 자동으로 구성됩니다. 이미지는 Slingshot 스택이 없는 일반 NeMo 이미지를  사용하고, `--nccl ofi`로 호스트 라이브러리를 연결합니다.

```bash
## kisti-container Pyxis 작업 스크립트 예제
## /apps/common/kisti-container/examples/06-nemo-megatron/run-kc-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=kc-nemo-pyxis
#SBATCH --partition=normal
#SBATCH --comment="field=misc;appl=megatron-lm"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
KC="/apps/common/kisti-container/bin/kisti-container"
IMAGE="$ROOT/images/nemo-25.04.00-aarch64.sqsh"
TRAIN_SCRIPT="$ROOT/examples/06-nemo-megatron/nemo_megatron_tp4_dp2_smoke.py"

CHECKPOINT_DIR=${SLURM_SUBMIT_DIR}/checkpoints/kc-nemo-pyxis-${SLURM_JOB_ID}
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
$ sbatch run-pyxis.sbatch
$ sbatch run-kc-pyxis.sbatch
```

결과 파일:

```bash
$ cat nemo-pyxis-20467.out
--[중략]--
NEMO_MCORE_TPDP_RESULT status=PASS world_size=8 tp_size=4 dp_size=2 steps=5 last_dp_loss_avg=8.43100882 mean_step_ms=395.784 steady_mean_step_ms=49.037 collective=28.0 expected=28.0
```

통신 설정을 확인하려면 주석 처리된 `NCCL_DEBUG` 두 줄을 해제하고 실행한 뒤, 출력에서 다음을 확인하시기 바랍니다.

```bash
grep -E 'Selected provider|Using network' nemo-pyxis-*.out | sort -u
```

`Selected provider is cxi`가 나오면 Slingshot을 사용하는 것이고, `Using network Socket`이면 일반 TCP로 동작하는 상태라 성능이 크게 떨어집니다.



**4) DeepSpeed**&#x20;

ZeRO 최적화로 옵티마이저 상태와 그래디언트를 GPU들에 나눠 저장하는 데이터 병렬 학습 예제입니다. GH200 2노드에서 노드당 GPU 4장, 총 8개 rank로 ZeRO-2 학습이 동작합니다.

{% hint style="info" %}
**DeepSpeed**: Microsoft가 개발한 대규모 모델 학습 최적화 라이브러리입니다. 현재 예제의 ZeRO-2(Zero Redundancy Optimizer stage 2)는 각 GPU가 모델 전체를 가지되, 옵티마이저 상태와 그래디언트는 GPU 수만큼 나눠 저장해 GPU당 메모리 사용량을 줄이는 방식입니다.
{% endhint %}

**a. Pyxis 작업 스크립트 예제**

NeMo 예제와 같은 torchrun 구조이며, 8개 워커가 ZeRO-2 데이터 병렬을 구성합니다.

DeepSpeed는 실행 중 CUDA 커널을 컴파일해 캐시에 저장합니다. 경로를 지정하지 않으면 홈 디렉터리에 쓰려다 실패하거나 작업끼리 충돌할 수 있어, `CACHE_ROOT`를 작업 ID별로 나누어 `/scratch` 아래에 둡니다. ZeRO 단계 등 세부 설정은 `ds_config.json`에 있습니다.

<pre class="language-bash"><code class="lang-bash"><strong>## Pyxis 작업 스크립트 예제
</strong>## /apps/common/kisti-container/examples/07-pytorch-deepspeed/run-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=ds-pyxis
#SBATCH --partition=normal
#SBATCH --comment="field=misc;appl=deepspeed"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ROOT="/scratch/$USER/kisti-container"
IMAGE="$ROOT/images/slingshot/kisti-pytorch-25.03-gh200-r1-aarch64.sqsh"
TRAIN_SCRIPT="$ROOT/examples/07-pytorch-deepspeed/train_deepspeed.py"

module purge
module load enroot/4.2.0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,NET

echo "MASTER=$MASTER_ADDR:$MASTER_PORT NODES=$SLURM_NNODES"

USER=`id -un`
export CACHE_ROOT=/scratch/$USER/.cache/job-$SLURM_JOB_ID
mkdir -p $CACHE_ROOT

export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$CACHE_ROOT/torch_extensions
export XDG_CACHE_HOME=$CACHE_ROOT/xdg

srun --mpi=none --kill-on-bad-exit=1 --distribution=block:block \
  --gpu-bind=none --gres-flags=allow-task-sharing \
  --container-image="$IMAGE" \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --max-restarts=0 \
    "$TRAIN_SCRIPT" --deepspeed --deepspeed_config ${SLURM_SUBMIT_DIR}/ds_config.json --epochs 2
</code></pre>

**b.kisti-container Pyxis 작업 스크립트 예제**

NeMo 예제와 동일한 구조입니다. 캐시 환경변수는 래퍼가 다루지 않으므로 앞 예제와 같이 설정합니다.

```bash
## kisti-container Pyxis 작업 스크립트 예제
## /apps/common/kisti-container/examples/07-pytorch-deepspeed/run-kc-pyxis.sbatch

#!/bin/bash
#SBATCH --job-name=kc-ds-pyxis
#SBATCH --partition=normal
#SBATCH --comment="field=misc;appl=deepspeed"
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
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
$ sbatch run-pyxis.sbatch
$ sbatch run-kc-pyxis.sbatch
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



### 7. 참고 자료

#### 7.1 kisti-container&#x20;

**1) 실행 워크플로우**&#x20;

사용자 옵션과 Slurm 환경을 해석하여 컨테이너 실행 명령을 구성하는 Bash Wrapper

<figure><img src="../.gitbook/assets/image (19).png" alt=""><figcaption></figcaption></figure>

**2) 런타임별 실행과 통신 환경**

Slurm 작업에서 직접 런타임은 task 안에서 실행하고, Pyxis는 Wrapper가 srun을 생성

<figure><img src="../.gitbook/assets/image (22).png" alt=""><figcaption></figcaption></figure>



**3) 문제 해결**

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

[^1]: 
