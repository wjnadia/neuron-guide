---
hidden: true
---

# 컨테이너 활용 가이드(작성 중 초안)

뉴론 시스템은 HPC 응용 및 AI 학습/추론 등에서 복잡한 소프트웨어 의존성을 해결하고 다양한 시스템에서 일관된 작업 환경을  유지할 수 있도록 최적화된 컨테이너 활용 환경을 제공합니다. 사용자는 자신의 작업 단계(빌드, 실행)에 맞춰 적절한 도구를 선택하여 사용할 수 있습니다.

{% hint style="info" %}
* **Podman**: 일반 사용자 권한 기반의 이미지 빌드 및 관리 도구입니다. Docker를 대체하여 이미지를 생성할 때 사용합니다.
* **Enroot**: NVIDIA에서 개발한 HPC 전용 런타임입니다. 컨테이너 이미지를 SquashFS(`.sqsh`)로 변환하여 GPU 연산 성능을 극대화하며, Pyxis 플러그인을 통해 Slurm 스케줄러와 유기적으로 연동됩니다.
* **Singularity**: HPC 환경의 표준 컨테이너 도구로, 높은 범용성과 기존 구축된 `.sif` 이미지와의 호환성을 제공합니다
{% endhint %}

### 1.  컨테이너 도구 선택 가이드

사용자의 목적 및 실행 노드 특성에 따라 최적의 도구를 선택할 수 있습니다.

<table data-header-hidden><thead><tr><th width="104.800048828125" align="center"></th><th align="center"></th><th align="center"></th><th align="center"></th></tr></thead><tbody><tr><td align="center"><strong>구분</strong></td><td align="center"><strong>Podman</strong><br><strong>(빌드/관리)</strong></td><td align="center"><strong>Enroot</strong><br><strong>(고성능 실행)</strong></td><td align="center"><strong>Singularity</strong><br><strong>(범용 실행)</strong></td></tr><tr><td align="center">주요 역할</td><td align="center"><p>이미지 생성 </p><p>및 커스터마이징</p></td><td align="center"><p>GPU 연산 가속 </p><p>(권장)</p></td><td align="center">시스템 간<br>호환성 유지</td></tr><tr><td align="center">저장 형식</td><td align="center"><p>OCI 레이어 </p><p>(디렉터리)</p></td><td align="center"><p><code>.sqsh</code> </p><p>(단일 압축 파일)</p></td><td align="center"><p><code>.sif</code> </p><p>(단일 이미지 파일)</p></td></tr><tr><td align="center">특징</td><td align="center">Docker 명령어 호환</td><td align="center">빠른 로딩, <br>Nvidia GPU 최적화</td><td align="center">기존 뉴론 컨테이너환경 유지</td></tr></tbody></table>

{% hint style="info" %}
Singularity 에 대한 자세한 사용법은 [**Singularity 컨테이너**](https://docs-ksc.gitbook.io/neuron-user-guide/undefined-2/appendix-3-how-to-use-singularity-container) 를 참조하세요.
{% endhint %}

### 2. 이미지 빌드 및 관리 (Podman)

로그인 노드 또는 계산 노드에서 Podman을 사용하여 컨테이너 이미지를 준비합니다.

#### 가. 외부 이미지 가져오기 (Pull)

NGC(NVIDIA GPU Cloud) 등에서 이미지를 가져옵니다.

<pre><code># NGC에서 PyTorch 25.12 버전 가져오기
$ podman pull nvcr.io/nvidia/pytorch:25.12-py3
$ podman images
REPOSITORY              TAG         IMAGE ID      CREATED      SIZE
<strong>nvcr.io/nvidia/pytorch  25.12-py3   dd94fce2f83a  7 weeks ago  20.6 GB
</strong></code></pre>

#### 나. Dockerfile을 이용한 로컬 빌드

사용자 소스 코드나 특정 라이브러리를 포함한 커스텀 이미지를 생성합니다.

이미지 빌드에 많은 시간이 소요되고 부하가 많이 걸리는 경우, 스케줄러(SLURM)를 통해 인터랙티브 모드로 할당된  계산  노드에 접속하여 빌드하는 것을 권장합니다. &#x20;

<pre><code>## 스케줄러(SLURM)를 통해 인터랙티브 모드로 할당된 계산노드에 접속
$  srun --partition=cas_v100_4 --nodes=1 --ntasks-per-node=2 --cpus-per-task=10 --comment=pytorch --pty bash

# 현재 디렉터리(.)의 Dockerfile로 'my_pytorch:v1' 이미지 빌드
$ ls Dockerfile
Dockerfile
$ podman build -t my_pytorch:v1 . 
$ podman images
REPOSITORY              TAG         IMAGE ID      CREATED        SIZE
<strong>localhost/my_pytorch    v1          d9c8064f0996  6 seconds ago  20.6 GB
</strong>nvcr.io/nvidia/pytorch  25.12-py3   dd94fce2f83a  7 weeks ago    20.6 GB
</code></pre>

```
[Dockerfile 예시]
# 1. 베이스 이미지 지정 (NVIDIA GPU Cloud 제공 이미지)
FROM nvcr.io/nvidia/pytorch:25.12-py3

# 2. 메타데이터 설정
LABEL maintainer="user_id@ksc.re.kr"

# 3. 추가 시스템 패키지 설치 (필요시)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Python 라이브러리 추가 설치 (pip 활용)
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    matplotlib

# 5. 작업 디렉토리 설정
WORKDIR /workspace
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

# 특정 파일을 지정하여 빌드
$ podman build -f Dockerfile.gpu -t my_pytorch:gpu_ver .
```

{% hint style="info" %}
**이미지 관리 명령어 요약**
{% endhint %}

<table data-header-hidden><thead><tr><th width="91.4000244140625" align="center"></th><th width="284.1334228515625"></th><th></th></tr></thead><tbody><tr><td align="center"><sub><strong>기능</strong></sub></td><td><sub><strong>명령어</strong></sub></td><td><sub><strong>설명</strong></sub></td></tr><tr><td align="center"><sub>목록 확인</sub></td><td><sub><code>$ podman images</code></sub></td><td><sub>로컬에 저장된 이미지 리스트 출력</sub></td></tr><tr><td align="center"><sub>이미지 삭제</sub></td><td><sub><code>$ podman rmi [이미지ID]</code></sub></td><td><sub>불필요한 이미지 제거</sub></td></tr><tr><td align="center"><sub>상세 정보</sub></td><td><sub><code>$ podman inspect [이미지ID]</code></sub></td><td><sub>이미지 레이어, 환경변수 등 상세 정보 확인</sub></td></tr><tr><td align="center"><sub>태그 변경</sub></td><td><sub><code>$ podman tag [기존이름] [새이름]</code></sub></td><td><sub>이미지에 새로운 이름/태그 부여</sub></td></tr></tbody></table>

### 2. 이미지 업로드

Podman으로 빌드한 이미지를 외부 저장소인 Docker Hub 또는 사용자 레지스트리에 업로드하여 관리 및 공유 할 수 있습니다. 특히, 로그인  또는 계산 노드의 로컬 파일시스템에 저장된 이미지는 영구 보관되지  않기  때문에,  외부 저장소에 이미지를 업로드하는 것을 권장합니다.  &#x20;

#### 가. Docker Hub 로그인

먼저 Podman을 통해 Docker Hub 등의 계정에 인증합니다.

```
$ podman login docker.io
# Username과 Password(또는 Access Token)를 입력합니다.
```

#### 나. 이미지 태그(Tag) 설정

업로드를 위해서는 이미지 이름을 `계정명/이미지명:태그` 형식으로 지정해야 합니다.

```
# 로컬 이미지(my_pytorch:v1)를 Docker Hub 형식으로 태그
$ podman tag localhost/my_pytorch:v1 docker.io/내계정ID/my_pytorch:v1
```

#### 다. 이미지 업로드(Push)

태그가 완료된 이미지를 Docker Hub 레지스트리로 전송합니다.

```
$ podman push docker.io/내계정ID/my_pytorch:v1
```



### 3. 이미지 변환&#x20;

Podman으로 준비한 이미지는 컨테이너 실행 도구에 맞는 변환 과정을 거쳐야  합니다.

#### 가. Enroot&#x20;

이미지를 SquashFS 포맷으로 변환하여 로딩 속도를 높이고 GPU 연산 효율을 최적화 합니다.

```
# .sqsh 이미지 생성
$ enroot import -o my_pytorch.sqsh podman://my_pytorch:v1ㄸ
```

{% hint style="info" %}
**Enroot 이미지 관련 명령어**&#x20;
{% endhint %}

<table data-header-hidden><thead><tr><th width="74.86663818359375" align="center"></th><th width="149.5999755859375" align="center"></th><th></th></tr></thead><tbody><tr><td align="center"><sub><strong>단계</strong></sub></td><td align="center"><sub><strong>작업 내용</strong></sub></td><td><sub><strong>명령어 / 설정 예시</strong></sub></td></tr><tr><td align="center"><sub>이미지 가져오기</sub></td><td align="center"><sub>Docker Hub 등      외부 레지스트리에서 직접 가져오기</sub></td><td><p></p><p><sub><code>$ enroot import docker://ubuntu:latest//image:tag</code></sub></p></td></tr><tr><td align="center"><sub>이미지 리스트</sub></td><td align="center"><sub>이미지 목록 확인</sub></td><td><sub><code>$ enroot list</code></sub></td></tr><tr><td align="center"><sub>이미지 삭제</sub></td><td align="center"><sub>더 이상 사용하지 않는 이미지 제거</sub></td><td><sub><code>$ enroot remove [image_name]</code></sub></td></tr></tbody></table>

#### 나. Singularity

기존 작업 방식 유지 또는 타 시스템과의 이미지 공유가 필요한 경우 활용합니다.

```
# Podman 이미지를 tar로 내보낸 후 .sif 파일로 빌드
$ podman save my_pytorch:v1 -o my_pytorch.tar
$ singularity build --fakeroot my_pytorch.sif docker-archive://my_pytorch.tar
```



### 4. 이미지 실행

생성된 이미지는  Enroot 또는 Singularity 환경에 배포하여 실행할 수 있습니다.

#### 가.Enroot&#x20;

```bash
# GPU 계산 노드에서 squashFS 이미지를 로드하고 실행
# GPU 가속 연동 옵션 필요 없음(자동 연동됨)
$ enroot start my_pytorch.sqsh nvidia-smi
$ enroot start my_pytorch.sqsh python train.py
```

#### 나. Singularity

```bash
# GPU 계산노드에서 Singularity 이미지를 로드하여 실행
# --nv: GPU 가속 연동 옵션 필수
$ singularity exec --nv my_pytorch.sif nvidia-smi
$ singularity exec --nv my_pytorch.sif python train.py
```

### 5. 스케줄러(SLURM)를 통한 작업 실행

스케줄러(Slurm)를 통해 컨테이너 작업을 제출하는 방법입니다. 사용 도구(Enroot(Pyxis) 또는 Singularity)에 따라 스크립트를 작성합니다.

#### 가. Enroot(Pyxis) 활용 예시

Pyxis는 Slurm의 `srun` 옵션을 확장하여, 사용자가 복잡한 Enroot 명령어를 직접 입력하지 않아도 컨테이너 환경을 자동으로 구성해 줍니다.

```
[예시 1]
#!/bin/bash
#SBATCH –J pytorch # job name
#SBATCH --time=1:00:00 # wall_time
#SBATCH -p cas_v100_4
#SBATCH --comment pytorch # application name
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=10 
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH —gres=gpu:2 # number of GPUs per node

# Pyxis 플러그인을 이용한 컨테이너 실행
# --container-image: 사용할 .sqsh 이미지 경로
# --container-workdir: 컨테이너 내 작업 디렉토리 설정
srun --container-image=./my_env.sqsh \
     --container-workdir=/scratch/[ID] \
     python train.py
```

```
[예시 2]
#!/bin/bash
#SBATCH –J pytorch # job name
#SBATCH --time=1:00:00 # wall_time
#SBATCH -p cas_v100_4
#SBATCH --comment pytorch # application name
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=10 
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH —gres=gpu:2 # number of GPUs per node
# Pyxis 전용 #SBATCH 파라미터 설정
#SBATCH --container-image=./my_pytorch.sqsh    # 사용할 Enroot 이미지 경로
#SBATCH --container-workdir=/scratch/[ID]      # 컨테이너 내 작업 디렉토리 설정

srun python train.py
```

{% hint style="info" %}
**Pyxis 주요 #SBATCH 파라미터 설명**
{% endhint %}

<table data-header-hidden><thead><tr><th width="202.1334228515625"></th><th></th></tr></thead><tbody><tr><td><strong>파라미터</strong></td><td><strong>설명</strong></td></tr><tr><td><code>--container-image</code></td><td>사용할 컨테이너 이미지 경로 (<code>.sqsh</code> 파일 또는 <code>docker://</code> 주소)</td></tr><tr><td><code>--container-mounts</code></td><td>마운트할 경로 설정 (형식: <code>호스트경로:컨테이너경로</code>)    <sub>* /home01, /scratch, /apps는 지정하지 않아도 자동 마운트 됨</sub></td></tr><tr><td><code>--container-workdir</code></td><td>컨테이너 실행 시 시작 위치(Working Directory) 지정</td></tr><tr><td><code>--container-name</code></td><td>실행 중인 컨테이너에 부여할 이름 (디버깅 용도)</td></tr><tr><td><code>--container-save</code></td><td>작업 종료 후 변경된 컨테이너 상태를 <code>.sqsh</code>로 저장 (필요 시)</td></tr></tbody></table>

#### 나. Singularity 활용 예시

```
#!/bin/bash
#SBATCH –J pytorch # job name
#SBATCH --time=1:00:00 # wall_time
#SBATCH -p cas_v100_4
#SBATCH --comment pytorch # application name
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=10 
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH —gres=gpu:2 # number of GPUs per node

# --nv: GPU 가속 연동 옵션 필수
singularity exec --nv my_pytorch.sif python train.py
```
