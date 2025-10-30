# Singularity 컨테이너

싱귤레러티(Singularity)는 도커(Docker)와 같이 OS 가상화를 구현하기 위한 HPC 환경에 적합한 컨테이너 플랫폼입니다. 사용자 작업 환경에 적합한 리눅스 배포판, 컴파일러, 라이브러리, 애플리케이션 등을 포함하는 컨테이너 이미지를 빌드하고 빌드된 컨테이너 이미지를 구동하여 사용자 프로그램을 실행할 수 있습니다.

Tensorflow, Caffe, Pytorch와 같은 딥러닝 프레임워크와 Quantum Espresso, Lammps, Gromacs, Paraview 등을 지원하는 빌드된 **컨테이너 이미지는** **/apps/applications/singularity\_images/ngc 디렉터리**에서 액세스 할 수 있습니다.

![](../.gitbook/assets/vm-container_comparison.JPG)       ![](<../.gitbook/assets/image (2).png>)

&#x20; < 가상머신과 컨테이너 아키텍처 비교>                 \<Singularity 컨테이너 아키텍처>

※ 가상머신은 애플리케이션이 하이퍼바이저와 게스트 OS를 거쳐 올라가는 구조이나, 컨테이너는 물리적인 하드웨어에 더 가까우며 별도의 게스트 OS가 아닌 호스트 OS를 공유하기 때문에 오버헤드가 더 적습니다. 최근 클라우드 서비스에서 컨테이너의 활용이 증가하고 있습니다.



### (동영상가이드)  싱귤레러티 컨테이너 이미지 빌드 및 실행 방법&#x20;

{% embed url="https://youtu.be/5PYAE0fuvBk?si=23ZvE7NhBAyZ9LM1" %}

##

## 가. 컨테이너 이미지 빌드하기

### **1. 싱귤레러티 모듈 적재 혹은 경로 설정**

```
$ module load singularity/3.11.0
or
$ $HOME/.bash_profile
export PATH=$PATH:/apps/applications/singularity/3.11.0/bin/
```

### **2. 로컬 빌드**

* 뉴론 시스템의 로그인 노드에서 컨테이너 이미지를 로컬 빌드하기 위해서는, 먼저 [**KISTI 홈페이지 > 기술지원 > 상담신청**을](https://www.ksc.re.kr/gsjw/gsjw/qna/edit) 통해 아래와 같은 내용으로 fakeroot 사용 신청을 해야합니다.
  * 시스템명 : 뉴론
  * 사용자 ID : a000bcd
  * 요청사항 : 싱귤레러티 fakeroot 사용 설정
* [**NGC(Nvidia GPU Cloud)에서 배포하는 도커 컨테이너**](https://catalog.ngc.nvidia.com/containers)로 부터 뉴론 시스템의 Nvidia GPU에 최적화된 딥러닝 프레임워크 및 HPC 애플리케이션 관련 싱규레러티 컨테이너 이미지를 빌드할 수 있습니다.



{% code title="[이미지 빌드 명령어] " overflow="wrap" %}
```
$ singularity [global options...] build [local options...] ＜IMAGE PATH＞ ＜BUILD SPEC＞

[주요 global options]
    -d : 디버깅 정보를 출력함
    -v : 추가 정보를 출력함
    --version : 싱귤레러티 버전 정보를 출력

[관련 주요 local options]
    --fakeroot : roor 권한 없이 일반사용자가 가짜 root 사용자로 이미지 빌드 
    --remote : 외부 싱귤레러티 클라우드(Sylabs Cloud)를 통한 원격 빌드(root 권한 필요 없음)
    --sandbox : 샌드박스 형태의 쓰기 가능한 이미지 디렉터리 빌드

＜IMAGE PATH＞
   default : 읽기만 가능한 기본 이미지 파일(예시 : ubuntu1.sif)
   sandbox : 읽기 및 쓰기 가능한 디렉터리 구조의 컨테이너(예시 : ubuntu4) 

＜BUILD SPEC＞
definition file : 컨테이너를 빌드하기 위해 recipe를 정의한  파일(예시 : ubuntu.def)
local image : 싱귤레러티 이미지 파일 혹은 샌드박스 디렉터리(IMAGE PATH 참조)
URI 
library:// 컨테이너 라이브러리 (default https://cloud.sylabs.io/library) 
docker:// 도커 레지스트리 (default 도커 허브)
docker-archive:// 도커 아카이브 파일
shub:// 싱규레러티 레지스트리 (default 싱귤레러티 허브)
oras:// OCI 레지스트리
oci-archive:// OCI 아카이브 파일
```
{% endcode %}

***

{% code title="[예시]" overflow="wrap" fullWidth="false" %}
```shell-session
① Definition 파일로부터 ubuntu1.sif 이미지 빌드하기
 $ singularity build --fakeroot ubuntu1.sif ubuntu.def* 

② 싱규레러티 라이브러리로부터 ubuntu2.sif 이미지 빌드하기
 $ singularity build --fakeroot ubuntu2.sif library://ubuntu:18.04 

③ 도커 허브로부터 ubuntu3.sif 이미지 빌드하기
 $ singularity build --fakeroot ubuntu3.sif docker://ubuntu:18.04 
 
④ 도커 tar 파일로부터 pytorch.sif 이미지 빌드하기
 $ singularity build --fakeroot pytorch.sif docker-archive://pytorch.tar
 
⑤ NGC(Nvidia GPU Cloud) 도커 레지스트리로부터 '22년 03월 배포 pytorch 이미지 빌드하기
 $ singularity build --fakeroot pytorch1.sif docker://nvcr.io/nvidia/pytorch:22.03-py3

⑥ Definition 파일로부터 pytorch.sif 이미지 빌드하기
 $ singularity build --fakeroot pytorch2.sif pytorch.def**

⑦ fakeroot 사용하지 않고 Definition 파일로부터  ubuntu4.sif 이미지 빌드하기
   # singularity 3.11.0 버전 이상에서 지원
   # Definition 파일에서 기존 컨테이너 이미지 기반 패키지 설치에 적합하며,     
     apt-get과 같은 시스템패키지 관리자를 사용하는 경우 일부 패키지(git 등) 설치 과정에서 
     에러가 발생할 수 있습니다.
 $ singularity build ubuntu4.sif ubuntu.def*
 
* ) ubuntu.def 예시
 bootstrap: docker
 from: ubuntu:18.04
 %post
 apt-get update
 apt-get install -y wget bash gcc gfortran g++ make file
 %runscript
 echo "hello world from ubuntu container!"

** ) pytorch.def 예시
 # 로컬 이미지 파일로부터 콘다를 사용하여 새로운 패키지 설치를 포함한 이미지 빌드
 bootstrap: localimage
 from: /apps/applications/singularity_images/ngc/pytorch:22.03-py3.sif
 %post
 pip install jupyter
 
 # 외부 NGC 도커 이미지로부터 콘다를 사용하여 새로운 패키지 설치를 포함한 이미지 빌드
 bootstrap: docker
 from: nvcr.io/nvidia/pytorch:22.03-py3
 %post
 pip install jupyter
```
{% endcode %}



### 3.cotainr 사용하여  빌드

cotainr은 사용자가  뉴론  및  자신의 시스템에서 사용중인 conda 패키지를 포함하는 싱귤레러티 컨테이너  이미지를 좀 더 쉽게 빌드할 수 있게 지원하는 도구입니다.

* 사용자  conda environment를 yml 파일로 export하여 사용자  conda 패키지를 포함하는 뉴론 시스템을 위한 사용자 싱귤레러티 컨테이너 이미지를 빌드할 수 있습니다.
* 뉴론 및 자신의 시스템에서 기존  conda environment를  yml 파일로 export 하는 방법은 아래와 같습니다.

```
(base) $ conda env export > my_conda_env.yml
```

{% code fullWidth="false" %}
```
(base) $ cat my_conda_env.yml   <-- 예시
name: base
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_kmp_llvm
  - archspec=0.2.3=pyhd8ed1ab_0
  - boltons=23.1.1=pyhd8ed1ab_0
  - brotli-python=1.1.0=py39h3d6467e_1
  - bzip2=1.0.8=hd590300_5
  - c-ares=1.27.0=hd590300_0
  - ca-certificates=2024.2.2=hbcca054_0
  - certifi=2024.2.2=pyhd8ed1ab_0
  - cffi=1.16.0=py39h7a31438_0
  - charset-normalizer=3.3.2=pyhd8ed1ab_0
  - cloudpickle=3.0.0=pyhd8ed1ab_0
  - colorama=0.4.6=pyhd8ed1ab_0
  - conda=24.1.2=py39hf3d152e_0
  - conda-libmamba-solver=24.1.0=pyhd8ed1ab_0
  - conda-package-handling=2.2.0=pyh38be061_0
  - conda-package-streaming=0.9.0=pyhd8ed1ab_0
  - cuda-cudart=12.1.105=0
  
  ......
  
  - pip:
      - annotated-types==0.6.0
      - deepspeed==0.14.0
      - hjson==3.1.0
      - ninja==1.11.1.1
      - py-cpuinfo==9.0.0
      - pydantic==2.6.3
      - pydantic-core==2.16.3
      - pynvml==11.5.0
```
{% endcode %}



cotainr을 사용하기 위해서는 module 명령을 사용하여 singularity 및cotainr 모듈을 먼저 로드해야 합니다.

```
$ module load  singularity cotainr
```



cotainr build를 사용하여 컨테이너  이미지를 빌드할 때 컨테이너에 대한 기본 이미지를 직접 지정하거나(-base-image 옵션 사용) --system 옵션을 사용하여 뉴론  시스템에 권장되는 기본 이미지를 사용할 수 있습니다.

{% code overflow="wrap" fullWidth="false" %}
```
$ cotainr info 
.....
System info
Available system configurations: 
- neuron-cuda     <- Ubuntu 20.04, CUDA 11.6.1, IB 사용자라이브러리 등이 설치된 컨테이너 이미지
```
{% endcode %}

{% code overflow="wrap" fullWidth="false" %}
```

$ cotainr build --system=neuron-cuda --conda-env=my_conda_env.yml --accept-licenses my_container.sif
  
```
{% endcode %}



&#x20;singularity exec 명령어로 빌드한  컨테이너 이미지를 실행하여 아래 예제와 같이  컨테이너에  생성된  conda environment 리스트를 확인할 수 있습니다.

```
$ singularity exec --nv my_container.sif conda env list
# conda environments:
#
base                  *  /opt/conda/envs/conda_container_env
```

###

### **4. 원격빌드**

{% code overflow="wrap" fullWidth="false" %}
```shell-session
① Sylabs Cloud에서 제공하는 원격 빌드 서비스를 이용하여 Definition 파일로부터 ubuntu4.sif 이미지 빌드하기
   $ singularity build --remote ubuntu4.sif ubuntu.def 
```
{% endcode %}

※ Sylabs Cloud(https://cloud.sylabs.io)에서 제공하는 원격빌드 서비스를 이용하려면 액세스 토큰을 생성하여 뉴론 시스템에 등록해야 합니다. [**\[참조 1\]**](appendix-3-how-to-use-singularity-container.md#h_7595929387701647407267724)\
※ 또한, Sylabs Cloud에 웹 브라우저 접속을 통해서 싱귤레러티 컨테이너 이미지의 생성∙관리가 가능합니다.  [**\[참조 2\]**](appendix-3-how-to-use-singularity-container.md#h_6582700487751647407356239)



### **5. 컨테이너 이미지 가져오기/내보내기**

{% code fullWidth="false" %}
```
① Sylabs cloud 라이브러리에서 컨테이너 이미지 가져오기  
$ singularity pull tensorflow.sif library://dxtr/default/hpc-tensorflow:0.1 

② 도커 허브에서 이미지를 가져와서 싱귤레러티 이미지로 변환
 $ singularity pull tensorflow.sif docker://tensorflow/tensorflow:latest

③ Sylabs Cloud 라이브러리에 싱귤레러티 이미지 내보내기(업로드)
 $ singularity push -U tensorflow.sif library://ID/default/tensorflow.sif 
```
{% endcode %}

※ Sylabs Cloud 라이브러리에 컨테이너 이미지를 내보내기(업로드) 위해서는 먼저 액세스 토큰을 생성하여 뉴론 시스템에 등록해야 합니다. [**\[참조 1\]**](appendix-3-how-to-use-singularity-container.md#h_7595929387701647407267724)

***

### **6. 컨테이너 이미지에서 제공되지 않는 파이썬 패키지 등을 사용자 홈 디렉터리에 설치하는 방법**

{% code overflow="wrap" fullWidth="false" %}
```shell-session
① pip install --user [파이썬 패키지 이름/버전], 사용자의 /home01/ID/.local 디렉터리에 설치됨 
 $ module load ngc/tensorflow:20.09-tf1-py3 (텐서플로우 컨테이너 모듈 로드)
 $ pip install --user keras==2.1.2 
 $ pip list --user
 Package Version
 ----------- -------
 Keras 2.1.2

② conda install —use-local [콘다 패키지 이름/버전], 사용자의 /home01/ID/.conda/pkgs 디렉터리에 설치됨
 $ module load ngc/pytorch:20.09-py3 (파이토치 컨테이너 모듈 로드)
 $ conda install --use-local matplotlib -y 
 $ conda list matplotlib
 # Name Version Build Channel
 matplotlib 3.3.3 pypi_0 pypi
```
{% endcode %}

※ 단, 여러 가지 컨테이너 이미지를 사용하는 경우 사용자 프로그램 실행 시 사용자 홈 디렉터리에 추가로 설치한 패키지를 먼저 찾기 때문에 다른 컨테이너 이미지에서 요구하는 패키지와 충돌이 발생하여 정상적으로 동작하지 않을 수 있습니다.



## 나. 싱귤레러티 컨테이너에서 사용자 프로그램 실행

### **1. 싱귤레러티 모듈 적재 혹은 경로 설정**

```
$ module load singularity/3.11.0
or
$ $HOME/.bash_profile
export PATH=$PATH:/apps/applications/singularity/3.11.0/bin/
```

### **2. 싱귤레러티 컨테이너에서 프로그램 실행 명령어**

```
$ singularity [global options...] shell [shell options...] ＜container＞
$ singularity [global options...] exec [exec options...] ＜container＞ ＜command＞
$ singularity [global options...] run [run options...] ＜container＞
```

{% code title="[예제]" overflow="wrap" fullWidth="false" %}
```
① Nvidia GPU 장착 계산 노드의 싱귤레러티 컨테이너에서 쉘 실행 후 사용자 프로그램 실행 
$ singularity shell --nv＊ tensorflow_22.03-tf1-keras-py3.sif
Singularity> python test.py

② Nvidia GPU 장착 계산 노드의 싱귤레러티 컨테이너에서 사용자 프로그램 실행
$ singularity exec --nv tensorflow_22.03-tf1-keras-py3.sif python test.py 
$ singularity exec --nv docker://tensorflow/tensorflow:latest python test.py
$ singularity exec --nv library://dxtr/default/hpc-tensorflow:0.1 python test.py

③ Nvidia GPU 장착 계산 노드의 싱귤레러티 컨테이너에서 runscript(이미지 빌드 시 생성)가 존재하면 이 스크립트를 먼저 실행한 후, 
사용자 명령어(아래 예제에서 python --version)가 존재하면 이어서 실행됩니다.

$ singularity run --nv /apps/applications/singularity_images/ngc/tensorflow_22.03-tf1-keras-py3.sif ＼
 python --version 
================
== TensorFlow ==
================

NVIDIA Release 22.03-tf1 (build 33659237)
TensorFlow Version 1.15.5

Container image Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright 2017-2022 The TensorFlow Authors. All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

NOTE: CUDA Forward Compatibility mode ENABLED.
 Using CUDA 11.6 driver version 510.47.03 with kernel driver version 460.32.03.
 See https://docs.nvidia.com/deploy/cuda-compatibility/ for details.

NOTE: Mellanox network driver detected, but NVIDIA peer memory driver not
 detected. Multi-node communication performance may be reduced.

Python 3.8.10 (default, Nov 26 2021, 20:14:08)
```
{% endcode %}

※ 싱귤레러티의 명령어\[shell | exec | run | pull ...] 별 도움말을 보려면 “singularity help \[command]”를 실행합니다.

※ 계산/로그인 노드에서 Nvidia GPU를 사용하기 위해서는 --nv 옵션을 사용해야 합니다.

### **3. NGC 컨테이너 모듈을 사용하여 사용자 프로그램 실행**

모듈 명령어를 사용하여 NGC 싱규레러티 컨테이너 이미지와 관련된 모듈을 로드하면 싱귤레러티 명령어를 입력하지 않아도 자동으로 컨테이너 이미지가 구동되어 좀 더 쉽게 싱귤레러티 컨테이너에서 사용자 프로그램을 실행할 수 있습니다.

* NGC 컨테이너 모듈을 로드하여 컨테이너에서 사용자 프로그램 실행하기

{% code overflow="wrap" %}
```shell-session
① tensorflow 1.15.5 지원 싱규레러티 컨테이너 이미지(tensorflow_22.03-tf1-keras-py3.sif)를 자동으로 구동하여 사용자 프로그램 실행
 $ module load singularity/3.9.7 ngc/tensorflow:22.03-tf1-py3 
 $ mpirun -H gpu39:2,gpu44:2 –n 4 python keras_imagenet_resnet50.py

② lammps 지원 싱규레러티 컨테이너 이미지(lammps:15Jun2020-x86_64.sif)를 자동으로 구동하여 lammps 실행
 $ module load singularity/3.6.4 ngc/lammps:15Jun2020 
 $ mpirun –H gpu39:2,gpu44:2 -n 4 lmp -in in.lj.txt -var x 8 -var y 8 -var z 8 -k on g 2 \
 -sf kk -pk kokkos cuda/aware on neigh full comm device binsize 2.8

③ gromacs 지원 싱규레러티 컨테이너 이미지(gromacs:2020.2-x86_64.sif)를 자동으로 구동하여 gromacs 실행
 $ module load singularity/3.6.4 ngc/gromacs:2020.2 
 $ gmx mdrun -ntmpi 2 -nb gpu -ntomp 1 -pin on -v -noconfout –nsteps 5000 \
 –s topol.tpr singularity shell --nv＊ tensorflow:20.09-tf1-py3.sif 
```
{% endcode %}

※ 컨네이너 이미지 모듈 로드 후 실행명령어 입력만으로 “singularity run --nv <컨테이너> \[실행명령어]”가 자동 실행됩니다.

* NGC 컨테이너 모듈 리스트

※ NGC(https://ngc.nvidia.com)에서 Nvidia GPU에 최적화하여 빌드 배포한 도커 컨테이너 이미지를 싱귤레러티로 변환합니다.\
※ 컨테이너 이미지 파일 경로 : /apps/applications/singularity\_images/ngc

```shell-session
1009% [a123a01@glogin01 ngc]$ module av
```

\-- \[중략] --

<figure><img src="../.gitbook/assets/singularity-neuron-ngc-module.png" alt=""><figcaption></figcaption></figure>

### **4. 스케줄러(SLURM)를 통한 컨테이너 실행 방법**

* **GPU 싱귤레러티 컨테이너 작업 실행**

#### **1) 작업 스크립트를 작성하여 배치 형태 작업 실행**

* 실행 명령어 : sbatch＜작업 스크립트 파일＞

```
[id@glogin01]$ sbatch job_script.sh
Submitted batch job 12345
```

※ 자세한 스케줄러(SLURM) 사용 방법은 "뉴론 지침서-스케줄러(SLURM)를 통한 작업실행" 참조

※ [**\[참조 3\]**](appendix-3-how-to-use-singularity-container.md#h_9146219267821647408138749)를 통해 병렬 학습 실행 예제 프로그램을 따라해 볼 수 있습니다.

***

#### **2) 스케줄러가 할당한 계산 노드에서 인터랙티브 작업 실행**

* 스케줄러를 통해 계산노드를 할당받아 첫번째 계산노드에 쉘 접속 후 인터렉티브 모드로 사용자 프로그램 실행

{% code overflow="wrap" %}
```
[id@glogin01]$ srun --partition=cas_v100_4 --nodes=1 --ntasks-per-node=2 --cpus-per-task=10 --gres=gpu:2 --comment=pytorch --pty bash 
[id@gpu10]$ 
[id@gpu10]$ module load singularity/3.11.0 
[id@gpu10]$ singularity run --nv /apps/applications/singularity_images/ngc/pytorch_22.03-hd-py3.sif python test.py
```
{% endcode %}

※ 1노드 점유, 노드 당 2 타스크, 타스크 당 10 CPUs, 노드 당 2GPU 사용 예제



* **GPU 싱귤레러티 컨테이너 작업 스크립트 예시**

#### **1) 단일 노드**

* 실행 명령어 : singularity run --nv <컨테이너> \[사용자 프로그램 실행 명령어]

{% code overflow="wrap" %}
```
#!/bin/sh
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

module load singularity/3.11.0 

singularity run --nv /apps/applications/singularity_images/ngc/pytorch_22.03-hd-py3.sif python test.py
```
{% endcode %}

※ 1노드 점유, 노드 당 2 타스크, 타스크 당 10 CPUs, 노드 당 2GPU 사용 예제



#### **2) 멀티 노드-1**&#x20;

* 실행 명령어 : srun singularity run --nv <컨테이너> \[사용자 프로그램 실행 명령어]

{% code overflow="wrap" %}
```
#!/bin/sh
#SBATCH –J pytorch_horovod # job name
#SBATCH --time=1:00:00 # wall_time
#SBATCH –p cas_v100_4 # partition name
#SBATCH --comment pytorch # application name
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=2 
#SBATCH --cpus-per-task=10 
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

module load singularity/3.11.0 gcc/4.8.5 mpi/openmpi-3.1.5

srun singularity run --nv /apps/applications/singularity_images/ngc/pytorch_22.03-hd-py3.sif \
python pytorch_imagenet_resnet50.py 
```
{% endcode %}

※ 2노드 점유, 노드 당 2 타스크(총 4개 MPI 프로세스-horovod 사용), 타스크 당 10CPUs, 노드 당 2GPU 사용 예제



#### **3) 멀티 노드-2**

* NGC 컨테이너 모듈을 로드하면 사용자 프로그램 실행 시 지정된 싱귤레러티 컨테이너를 자동으로 구동합니다.
* 실행 명령어 : mpirun\_wrapper \[사용자 프로그램 실행 명령어]

```
#!/bin/sh
#SBATCH –J pytorch_horovod # job name
#SBATCH --time=1:00:00 # wall_time
#SBATCH –p cas_v100_4 # partition name
#SBATCH --comment pytorch # application name
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10 
#SBATCH –o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

module load singularity/3.11.0  ngc/pytorch:22.03-py3

mpirun_wrapper python pytorch_imagenet_resnet50.py 
```

※ 2노드 점유, 노드 당 2 타스크(총 4개 MPI 프로세스-horovod 사용), 타스크 당 10CPUs, 노드 당 2GPU 사용 예제



##

## 다. 참조

### \[참조1] Sylabs Cloud 액세스 토큰 생성 및 뉴론 시스템에 토큰 등록하기 <a href="#h_7595929387701647407267724" id="h_7595929387701647407267724"></a>

[**\[Sylabs Cloud 바로가기\]**](https://cloud.sylabs.io/)

![](<../.gitbook/assets/Sylabs Cloud 계정 등록 및 로그인 하기.png>)

![](<../.gitbook/assets/새로운 토큰 생성하기.png>)

![](<../.gitbook/assets/클립보드로 토큰 복사하기.png>)

![](<../.gitbook/assets/토큰 입력하기.png>)

### \[참조2] 웹 브라우저에서 리모트 빌더에 의한 싱귤레러티 컨테이너 빌드하기 <a href="#h_6582700487751647407356239" id="h_6582700487751647407356239"></a>

[**\[Sylabs Cloud 바로가기\]**](https://cloud.sylabs.io/)

![](<../.gitbook/assets/웹 브라우저에서 컨테이너 이미지 빌드하기.png>)

![](<../.gitbook/assets/빌드한 컨테이너 이미지 목록 보기.png>)

### \[참조3] 병렬 학습 프로그램 실행 예제 <a href="#h_9146219267821647408138749" id="h_9146219267821647408138749"></a>

* 아래 예제는 싱귤레러티 컨테이너에서 pytorch 혹은 keras(tensorflow)로 작성된 resnet50 모델을 사용하여 imagenet 이미지 분류를 위한 병렬 학습 실행을 사용자가 직접 따라해 볼 수 있도록 구성됩니다.

{% code overflow="wrap" %}
```
▪ 병렬 학습 작업 스크립트 경로 : /apps/applications/singularity_images/examples

▪ 컨테이너 이미지 디렉터리 경로 : /apps/applications/singularity_images/ngc

▪ 병렬 학습 예제 프로그램 경로 
  - pytorch 프로그램
    (단일노드) /apps/applications/singularity_images/examples/pytorch/resnet50v1.5
    (멀티노드-horovod) /apps/applications/singularity_images/examples/horovod/examples/pytorch
  - keras(Tensorflow) 프로그램
    (멀티노드-horovod) /apps/applications/singularity_images/examples/horovod/examples/keras

▪ imagent 이미지 데이터 경로
  - (학습 데이터) /apps/applications/singularity_images/imagenet/train
  - (검증 데이터) /apps/applications/singularity_images/imagenet/val
```
{% endcode %}

***

**1) /apps/applications/singularity\_images/examples 디렉터리에서 아래 작업 스크립트 파일을 사용자 작업 디렉터리로 복사합니다.**

{% code overflow="wrap" %}
```shell-session
[a1234b5@glogin01]$ cp /apps/applications/singularity_images/examples/*.sh /scratch/ID/work/
```
{% endcode %}

***

**2) STATE가 idle 상태인 계산 노드가 있는 파티션을 확인합니다.**\
아래 예제에서는 cas\_v100nv\_8, cas\_v100nv\_4, cas\_v100\_4, cas\_v100\_2 등의 파티션에 가용 계산노드가 존재합니다.

```shell-session
[a1234b5@glogin01]$ sinfo
```

![](../.gitbook/assets/MsH42Y1SsRiobwT.png)

***

**3) 작업 스크립트 파일에서 작업명(-J), wall\_time(--time), 작업 큐(-p), Application이름(--comment), 계산노드 자원 요구량(--nodes, --ntasks-per-node, --gres) 등의 스케줄러 옵션과 학습 프로그램의 파라미터를 변경합니다.**

{% code overflow="wrap" %}
```bash
[a1234b5@glogin01]$ vi 01.pytorch.sh
#!/bin/sh
#SBATCH -J pytorch #job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=pytorch # application name
#SBATCH -p cas_v100_4 # partition name (queue or class)
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=2 # number of tasks per node
#SBATCH --cpus-per-task=10 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

## Training Resnet-50(Pytorch) for image classification on single node & multi GPUs
Base=/apps/applications/singularity_images
module load ngc/pytorch:22.03-py3

python $Base/examples/pytorch/resnet50v1.5/multiproc.py --nproc_per_node 2 $Base/examples/pytorch/resnet50v1.5/main.py $Base/imagenet \
--data-backend dali-gpu --raport-file report.json -j2 --arch resnet50 -c fanin --label-smoothing 0.1 -b 128 --epochs 50
```
{% endcode %}

***

**4) 스케줄러에 작업을 제출합니다.**

```shell-session
[a1234b5@glogin01]$ sbatch 01.pytorch.sh
Submitted batch job 99982
```

***

**5) 스케줄러에 의해 할당된 계산 노드를 확인합니다.**

```shell-session
[a1234b5@glogin01]$ squque –u a1234b5
 JOBID PARTITION NAME USER STATE TIME TIME_LIMI NODES NODELIST(REASON)
 99982 cas_v100_2 pytorch a1234b5 RUNNING 10:13 24:00:00 1 gpu41
```

***

**6) 스케줄러에 의해 생성되는 로그 파일을 모니터링 합니다.**

```shell-session
[a1234b5@glogin01]$ tail –f pytorch_99982.out
 or 
[a1234b5@glogin01]$ tail –f pytorch_99982.err  
```

![](../.gitbook/assets/NYgTM0g5oP6nCvM.png)

***

**7) 스케줄러에 의해 할당된 계산 노드에서 학습 프로세스 및 GPU 활용 현황을 모니터링 합니다.**

```shell-session
[a1234b5@glogin01]$ ssh gpu41
[a1234b5@gpu41]$ module load nvtop
[a1234b5@gpu41]$ nvtop
```

![](../.gitbook/assets/0AuZDKKItzLxPGz.png)

***

* **작업 스크립트**

#### **1) pytorch 단일 노드 병렬 학습(01.pytorch.sh)**

{% code overflow="wrap" %}
```bash
#!/bin/sh
#SBATCH -J pytorch #job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=pytorch # application name
#SBATCH -p cas_v100_4 # partition name (queue or class)
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=2 # number of tasks per node
#SBATCH --cpus-per-task=10 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

## Training Resnet-50(Pytorch) for image classification on single node & multi GPUs
Base=/apps/applications/singularity_images
module load ngc/pytorch:22.03-py3

python $Base/examples/pytorch/resnet50v1.5/multiproc.py --nproc_per_node 2 $Base/examples/pytorch/resnet50v1.5/main.py $Base/imagenet \
--data-backend dali-gpu --raport-file report.json -j2 --arch resnet50 -c fanin --label-smoothing 0.1 -b 128 --epochs 50

```
{% endcode %}

※ 1노드 점유, 노드 당 2 타스크, 타스크 당 10 CPUs, 노드 당 2GPU 사용



#### **2) pytorch\_horovod 멀티 노드 병렬 학습(02.pytorch\_horovod.sh)**

{% code overflow="wrap" %}
```bash
#!/bin/sh
#SBATCH -J pytorch_horovod # job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=pytorch # application name
#SBATCH -p cas_v100_4 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=2 # number of tasks per node
#SBATCH --cpus-per-task=10 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

## Training Resnet-50(Pytorch horovod) for image classification on multi nodes & multi GPUs
Base=/apps/applications/singularity_images
module load ngc/pytorch:22.03-py3

mpirun_wrapper \
python $Base/examples/horovod/examples/pytorch/pytorch_imagenet_resnet50.py \
--batch-size=128 --epochs=50
```
{% endcode %}

※ 2노드 점유, 노드 당 2 MPI 타스크, 타스크 당 10 CPUs, 노드 당 2GPU 사용

***

#### **3) keras(tensorflow)\_horovod 멀티 노드 병렬 학습(03.keras\_horovod.sh)**

{% code overflow="wrap" %}
```bash
#!/bin/sh
#SBATCH -J keras_horovod # job name
#SBATCH --time=24:00:00 # walltime
#SBATCH --comment=tensorflow # application name
#SBATCH -p cas_v100_4 # partition name (queue or class)
#SBATCH --nodes=2 # the number of nodes
#SBATCH --ntasks-per-node=2 # number of tasks per node
#SBATCH --cpus-per-task=10 # number of cpus per task
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:2 # number of GPUs per node

## Training Resnet-50(Keras horovod) for image classification on multi nodes & multi GPUS
Base=/apps/applications/singularity_images/examples
module load ngc/tensorflow:22.03-tf1-py3

mpirun_wrapper python $Base/horovod/examples/keras/keras_imagenet_resnet50.py \
--batch-size=128 --epochs=50
```
{% endcode %}

※ 2노드 점유, 노드 당 2 MPI 타스크, 타스크 당 10 CPUs, 노드 당 2GPU 사용

{% hint style="info" %}
2023년 3월 2일에 마지막으로 업데이트 되었습니다.
{% endhint %}
