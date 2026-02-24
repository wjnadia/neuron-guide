# Conda 기반 Horovod 설치 방법

Horovod는 고성능 분산 컴퓨팅 환경에서 노드간 메시지 전달 및 통신관리를 위해 일반적인 표준 MPI 모델을 사용하며, Horovod의 MPI구현은 표준 Tensorflow 분산 훈련 모델보다 간소화된 프로그래밍 모델을 제공합니다. NEURON시스템에서도 콘다 환경을 기반으로 멀티노드를 이용한 훈련 모델을 학습시키고자 한다면 다음과 같은 방법으로 설치 후 실행할 수 있습니다.

※ Horovod 사용법은 \[별첨8] 참고 바랍니다.

## 가. Tensorflow-horovod 설치

### 1. 콘다 환경 생성

```shell-session
$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1 cmake/3.16.9
$ conda create -n my_tensorflow
$ source activate my_tensorflow
(my_tensorflow) $
```

※ 자세한 콘다 사용방법은 \[별첨 5] 참고 바랍니다.

### 2. Tensorflow 설치 및 horovod 설치

```shell-session
(my_tensorflow) $ conda install tensorflow-gpu=2.0.0 tensorboard=2.0.0 tensorflow-estimator=2.0.0 python=3.7 cudnn cudatoolkit=10 nccl=2.8.3
(my_tensorflow) $ HOROVOD_WITH_MPI=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_TENSORFLOW=1 \
pip install --no-cache-dir horovod==0.23.0
```

### 3. Horovod 설치 확인

```shell-session
(my_tensorflow) $ pip list | grep horovod
horovod 0.23.0
(my_tensorflow) $ python
>>> import horovod
>>> horovod.__version__
'0.23.0'
```

### 4. Horovod 실행 예시

#### 1) interactive 실행 예시

```shell-session
$ salloc --partition=cas_v100_4 -J debug --nodes=2 --ntasks-per-node=2 --time=08:00:00 --gres=gpu:2 --comment=tensorflow
$ echo $SLURM_NODELIST
gpu[12-13]
$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1
$ source activate my_tensorflow
(my_tensorflow) $ horovodrun -np 4 -H gpu12:2,gpu13:2 python tensorflow2_mnist.py
```

#### 2) batch 스크립트 실행 예시

```bash
#!/bin/bash
#SBATCH -J test_job
#SBATCH -p cas_v100_4
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:2
#SBATCH --comment tensorflow

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1

source activate my_tensorflow

horovodrun -np 2 python tensorflow2_mnist.py
```

## 나. Pytorch-horovod 설치

### 1. 콘다 환경 생성

```shell-session
$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1 cmake/3.16.9
$ conda create -n my_pytorch
$ source activate my_pytorch
(my_pytorch) $ 
```

### 2. Pytorch 설치 및 horovod 설치

```shell-session
(my_pytorch) $ conda install pytorch=1.11.0 python=3.9 torchvision=0.12.0 torchaudio=0.11.0 cudatoolkit=10.2 -c pytorch 
(my_pytorch) $ HOROVOD_WITH_MPI=1 HOROVOD_NCCL_LINK=SHARED HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 \
pip install --no-cache-dir horovod==0.24.0
```

### 3. Horovod 설치 확인

```shell-session
(my_pytorch) $ pip list | grep horovod
horovod 0.24.0

(my_pytorch) $ python
>>> import horovod
>>> horovod.__version__
'0.24.0'
```

### 4. Horovod 실행 예시

#### 1) interactive 실행 예시

```shell-session
$ salloc --partition=cas_v100_4 -J debug --nodes=2 --ntasks-per-node=2 --time=08:00:00 --gres=gpu:2 --comment=pytorch
$ echo $SLURM_NODELIST
gpu[22-23]
$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1 
$ source activate my_pytorch
(my_pytorch) $ horovodrun -np 4 -H gpu22:2,gpu23:2 python pytorch_ex.py
```

#### 2) batch 스크립트 실행 예시

```bash
#!/bin/bash
#SBATCH -J test_job
#SBATCH -p cas_v100_4
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --time 00:30:00
#SBATCH --gres=gpu:2
#SBATCH --comment pytorch

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 python/3.7.1

source activate my_pytorch

horovodrun -np 2 python pytorch_ex.py
```

{% hint style="info" %}
2022년 7월 28일에 마지막으로 업데이트 되었습니다.
{% endhint %}
