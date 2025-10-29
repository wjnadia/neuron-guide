# 스케줄러(SLURM)를 통한 작업 실행

Neuron 시스템의 작업 스케쥴러는 SLURM을 사용한다. 이 장에서는 SLURM을 통해 작업 제출하는 방법 및 관련 명령어들을 소개한다. SLURM에 작업 제출을 위한 작업 스크립트 작성법은 \[별첨1]과 작업스크립트 파일 작성 예시를 참고하도록 한다.

**※ 사용자 작업은 /scratch/$USER  에서만 제출 가능.**&#x20;



## 가. 큐 구성

<table><thead><tr><th width="212" align="center">Queue Name (cpu_gpu_num)</th><th width="161" align="center">Node Range</th><th width="77" align="center">Max Active Jobs</th><th width="85" align="center">Max Running Jobs</th><th width="75" align="center">Max Avail GPUs</th><th align="center">Total Cores / GPUs</th></tr></thead><tbody><tr><td align="center">cas_v100nv_8</td><td align="center">gpu[01-05]</td><td align="center">6</td><td align="center">5</td><td align="center">40</td><td align="center">32/8</td></tr><tr><td align="center">cas_v100nv_4</td><td align="center">gpu[08-09]</td><td align="center">4</td><td align="center">2</td><td align="center">8</td><td align="center">40/4</td></tr><tr><td align="center">cas_v100_4</td><td align="center">gpu[10-20]</td><td align="center">4</td><td align="center">3</td><td align="center">60</td><td align="center">40/4</td></tr><tr><td align="center">cas_v100_2</td><td align="center">gpu[25-26,29]</td><td align="center">6</td><td align="center">3</td><td align="center">6</td><td align="center">32/2</td></tr><tr><td align="center">amd_a100nv_8</td><td align="center">gpu[30-36]</td><td align="center">3</td><td align="center">2</td><td align="center">56</td><td align="center">64/8</td></tr><tr><td align="center">eme_h200nv_8</td><td align="center">gpu[46-50]</td><td align="center">3</td><td align="center">2</td><td align="center">40</td><td align="center">96/8</td></tr><tr><td align="center">gh200_1</td><td align="center">gpu[51-52]</td><td align="center">2</td><td align="center">1</td><td align="center">2</td><td align="center">72/1</td></tr><tr><td align="center">cpu</td><td align="center">cpu[01-06]</td><td align="center">8</td><td align="center">4</td><td align="center">-</td><td align="center">48/-</td></tr><tr><td align="center">bigmem</td><td align="center">bigmem01</td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center">48/-</td></tr><tr><td align="center">bigmem</td><td align="center">bigmem02</td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center">56/-</td></tr><tr><td align="center">bigmem</td><td align="center">bigmem03</td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center">24/-</td></tr><tr><td align="center">bigmem</td><td align="center">bigmem04</td><td align="center"></td><td align="center"></td><td align="center"></td><td align="center">48/-</td></tr></tbody></table>

* 기본wall clock time 시간: 2일 (48시간)
  * jupyter.ksc.re.kr 에서 submit 되는 jupyter 작업의 wall clock time은 24시간
  * 인터렉티브 작업의 wall clock time은 8시간
* 모든 큐(파티션)에서 작업 할당은 <mark style="color:red;">**공유 노드 정책**</mark>**(하나의 노드에 복수개 작업 동시 수행 가능)**&#xC774; 적용된다.(22.03.17. 이후)  : 자원 활용의 효율성을 위해 기존 배타적 노드 정책에서 공유 노드 정책으로 변경됨
* 작업 큐(파티션)
  * <mark style="color:red;">**일반사용자가 사용할 수 있는 파티션은 jupyter, cas\_v100nv\_8, cas\_v100nv\_4, cas\_v100\_4, cas\_v100\_2, amd\_a100nv\_8, skl, bigmem,  eme\_h200nv\_8,**</mark>**&#x20;**<mark style="color:red;">**gh200\_1**</mark> <mark style="color:red;">**으로 구성되어 있다. (sinfo 명령으로 노드 수, 최대 작업 실행 시간, 노드 리스트 확인 가능)**</mark>
  * <mark style="color:red;">**gh200\_1 파티션의 계산노드는 AArch64 아키텍처로 로그인 노드 및 다른 계산노드의 바이너리 실행 시 오류가 발생하며 ARM\_ 접두어가 붙은 전용 module을 사용하여야 한다.**</mark>
* 작업제출개수제한
  * 사용자별 최대 제출 작업 개수 : 초과하여 작업을 제출한 경우 제출 시점에 에러 발생한다.
  * 사용자별 최대 실행 작업 개수 : 초과하여 작업을 제출한 경우 이전 작업이 끝날 때까지 기다린다.
* 리소스점유제한
  * 작업별 최대 노드 점유 개수 : 초과하여 작업을 제출한 경우 작업이 실행되지 않는다. 사용자의 실행 중인 여러 작업이 한 시점에 점유하고 있는 노드 개수와는 무관하다.
  * 사용자별 최대 GPU 점유 개수 : 사용자별 총 GPU 점유 개수를 제한하는 설정으로 초과하는 경우 이전 작업이 끝날 때까지 기다린다. 사용자의 실행 중인 여러 작업이 한 시점에 점유하고 있는 GPU 개수를 제한한다.

<mark style="color:red;">**※ 노드 구성/파티션은 시스템 사용량에 따라 시스템 운영 중에 조정될 수 있음.**</mark>



## 나. 작업 제출 및 모니터링

### **1. 기본명령어 요약**

| **명령어**               | **내용**                 |
| --------------------- | ---------------------- |
| $ sbatch \[옵션..] 스크립트 | 작업 제출                  |
| $ scancel 작업ID        | 작업 삭제                  |
| $ squeue              | 작업 상태 확인               |
| $ smap                | 작업 상태 및 노드 상태 그래픽으로 확인 |
| $ sinfo \[옵션..]       | 노드 정보 확인               |

※ sinfo --help 명령어를 이용하여 sinfo의 옵션을 확인하실 수 있습니다.

<mark style="color:red;">※</mark> <mark style="color:red;">**Neuron 시스템**</mark> <mark style="color:red;">**사용자 편익 증대를 위한 자료 수집의 목적으로, 아래와 같이 SBATCH 옵션을 통한 사용 프로그램 정보 작성을 의무화한다. 즉, 사용하는 어플리케이션에 맞게 SBATCH의 --comment 옵션을 아래 표를 참조하여 반드시**</mark> <mark style="color:red;">**기입한 후 작업을 제출해야 한다.**</mark>

<mark style="color:red;">**※ 딥러닝 또는 기계학습을 위한 application을 사용하시는 경우 tensorflow, caffe, R, pytorch 등으로 구체적으로 명시해주시기 바랍니다.**</mark>

<mark style="color:red;">**※ 어플리케이션 구분을 추가는 주기적으로 수집된 사용자 요구에 맞추어 진행됩니다. 추가를 원하시면**</mark> [<mark style="color:red;">**consult@ksc.re.kr로**</mark>](mailto:consult@ksc.re.kr%EB%A1%9C) <mark style="color:red;">**해당 어플리케이션에 대한 추가 요청을 해주시기 바랍니다.**</mark>

***

***

**\[Application 별 SBATCH 옵션 이름표]**

| **Application종류** | **SBATCH 옵션 이름** | **Application종류** | **SBATCH 옵션 이름** |
| ----------------- | ---------------- | ----------------- | ---------------- |
| Charmm            | charmm           | LAMMPS            | lammps           |
| Gaussian          | gaussian         | NAMD              | namd             |
| OpenFoam          | openfoam         | Quantum Espresso  | qe               |
| WRF               | wrf              | SIESTA            | siesta           |
| in-house code     | inhouse          | Tensorflow        | tensorflow       |
| PYTHON            | python           | Caffe             | caffe            |
| R                 | R                | Pytorch           | pytorch          |
| VASP              | vasp             | Sklearn           | sklearn          |
| Gromacs           | gromacs          | 그 외 applications  | etc              |

***

***

### **2. 배치 작업 제출**

sbatch 명령을 이용하여 “sbatch {스크립트 파일}” 과 같이 작업을 제출 한다.

```shell-session
$ sbatch [UserJob.script] 
```

* **작업 진행 확인**\
  할당 받은 노드에 접속하여 작업 진행 여부를 확인할 수 있다.

#### 1) squeue 명령어로 진행 중인 작업이 할당된 노드명(NODELIST)을 확인

```shell-session
$ squeue -u [userID]
JOBID PARTITION   NAME   USER   STATE   TIME   TIME_LIMI NODES NODELIST(REASON)
99792 cas_v100_4    ior  userID RUNNING  0:12    5:00:00    1      gpu25
```

#### 2) ssh 명령을 이용하여 해당 노드에 접속

```shell-session
$ ssh gpu25 
```

#### 3) 계산노드에 진입하여 top 또는 nvidia-smi 명령어를 이용하여 작업 진행 여부 조회 가능

※ 2초 간격으로 GPU 사용률을 모니터링하는 예제

```shell-session
$ nvidia-smi -l 2
```

* 작업 스크립트 파일 작성 예시
  * SLURM에서 배치 작업을 수행하기 위해서는 SLURM 키워드들을 사용하여 작업 스크립트 파일을 작성해야 한다.

※ ‘\[별첨1] 작업스크립트 파일 주요 키워드’를 참조

※ 기계학습 프레임워크 Conda 활용은 KISTI 슈퍼컴퓨팅 블로그 ([http://blog.ksc.re.kr/127](http://blog.ksc.re.kr/127)) 참조

* SLURM 키워드

| 키워드                                  | 설명                    |
| ------------------------------------ | --------------------- |
| #SBATCH –J                           | 작업명 지정                |
| #SBATCH --time                       | 최대 작업 수행 시간 지정        |
| #SBATCH –o                           | 작업 로그 파일명 지정          |
| #SBATCH –e                           | 에러 로그 파일명 지정          |
| #SBATCH –p                           | 사용할 파티션 지정            |
| #SBATCH --comment                    | 사용할 애플리케이션명           |
| #SBATCH -–nodelist=(노드 리스트)          | 작업을 수행할 노드 지정         |
| #SBATCH -–nodes=(노드 수)               | 작업을 수행할 노드 수 지정       |
| #SBATCH --ntasks-per-node=(프로세스 수)   | 노드 당 수행될 프로세스 수 지정    |
| #SBATCH --cpus-per-task=(cpu core 수) | 프로세스 당 할당될 cpu core 수 |
| #SBATCH --cpus-per-gpu=(cpu core 수)  | GPU 당 할당될 cpu core 수  |
| #SBATCH --exclusive                  | 노드를 전용으로 사용하기 위한 옵션   |

***

***

* **뉴론 공유 노드 정책에서 메모리 할당량 설정**\
  뉴론 시스템 자원 활용의 효율성 및 사용자의 안정적인 작업 수행을 위하여 아래와 같이 메모리 할당량을 자동 조절

```
memory-per-node = ntasks-per-node * cpus-per-task * (단일 노드 메모리 가용량의 95% / 단일 노드 총 core 수)
```

※ '--exclusive 옵션 사용시에 단일 노드 메모리 가용량의 95%가 작업에 할당되며, 노드를 전용으로 사용할 수 있음. 단, 전용으로 사용가능한 노드가 확보될 때까지 대기 시간이 길어질 수 있음.

* **뉴론 공유 노드 정책에서 GPU 당 CPU core 할당 개수 설정**\
  GPU 어플리케이션의 안정적인 수행을 위해 노드당 CPU core 개수를 GPU에 비례하여 아래와 같이 기본 할당 (메모리 용량도 자동으로 설정, 참조: 뉴론 공유 노드 정책에서 메모리 할당량 설정)

```
cpus-per-gpu = node의 총 core 수 / node의 총 GPU 수 * 요청 GPU 수(--gres=gpu:x)
```

※ 메모리 요구량 추가로 필요한 경우 기본 할당된 cpus-per-gpu 수 보다 크게 자원을 요청하여 메모리 할당량을 확보할 수 있음.



* **CPU Serial 프로그램**

```bash
#!/bin/sh
#SBATCH -J Serial_cpu_job
#SBATCH -p skl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

export OMP_NUM_THREADS=1

module purge
module load intel/19.1.2

srun ./test.exe

exit 0
```

※ 1노드 점유, 순차 사용 예제

* **CPU OpenMP 프로그램**

```bash
#!/bin/sh
#SBATCH -J OpenMP_cpu_job
#SBATCH -p skl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

export OMP_NUM_THREADS=10

module purge
module load intel/19.1.2

mpirun ./test_omp.exe

exit 0
```

※ 1노드 점유, 노드 당 10스레드 사용 예제

* **CPU MPI 프로그램**

```bash
#!/bin/sh
#SBATCH -J MPI_cpu_job
#SBATCH -p skl
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=4 
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

module purge
module load intel/19.1.2 mpi/impi-19.1.2

mpirun ./test_mpi.exe
```

※ 2노드 점유, 노드 당 4 프로세스(총 8 MPI 프로세스) 사용 예제

* **CPU Hybrid (OpenMP+MPI) 프로그램**

```bash
#!/bin/sh
#SBATCH -J hybrid_cpu_job
#SBATCH -p skl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

module purge
module load intel/19.1.2 mpi/impi-19.1.2

export OMP_NUM_THREADS=10

mpirun ./test_mpi.exe
```

※ 1노드 점유, 노드 당 2 프로세스, 프로세스 당 10 스레드(총 2 MPI 프로세스, 20 OpenMP 스레드) 사용 예제

* **GPU Serial 프로그램**

```bash
#!/bin/sh
#SBATCH -J Serial_gpu_job
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1 # using 2 gpus per node
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

export OMP_NUM_THREADS=1

module purge
module load intel/19.1.2 cuda/11.4

srun ./test.exe

exit 0
```

※ 1노드 점유, 순차 사용 예제

* **GPU OpenMP 프로그램**

```bash
#!/bin/sh
#SBATCH -J openmp_gpu_job
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2 # using 2 gpus per node
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

export OMP_NUM_THREADS=10

module purge
module load intel/19.1.2 cuda/11.4

srun ./test_omp.exe

exit 0
```

※ 1노드 점유, 노드 당 10스레드 2GPU 사용 예제

* **GPU MPI 프로그램**

```bash
#!/bin/sh
#SBATCH -J mpi_gpu_job
#SBATCH -p cas_v100_4
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2 # using 2 gpus per node
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

module purge
module load intel/19.1.2 cuda/11.4 cudampi/mvapich2-2.3.6

srun ./test_mpi.exe
```

※ 2노드 점유, 노드 당 4 프로세스(총 8 MPI 프로세스), 노드당 2GPU 사용 예제

* **GPU MPI 프로그램 - 1 node의 모든 CPU를 점유하는 실행예제**

```bash
#!/bin/sh
#SBATCH -J mpi_gpu_job
#SBATCH -p cas_v100_4
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=40
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2 
#SBATCH --comment xxx 

module purge
module load intel/19.1.2 cuda/11.4 cudampi/mvapich2-2.3.6

srun ./test_mpi.exe
```

※ cas\_v100\_4 1개 노드 모든 core 점유, 2GPU 사용 예제

* **GPU MPI 프로그램 - 1 node CPU의 절반만 점유하는 실행 예제**

```bash
#!/bin/sh
#SBATCH -J mpi_gpu_job
#SBATCH -p cas_v100nv_8
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=16
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --comment xxx 

module purge
module load intel/19.1.2 cuda/11.4 cudampi/mvapich2-2.3.6

srun ./test_mpi.exe
```

※ cas\_v100nv\_8 1개 노드의 절반 core 점유, 4 GPU 사용 예제

※ 파티션별 총 core 수는 스케줄러(SLURM)를 통한 작업 실행 > 가. 큐 구성 > Total CPU core 수 참고

* **많은 메모리 할당이 필요한 프로그램 실행 예제**

```bash
#!/bin/sh
#SBATCH -J mem_alloc_job
#SBATCH -p cas_v100_4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --time=01:00:00
#SBATCH --comment xxx #Application별 SBATCH 옵션 이름표 참고

module purge
module load intel/19.1.2 mpi/impi-19.1.2

mpirun -n 2 ./test.exe

exit 0
```

※ **프로그램 실행에 사용할 core 수는 적으나, 메모리 사용량이 큰 경우** 노드당 수행될 프로세스 수로 메모리 할당량을 조절하여 프로그램 실행하는 예제

※'--mem' (노드 당 메모리 할당) 옵션은 사용 불가함. 노드 당 수행될 프로세스 수(ntasks-per-node)와 프로세스 당 할당될 cpu core 수(cpus-per-task)를 입력하면 아래 수식에 따라 메모리 할당량이 자동 계산됨(memory-per-node = ntasks-per-node \* cpus-per-task \* (단일 노드 메모리 가용량의 95% / 단일 노드 총 core 수)

※'--exclusive 옵션 사용시에 단일 노드 메모리 가용량의 95%가 작업에 할당되며, 노드를 전용으로 사용할 수 있음, 단 전용으로 사용가능한 노드가 확보될 때까지 대기 시간이 길어질 수 있음.



### **3. 인터렉티브 작업 제출**

<mark style="color:red;">**※ 인터렉티브 작업 제출 시, 로그인 노드의 환경설정이 전달되므로 gh200\_1 파티션으로 작업 제출할 때는 module purge 이후 작업 제출하거나, 계산노드 접속 후 module purge 후 필요한 ARM module 설정 권장**</mark>

* 자원 할당\
  **cas\_v100\_4** 파티션의 gpu 2노드(각각 2core, 2gpu)를 interactive 용도로 사용

```shell-session
$ salloc --partition=cas_v100_4 --nodes=2 --ntasks-per-node=2 --gres=gpu:2 --comment={SBATCH 옵션이름} #Application별 SBATCH 옵션 이름표 참고
```

<mark style="color:red;">**※ 인터렉티브 작업의 walltime은 8시간으로 고정됨**</mark>

* 작업 실행

```shell-session
$ srun ./(실행파일) (실행옵션)
```

* 진입한 노드에서 나가기 또는 자원 할당 취소

```shell-session
$ exit
```

* 커맨드를 통한 작업 삭제

```shell-session
$ scancel [Job_ID]
```

※ Job ID는 squeue 명령으로 확인 가능

***

***

### **4. 작업 모니터링**

* **파티션 상태 조회**\
  sinfo 명령을 이용하여 조회

```shell-session
$ sinfo
PARTITION      AVAIL  TIMELIMIT  NODES  STATE NODELIST
jupyter           up 2-00:00:00      1    mix jupyter01
jupyter           up 2-00:00:00      1   idle jupyter02
cas_v100nv_8      up 2-00:00:00      4    mix gpu[01-04]
cas_v100nv_4      up 2-00:00:00      2    mix gpu[07,09]
cas_v100nv_4      up 2-00:00:00      1  alloc gpu08
cas_v100_4        up 2-00:00:00      1   plnd gpu18
cas_v100_4        up 2-00:00:00      9    mix gpu[10,12-17,19-20]
cas_v100_4        up 2-00:00:00      1  alloc gpu11
cas_v100_2        up 2-00:00:00      3   idle gpu[25-26,29]
amd_a100nv_8      up 2-00:00:00      1    mix gpu31
amd_a100nv_8      up 2-00:00:00      8  alloc gpu[30,32-37,43,44-45]
eme_h200nv_8      up 2-00:00:00      3   plnd gpu[48-50]
eme_h200nv_8      up 2-00:00:00      2    mix gpu[46-47]
gh200_1           up 2-00:00:00      2   idle gpu[51-52]
skl               up 2-00:00:00      7  alloc skl[01-07]
bigmem            up 2-00:00:00      1    mix bigmem02
bigmem            up 2-00:00:00      1  alloc bigmem01
bigmem            up 2-00:00:00      1   idle bigmem03
```

**※ 노드 구성은 시스템 부하에 따라 시스템 운영 중 조정될 수 있음.**

*   PARTITION : 현재 SLURM에서 설정된 파티션명.

    * AVAIL : 파티션의 상태 (up or down)
    * TIMELIMIT : wall clock time
    * NODES : 노드 수
    * STATE : 노드의 상태 (alloc-자원사용중/Idle-사용가능)
    * NODELIST : 노드 리스트


* **노드별 상세 정보**\
  sinfo 명령 뒤에 "-Nel" 옵션을 사용하면 상세 조회가 가능하다.

```shell-session
$ sinfo -Nel
Fri Mar 18 10:52:13 2022
NODELIST NODES PARTITION    STATE CPUS S:C:T  MEMORY TMP_DISK WEIGHT AVAIL_FE REASON
gpu01    1     cas_v100nv_8 idle  32   2:16:1 384000 0        1      TeslaV10 none
gpu02    1     cas_v100nv_8 idle  32   2:16:1 384000 0        1      TeslaV10 none
gpu03    1     cas_v100nv_8 idle  32   2:16:1 384000 0        1      TeslaV10 none
gpu04    1     cas_v100nv_8 idle  32   2:16:1 384000 0        1      TeslaV10 none
- -  이 하 생 략 - -
```

* **작업 상태 조회**\
  squeue 명령을 이용하여 작업 목록 및 상태를 조회

```shell-session
$squeue
  JOBID PARTITION     NAME         USER   ST     TIME     NODES.      NODELIST(REASON)
  760   cas_v100_4   gpu_burn     userid   R     0:00       10          gpu10
  761   cas_v100_4   gpu_burn     userid   R     0:00       10          gpu11
  762   cas_v100_4   gpu_burn     userid   R     0:00       10          gpu12
```

* **작업 상태확인 및 노드상태 그래픽으로 확인**

```shell-session
$ smap
```

* **제출된 작업 상세 조회**

```shell-session
$ scontrol show job [작업 ID]
```



## 다. 작업 제어

* **작업 삭제(취소)**\
  scancel 명령을 이용하여 “scancel \[Job\_ID]" 과 같이 작업을 삭제 한다.\
  Job\_ID 는 squeue 명령을 이용하여 조회해서 확인한다.

```shell-session
$ scancel 761
```



## 라. 컴파일, 디버깅, 작업제출 위치

* 로그인 노드에서 ssh로 직접 접속이 가능한 디버깅 노드를 제공하고 있음.
* 로그인/디버깅 노드에서 컴파일, 디버깅 및 모든 파티션에 대한 작업 제출이 가능함.
* 디버깅 노드는 CPU time Limit이 120분임.
* 필요시 모든 파티션에서 SLURM Interactive Job 기능을 이용해 컴파일, 디버깅이 가능함

{% hint style="info" %}
2025년 1월 02일에 마지막으로 업데이트되었습니다.
{% endhint %}
