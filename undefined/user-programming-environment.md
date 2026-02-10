# 사용자 프로그래밍 환경

## 가. 프로그래밍 도구 설치 현황

|                                    구분                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                        항목                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| :----------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                   컴파일러                                   |                                                                                                                                                                                                                                                                                                                                                                                            <ul><li>gcc/4.8.5</li><li>gcc/10.2.0</li><li>intel/19.1.2</li><li>nvidia_hpc_sdk/22.7</li><li>nvidia_hpc_sdk/24.1</li></ul>                                                                                                                                                                                                                                                                                                                                                                                           |
|                                 MPI 라이브러리                                |                                                                                                                                                                                                                                                                                                                                                    <ul><li>cudampi/mvapich2-2.3.6</li><li>cudampi/openmpi-3.1.5</li><li>cudampi/openmpi-4.1.1</li><li>mpi/impi-19.1.2</li><li>mpi/mvapich2-2.3.6</li><li>mpi/openmpi-3.1.5</li><li>mpi/openmpi-4.1.1</li></ul>                                                                                                                                                                                                                                                                                                                                                   |
|                                   라이브러리                                  |                                                                                                                                                                                                                                                                                                                                               <ul><li>common/nccl-rdma-sharp-2.1.0</li><li>hdf4/4.2.13</li><li>hdf5/1.10.2</li><li>hdf5/1.12.0</li><li>lapack/3.7.0</li><li>lapack/3.10.0 </li><li>libxc/4.3.4</li><li>libxc/5.2.3 </li><li>netcdf/4.6.1</li></ul>                                                                                                                                                                                                                                                                                                                                               |
|                               MPI 의존 라이브러리                               |                                                                                                                                                                                                                                                                                                                                                                                                                              <ul><li>fftw_mpi/2.1.5</li><li>fftw_mpi/3.3.7</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                             |
|                                CUDA 라이브러리                                |                                                                                                                                                                                                                                                                                                                                                                                                                                   <ul><li>cuda/11.4</li><li>cuda/12.3</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|                      CUDA 라이브러리(CUDA 라이브러리만 설치된 버전)                      |                                                                                                                                                                                                                                                                                                                                <ul><li>cuda/10.0</li><li>cuda/10.1</li><li>cuda/10.2</li><li>cuda/11.0</li><li>cuda/11.1</li><li>cuda/11.2</li><li>cuda/11.3</li><li>cuda/11.5</li><li>cuda/11.6</li><li>cuda/11.7</li><li>cuda/11.8</li><li>cuda/12.1</li><li>cuda/12.2</li></ul>                                                                                                                                                                                                                                                                                                                               |
|                                   소프트웨어                                  | <ul><li>R/3.5.0</li><li>R/4.2.1</li><li>cmake/3.16.9</li><li>cmake/3.26.2</li><li>cotainr/2023.11.0</li><li>gaussian/g16.b01</li><li>gaussian/g16.c01</li><li>gaussian/g16.c02</li><li>gaussian/g16</li><li>git/2.9.3</li><li>git/2.35.1</li><li>gnu-parallel/2022.03.22</li><li>gromacs/2021.3</li><li>gromacs/2023.3</li><li>htop/3.0.5</li><li>java/openjdk-11.0.1</li><li>java/openjdk-23</li><li>lammps/23Jun2022_a100</li><li>lammps/23Jun2022_v100</li><li>lammps/27Oct2021_a100</li><li>lammps/27Oct2021_v100</li><li>namd/2.14</li><li>ncl/6.6.2</li><li>nvtop/1.1.0</li><li>python/3.7.1 </li><li>python/3.9.5</li><li>python/3.12.4</li><li>qe/7.0_a100</li><li>qe/7.0_v100</li><li>qe/7.2_a100</li><li>qe/7.2_v100</li><li>qe/7.3_a100 </li><li>qe/7.3_v100</li><li>singularity/3.6.4</li><li>singularity/3.9.7</li><li>singularity/3.11.0</li><li>singularity/4.1.0</li><li>MATLAB/R2024a</li></ul> |
|                                    ngc                                   |                                                                                                                                                                                                                                                                                 <ul><li>ngc/caffe:20.03-py3</li><li>ngc/gromacs:2020.2</li><li>ngc/lammps:29Oct2020</li><li>ngc/paraview:5.9.0-py3</li><li>ngc/pytorch:20.09-py3</li><li>ngc/pytorch:20.12-py3</li><li>ngc/pytorch:22.03-py3</li><li>ngc/pytorch:23.12-py3</li><li>ngc/qe:6.7</li><li>ngc/tensorflow:22.03-tf1-py3</li><li>ngc/tensorflow:22.03-tf2-py3</li></ul>                                                                                                                                                                                                                                                                                |
|                               conda 패키지 환경                               |                                                                                                                                                                                                                                                                                                                                                          <ul><li>conda/pytorch_1.11.0</li><li>conda/pytorch_1.12.0</li><li>conda/pytorch_2.5.0</li><li>conda/tensorflow_2.4.1</li><li>conda/tensorflow_2.10.0</li><li>conda/tensorflow_2.16.1</li></ul>                                                                                                                                                                                                                                                                                                                                                          |
| <p>AArch64 아키텍처 <br><mark style="color:red;">(gh200_1 파티션 전용)</mark></p> |                                                                                                                                                                                                                                                                       <ul><li>ARM_cmake/3.26.2</li><li>ARM_cuda/12.3</li><li>ARM_cudampi/mvapich2-2.3.6</li><li>ARM_cudampi/openmpi-4.1.1 </li><li>ARM_fftw_mpi/3.3.7</li><li>ARM_gcc/11.4.1</li><li>ARM_gromacs/2023.3</li><li>ARM_htop/3.0.5</li><li>ARM_lammps/23Jun2022</li><li>ARM_nvidia_hpc_sdk/24.1</li><li>ARM_nvtop/1.1.0</li><li>ARM_python/3.12.4</li><li>ARM_qe/7.3</li></ul>                                                                                                                                                                                                                                                                       |

* conda/pytorch\_1.11.0, conda/tensorflow\_2.4.1 module은 pytorch/tensorflow 사용을 위한 라이브러리가 설치 되어있는 module로, conda 명령을 사용하려면 python/3.7.1 module 적용 후 conda 명령을 사용해야 합니다.
* Neuron시스템에서의 인공지능 프레임워크는 conda 환경을 사용하는 것을 권장합니다. (라이선스 조건 확인, conda-forge 채널활용)
* 사용자 요구 기반의 Singularity 컨테이너 이미지를 빌드 및 구동하여 사용자 프로그램을 실행할 수 있습니다.&#x20;
  * \[별첨3] Singularity 컨테이너 사용법’을 참조 바랍니다.
* MATLAB은 **사용자(소속기관)가 라이선스를 보유해야 합니다**. 사용법은 [MATLAB 가이드](https://docs-ksc.gitbook.io/myksc/app/matlab) 참조 바랍니다.

<mark style="color:red;">**※ GPU가 장착되지 않은 노드를 사용하기 위해서는 None CUDA MPI 라이브러리(예 :  mpi/openmpi-4.1.1) 모듈을 사용해야 합니다.**</mark>

<mark style="color:red;">**※ gh200\_1 파티션은 AArch64 아키텍처로 ARM\_ 접두어가 붙은 전용 module을 활용해야 합니다.**</mark> &#x20;

* <mark style="color:red;">**gh200\_1 계산노드에서는**</mark>**&#x20;**<mark style="color:red;">**x86\_64용 바이너리 실행 시 오류 발생합니다.**</mark>
* <mark style="color:red;">**x86\_64 아키텍처(뉴론 로그인 노드, gh200\_1 파티션을 제외한 모든 계산노드)에서**</mark>**&#x20;**<mark style="color:red;">**AArch64용 바이너리 실행 시 오류 발생합니다.**</mark>



## 나. 컴파일러 사용법

### 1. 컴파일러 및 MPI 환경설정(modules)

#### 1) 모듈 관련 기본 명령어

<mark style="color:blue;">**※ 사용자 편의를 위해 "module" 명령어는 "ml" 명령어로 축약하여 사용할 수 있습니다.**</mark>

* 사용 가능한 모듈 목록 출력\
  사용할 수 있는 컴파일러, 라이브러리 등의 모듈 목록을 확인할 수 있습니다.

```shell-session
$ module avail
혹은
$ module av
```

* 사용할 모듈 추가\
  사용하고자 하는 컴파일러, 라이브러리 등의 모듈을 추가할 수 있습니다.\
  사용할 모듈들을 한번에 추가할 수 있습니다.

```shell-session
$ module load [module name] [module name] [module name] ...
혹은
$ module add [module name] [module name] [module name] ...
혹은
$ ml [module name] [module name] [module name] ...  
ex) ml gcc/4.8.5 singularity/3.9.7 
```

* 사용 모듈 삭제\
  필요 없는 모듈을 제거합니다. 이 때 한번에 여러 개의 모듈을 삭제할 수 있습니다.

```shell-session
$ module unload [module name] [module name] [module name] ...
혹은
$ module rm [module name] [module name] [module name] ...
혹은
$ ml -[module name] -[module name] -[module name] ...     
ex) ml -gcc/4.8.5 -singularity/3.9.7 
```

* 사용 모듈 목록 출력\
  현재 설정된 모듈 목록을 확인할 수 있습니다.

```shell-session
$ module list
혹은
$ module li
혹은
$ ml
```

* 전체 사용 모듈 일괄 삭제

```shell-session
$ module purge
```

* 모듈 설치 경로 확인

```shell-session
$ module show [module name]
```

* 모듈 찾기

```shell-session
$ module spider  [module | string | name/version ]
```

* 사용자 모듈 모음(collection) 저장 관리

```shell-session
# 현재 로드된 모듈들을 default 모듈 모음에 저장하며, 다음 로그인 시 자동 로드
$ module save  
# 현재 로드된 모듈들을 지정된 이름을 가진 사용자 모듈 모음으로 저장함                  
$ module save [name]  
# 사용자 모듈 모음을 로드함        
$ module restore [name]    
# 사용자 모듈 모음의 내용을 출력함  
$ module describe [name]    
# 사용자 모듈 모음 리스트를 출력함  
$ module savelist  
# 사용자 모듈 모음을 삭제함             
$ module disable [name]     
```

### 2. 순차 프로그램 컴파일

순차 프로그램은 병렬 프로그램 환경을 고려하지 않은 프로그램을 말합니다. 즉, OpenMP, MPI와 같은 병렬 프로그램 인터페이스를 사용하지 않는 프로그램으로써, 하나의 노드에서 하나의 프로세서만 사용해 실행되는 프로그램입니다. 순차 프로그램 컴파일 시 사용되는 컴파일러별 옵션은 병렬 프로그램을 컴파일 할 때도 그대로 사용되므로, 순차 프로그램에 관심이 없다 하더라도 참조하는 것이 좋습니다.

#### 1) Intel 컴파일러

Intel 컴파일러를 사용하기 위해서 필요한 버전의 Intel 컴파일러 모듈을 추가하여 사용합니다. 사용 가능한 모듈은 module avail로 확인할 수 있습니다.

```shell-session
$ module load intel/18.0.2
```

※ 프로그래밍 도구 설치 현황 표를 참고하여 사용 가능 버전 확인하시기 바랍니다.

* 컴파일러 종류



| 컴파일러       | 프로그램    | 소스 확장자                                                 |
| ---------- | ------- | ------------------------------------------------------ |
| icc / icpc | C / C++ | .C, .cc, .cpp, .cxx,.c++                               |
| ifort      | F77/F90 | .f, .for, .ftn, .f90, .fpp, .F, .FOR, .FTN, .FPP, .F90 |



* Intel 컴파일러 사용 예제\
  다음은 test 예제파일을 intel 컴파일러로 컴파일하여 실행파일 test.exe를 만드는 예시입니다.

```shell-session
$ module load intel/19.1.2
$ icc -o test.exe test.c
혹은
$ ifort -o test.exe test.f90
$ ./test.exe
```

※ /apps/shell/job\_examples 에서 작업제출 test 예제파일을 복사하여 사용 가능합니다.

#### **2) GNU 컴파일러**

GNU 컴파일러를 사용하기 위해서 필요한 버전의 GNU 컴파일러 모듈을 추가하여 사용합니다. 사용 가능한 모듈은 module avail로 확인할 수 있습니다.

```shell-session
$ module load gcc/10.2.0
```

※ 프로그래밍 도구 설치 현황 표를 참고하여 사용가능 버전 확인하시기 바랍니다.

* 컴파일러 종류

| 컴파일러      | 프로그램    | 소스 확장자                                                 |
| --------- | ------- | ------------------------------------------------------ |
| gcc / g++ | C / C++ | .C, .cc, .cpp, .cxx,.c++                               |
| gfortran  | F77/F90 | .f, .for, .ftn, .f90, .fpp, .F, .FOR, .FTN, .FPP, .F90 |



* GNU 컴파일러 사용 예제

다음은 test 예제파일을 GNU 컴파일러로 컴파일하여 실행파일 test.exe를 만드는 예시입니다.

```shell-session
$ module load gcc/10.2.0
$ gcc -o test.exe test.c
혹은
$ gfortran -o test.exe test.f90
$ ./test.exe
```

※ /apps/shell/job\_examples 에서 작업제출 test 예제파일을 복사하여 사용 가능합니다.

#### 3) PGI 컴파일러

PGI 컴파일러를 사용하기 위해서 필요한 버전의 PGI 컴파일러 모듈을 추가하여 사용합니다. 사용 가능한 모듈은 module avail로 확인할 수 있습니다.

```shell-session
$ module load nvidia_hpc_sdk/22.7
```

※ 프로그래밍 도구 설치 현황 표를 참고하여 사용가능 버전 확인하시기 바랍니다.

* **컴파일러 종류**

| 컴파일러         | 프로그램    | 소스 확장자                                                 |
| ------------ | ------- | ------------------------------------------------------ |
| pgcc / pgc++ | C / C++ | .C, .cc, .cpp, .cxx,.c++                               |
| pgfortran    | F77/F90 | .f, .for, .ftn, .f90, .fpp, .F, .FOR, .FTN, .FPP, .F90 |



* **PGI 컴파일러 사용 예제**

다음은 test 예제파일을 PGI 컴파일러로 컴파일하여 실행파일 test.exe를 만드는 예시입니다.

```shell-session
$ module load pgi/19.1
$ pgcc -o test.exe test.c
혹은
$ pgfortran -o test.exe test.f90
$ ./test.exe
```

※ /apps/shell/job\_examples 에서 작업제출 test 예제파일을 복사하여 사용 가능합니다.

####

### 3. 병렬 프로그램 컴파일

#### 1) OpenMP 컴파일

OpenMP는 컴파일러 지시자만으로 멀티 스레드를 활용할 수 있도록 간단하게 개발된 기법으로 OpenMP를 사용한 병렬 프로그램 컴파일 시 사용되는 컴파일러는 순차프로그램과 동일하며, 컴파일러 옵션을 추가하여 병렬 컴파일을 할 수 있는데, 현재 대부분의 컴파일러가 OpenMP 지시자를 지원합니다.

| 컴파일러 옵션                  | 프로그램              | 옵션       |
| ------------------------ | ----------------- | -------- |
| icc / icpc / ifort       | C / C++ / F77/F90 | -qopenmp |
| gcc / g++ / gfortran     | C / C++ / F77/F90 | -fopenmp |
| pgcc / pgc++ / pgfortran | C / C++ / F77/F90 | -mp      |



* **OpenMP 프로그램 컴파일 예시 (Intel 컴파일러)**

다음은 **openMP**를 사용하는 test\_omp 예제파일을 intel 컴파일러로 컴파일하여 실행파일 test\_omp.exe를 만드는 예시입니다.

```shell-session
$ module load intel/19.1.2
$ icc -o test_omp.exe -qopenmp test_omp.c
혹은
$ ifort -o test_omp.exe -qopenmp test_omp.f90
$ ./test_omp.exe
```

* **OpenMP 프로그램 컴파일 예시 (GNU 컴파일러)**

다음은 **openMP**를 사용하는 test\_omp 예제파일을 GNU 컴파일러로 컴파일하여 실행파일 test\_omp.exe를 만드는 예시입니다.

```shell-session
$ module load gcc/10.2.0
$ gcc -o test_omp.exe -fopenmp test_omp.c
혹은
$ gfortran -o test_omp.exe -fopenmp test_omp.f90
$ ./test_omp.exe
```

* **OpenMP 프로그램 컴파일 예시 (PGI 컴파일러)**

다음은 **openMP**를 사용하는 test\_omp 예제파일을 PGI 컴파일러로 컴파일하여 실행파일 test\_omp.exe를 만드는 예시입니다.

```shell-session
$ module load nvidia_hpc_sdk/22.7
$ pgcc -o test_omp.exe -mp test_omp.c
혹은
$ pgfortran -o test_omp.exe -mp test_omp.f90
$ ./test_omp.exe
```



#### 2) MPI 컴파일

사용자는 다음 표의 MPI 명령을 실행할 수 있는데, 이 명령은 일종의 wrapper로써 .bashrc를 통해 지정된 컴파일러가 소스를 컴파일하게 됩니다.

| **구분**        | **Intel** | **GNU**  | **PGI**   |
| ------------- | --------- | -------- | --------- |
| Fortran       | ifort     | gfortran | pgfortran |
| Fortran + MPI | mpiifort  | mpif90   | mpif90    |
| C             | icc       | gcc      | pgcc      |
| C + MPI       | mpiicc    | mpicc    | mpicc     |
| C++           | icpc      | g++      | pgc++     |
| C++ + MPI     | mpiicpc   | mpicxx   | mpicxx    |

mpicc로 컴파일을 하더라도, 옵션은 wrapping되는 본래의 컴파일러에 해당하는 옵션을 사용해야 합니다.

* &#x20;**MPI 프로그램 컴파일 예시 (Intel 컴파일러)**

다음은 **MPI**를 사용하는 test\_mpi 예제파일을 intel 컴파일러로 컴파일하여 실행파일 test\_mpi.exe를 만드는 예시입니다.

```shell-session
$ module load intel/19.1.2 mpi/impi-19.1.2
$ mpiicc -o test_mpi.exe test_mpi.c
혹은
$ mpiifort -o test_mpi.exe test_mpi.f90
$ srun ./test_mpi.exe
```

* **MPI 프로그램 컴파일 예시 (GNU 컴파일러)**

다음은 **MPI**를 사용하는 test\_mpi 예제파일을 GNU 컴파일러로 컴파일하여 실행파일 test\_mpi.exe를 만드는 예시입니다.

```shell-session
$ module load gcc/10.2.0 mpi/openmpi-4.1.1
$ mpicc -o test_mpi.exe test_mpi.c
혹은
$ mpif90 -o test_mpi.exe test_mpi.f90
$ srun ./test_mpi.exe
```

* **MPI 프로그램 컴파일 예시 (PGI 컴파일러)**

다음은 **MPI**를 사용하는 test\_mpi 예제파일을 PGI 컴파일러로 컴파일하여 실행파일 test\_mpi.exe를 만드는 예시입니다.

```shell-session
$ module load nvidia_hpc_sdk/22.7
$ mpicc -o test_mpi.exe test_mpi.c
혹은
$ mpifort -o test_mpi.exe test_mpi.f90
$ srun ./test_mpi.exe
```

* **CUDA + MPI 프로그램 컴파일 예시**&#x20;

```shell-session
$ module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1
$ mpicc -c mpi-cuda.c -o mpi-cuda.o
$ mpicc mpi-cuda.o -lcudart -L/apps/cuda/11.4/lib64
$ srun ./a.out
```

※ intel 컴파일러 사용 시, gcc/10.2.0 대신 intel/19.1.2 module을 load을 적용합니다.<br>

{% hint style="info" %}
2024년 11월 22일에 마지막으로 업데이트 되었습니다.
{% endhint %}
