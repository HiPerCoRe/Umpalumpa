# Umpalumpa

Umpalumpa is a framework which aims to manage complex workloads on heterogeneous computers. 
Umpalumpa combines three aspects that ease programming and optimize code performance. 

Firstly, it implements data-centric design, where data are described by their physical properties (e. g. , location in memory, size) and logical properties (e.g. , dimensionality, shape, padding). 

Secondly, Umpalumpa utilizes task-based parallelism to schedule tasks on heterogeneous nodes. 

Thirdly, tasks can be dynamically autotuned on a source code level according to the hardware where the task is executed and the processed data. 

Altogether, Umpalumpa allows for the implementation of a complex workload, which is automatically executed on CPUs and accelerators, and which allows autotuning to maximize the performance with the given hardware and data input. 
Umpalumpa focuses on image processing workloads, but the concept is generic and can be extended to different types of workloads.

We demonstrated the usability of the proposed framework on two previously accelerated applications from cryogenic electron microscopy: 3D Fourier reconstruction and Movie alignment. 
Compared to the original implementations, Umpalumpa reduces the complexity and improves the maintainability of the main applications' loops while improving performance through automatic memory management and autotuning of the GPU kernels.

## Publication
_Submitted_

## Dependencies
Cmake 3.14 or newer

CUDA

FFTW

C++17 compatible compiler

StarPU (optional but highly recommended)

[spdlog](https://github.com/gabime/spdlog) (automatically fetched)

[KTT](https://github.com/HiPerCoRe/KTT) (automatically fetched)

## Instalation

```
source path_to_starpu_installation/bin/starpu_env
export CUDACXX=path_to_CUDA_installation/bin/nvcc
mkdir build
cd build
cmake ..
make
```

## Keywords
image processing, task-based systems, auto-tuning, data-aware architecture, CUDA
