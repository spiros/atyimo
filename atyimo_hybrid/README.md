# Atyimo - Record Linkage Application for Heterogeneous Platforms

## FEDERAL UNIVERSITY OF BAHIA (UFBA)
## ATYIMOLAB (www.atyimolab.ufba.br)
## University College London
## Denaxas Lab (www.denaxaslab.org)
## Robespierre Pita and Clicia Pinto and Marcos Barreto and Spiros Denaxas

Contextualization: Examining different databases to search for records that represent the same real world entity is a problem known as Record Linkage. As this is a high time-consuming computational task, this project has developed different solutions based on **heterogeneous computing systems** that offer high computational power, such as:

- [OpenMP]
- [CUDA C]

### Instructions
    To execute this application, you need to copy .bloom files into the directory where the executable file is.
    Bloom files need to follow this pattern: input_500000.bloom being "500000" the exact number of records in this file.

### To execute

* OpenMP code:

    To compile the OpenMP code, it is necessary to put the **-fopenmp** directive in the compilation command. This allows the OpenMP code to use as many threads as available cores in the computer.

    ```sh
    $ cd openmp
    $ gcc -fopenmp linkage.c -o linkage
    $ ./linkage <problem_size> <num_threads>
    ```
    Example:

    ```sh
    $ ./linkage 500000 16
    ```

* CUDA C code:

    To run the CUDA code, two prerequisites are necessary: your computer must have a **nvidia-capable device** and the **NVIDIA CUDA Compiler** ([NVCC](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4Rnk5ZlXr)).

    * single-GPU:

        ```sh
        $ cd cuda-c/one-GPU
        $ make clean
        $ make
        $ ./linkage <num_threads_per_block> <larger_file>
        ```
        Example:    
        ```sh
        $ ./linkage 32 500000
        ```

    * multi-GPU:

        ```sh
        $ cd cuda-c/multi-GPU
        $ make clean
        $ make
        $ ./linkage <num_threads_per_block> <file1> <threads_openmp> <percentage_each_gpu> <qtd_gpu>
        ```

        Example:    
        ```sh
        $ ./linkage 32 500000 16 25 2
        ```

#### Recommended Software Versions

> gcc 5.4.0 or higher (native suport for openmp).
> CUDA toolkit 4.0 or higher.

[OpenMP]: <http://www.openmp.org/>
[CUDA C]: <http://www.nvidia.com/object/cuda_home_new.html>
