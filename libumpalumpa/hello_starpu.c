#include <starpu.h>
#include <stdio.h>

int main(void){
	int ret;ret = starpu_init(NULL);
	if (ret != 0){return 1;}
	printf("%d CPU cores\n", starpu_worker_get_count_by_type(STARPU_CPU_WORKER));
	printf("%d CUDA GPUs\n", starpu_worker_get_count_by_type(STARPU_CUDA_WORKER));
	printf("%d OpenCL GPUs\n", starpu_worker_get_count_by_type(STARPU_OPENCL_WORKER));
	starpu_shutdown();
	return 0;
}
