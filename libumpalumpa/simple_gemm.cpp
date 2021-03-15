#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <time.h>


void gemm_cpu(void *buffers[], void *func_arg)
{ 
    //batch size and matrix size can be in func_arg or retrieved from buffers

    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* result = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

//    uint32_t batch = STARPU_VECTOR_GET_NX(buffers[0]); 

    int* arguments = (int*)func_arg;
    int a = arguments[0];
    int b = arguments[1];
    int c = arguments[2];
    int batch = arguments[3];

    for (int i = 0; i < batch; i++) 
        for (int j = 0; j < c; j++)
            for (int k = 0; k < b; k++) {
		float tmp = 0.0;
                for (int l = 0; l < a; l++)
                    tmp += srcA[i*a*b + k + l*b] * srcB[i*c*a + l + j*a];
                result[i*c*b + k + j*b] = tmp;
            }
}

extern "C" void gemm_batch(void *buffers[], void *_args);

struct starpu_codelet gemm =
{
	.where = STARPU_CPU|STARPU_CUDA,
	.cpu_func = gemm_cpu,
	.cuda_func = gemm_batch,
	.cuda_flags = {STARPU_CUDA_ASYNC},
	.nbuffers = 3,
	.modes = {STARPU_R, STARPU_R, STARPU_RW}	
};

#define MAX_MEM 900000000

unsigned switchtime = 0;

int matsize_a = 0;
int matsize_b = 0;
int matsize_c = 0;
int batch = 0;

int batchcount = 0;

void callback_func(void *callback_arg)
{
       
	struct starpu_task *finished_task = (struct starpu_task*) callback_arg;
	
//	float* results = (float*)finished_task->handles[2];

//	printf("%f",results[20]);

}


int main(int argc, char **argv)
{

    matsize_a = 2+(float)(rand())*31 / RAND_MAX;
    matsize_b = 2+(float)(rand())*31 / RAND_MAX;
    matsize_c = 2+(float)(rand())*31 / RAND_MAX;
    batch = ((MAX_MEM/(sizeof(float)*(matsize_a*matsize_b+matsize_c*matsize_a+matsize_c*matsize_b)))/512)*512;

    starpu_init(NULL);

    switchtime = (unsigned)time(NULL);

    for (int a=0; a < 1000; a++)
    {

	unsigned currenttime = (unsigned)time(NULL);    

	if (currenttime - switchtime > 9)
	{
		printf("Batch count: %i\n", batchcount);
		batchcount = 0;
		switchtime = currenttime;
	        
		matsize_a = 2+(float)(rand())*31 / RAND_MAX;
		matsize_b = 2+(float)(rand())*31 / RAND_MAX;
		matsize_c = 2+(float)(rand())*31 / RAND_MAX;
		batch = ((MAX_MEM/(sizeof(float)*(matsize_a*matsize_b+matsize_c*matsize_a+matsize_c*matsize_b)))/512)*512;
		printf("Switching matrix size: A = %i, B = %i, C = %i.\n", matsize_a, matsize_b, matsize_c);
	}

	batchcount++;

	//starpu_task_insert?

        struct starpu_task *newtask = starpu_task_create();

        newtask->cl = &gemm;

	void* matricesA;
	void* matricesB;
	void* results;

	starpu_malloc(&matricesA, matsize_a*matsize_b*batch*sizeof(float));
	starpu_malloc(&matricesB, matsize_c*matsize_a*batch*sizeof(float));
	starpu_malloc(&results, matsize_c*matsize_b*batch*sizeof(float));

	for (size_t i = 0; i < matsize_a*matsize_b*batch; i++)
	    reinterpret_cast<float*>(matricesA)[i] = 10.0f*((float)rand()) / ((float) RAND_MAX);
	for (size_t i = 0; i < matsize_c*matsize_a*batch; i++)
	    reinterpret_cast<float*>(matricesB)[i] = 10.0f*((float)rand()) / ((float) RAND_MAX);
	for (size_t i = 0; i < matsize_c*matsize_b*batch; i++)
	    reinterpret_cast<float*>(results)[i] = 0.0f;

	starpu_data_handle_t bufferA;
	starpu_vector_data_register(&bufferA, STARPU_MAIN_RAM, (uintptr_t)matricesA, matsize_a*matsize_b*batch, sizeof(float));
	newtask->handles[0] = bufferA;

	starpu_data_handle_t bufferB;
        starpu_vector_data_register(&bufferB, STARPU_MAIN_RAM, (uintptr_t)matricesB, matsize_c*matsize_a*batch, sizeof(float));
        newtask->handles[1] = bufferB;

	starpu_data_handle_t resultBuffer;
        starpu_vector_data_register(&resultBuffer, STARPU_MAIN_RAM, (uintptr_t)results, matsize_c*matsize_b*batch, sizeof(float));
        newtask->handles[2] = resultBuffer;

	int arguments[4] = {matsize_a,matsize_b,matsize_c,batch};
        newtask->cl_arg = (void*) arguments;

	newtask->callback_arg = newtask;
	newtask->callback_func = callback_func;

	starpu_task_submit(newtask);
	
	starpu_data_unregister_submit(bufferA);
	starpu_data_unregister_submit(bufferB);
	starpu_data_unregister_submit(resultBuffer);

    }	
	
    starpu_shutdown();

    return 0;
}
