#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <time.h>
#include <Ktt.h>

void gemm_cpu(void *buffers[], void *func_arg)
{ 
    //batch size and matrix size can be in func_arg or retrieved from buffers

    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* result = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

    //uint32_t batch = STARPU_VECTOR_GET_NX(buffers[0]); 

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

struct gemmArgs {
    int matsize_a;
    int matsize_b;
    int matsize_c;
    int batch;
    ktt::Tuner* tuner;
};

struct Codelet {
    starpu_codelet gemm;
    Codelet();
};

Codelet::Codelet():gemm{0}{
    gemm.where = STARPU_CPU|STARPU_CUDA;
    gemm.cpu_funcs[0] = gemm_cpu;
    gemm.cuda_funcs[0] = gemm_batch;
    gemm.cuda_flags[0] = STARPU_CUDA_ASYNC;
    gemm.nbuffers = 3;
    gemm.modes[0] = STARPU_R;
    gemm.modes[1] = STARPU_R;
    gemm.modes[2] = STARPU_RW;


}

Codelet codelet;

#define MAX_MEM 900000000

unsigned switchtime = 0;

int matsize_a = 0;
int matsize_b = 0;
int matsize_c = 0;
int batch = 0;

int batchcount = 0;

//right now, the callback function is useless
void callback_func(void *callback_arg)
{
	struct starpu_task *finished_task = (struct starpu_task*) callback_arg;
}


int main(int argc, char **argv)
{
    //const auto computeAPI = ktt::ComputeAPI::CUDA;
    //ktt::Tuner tuner(0, 0, computeAPI);

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

        struct starpu_task *newtask = starpu_task_create();

        newtask->cl = &codelet.gemm;

        float* matricesA;
        float* matricesB;
        float* results;

        starpu_malloc((void**)&matricesA, matsize_a*matsize_b*batch*sizeof(float));
        starpu_malloc((void**)&matricesB, matsize_c*matsize_a*batch*sizeof(float));
        starpu_malloc((void**)&results, matsize_c*matsize_b*batch*sizeof(float));

        for (size_t i = 0; i < matsize_a*matsize_b*batch; i++)
            matricesA[i] = 10.0f*((float)rand()) / ((float) RAND_MAX);
        for (size_t i = 0; i < matsize_c*matsize_a*batch; i++)
            matricesB[i] = 10.0f*((float)rand()) / ((float) RAND_MAX);
        for (size_t i = 0; i < matsize_c*matsize_b*batch; i++)
            results[i] = 0.0f;

        starpu_data_handle_t bufferA;
        starpu_vector_data_register(&bufferA, STARPU_MAIN_RAM, (uintptr_t)matricesA, matsize_a*matsize_b*batch, sizeof(float));
        newtask->handles[0] = bufferA;

        starpu_data_handle_t bufferB;
        starpu_vector_data_register(&bufferB, STARPU_MAIN_RAM, (uintptr_t)matricesB, matsize_c*matsize_a*batch, sizeof(float));
        newtask->handles[1] = bufferB;

        starpu_data_handle_t resultBuffer;
        starpu_vector_data_register(&resultBuffer, STARPU_MAIN_RAM, (uintptr_t)results, matsize_c*matsize_b*batch, sizeof(float));
        newtask->handles[2] = resultBuffer;

        gemmArgs gemmArg = {matsize_a, matsize_b, matsize_c, batch, nullptr};
        newtask->cl_arg = (void*)&gemmArg;

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
