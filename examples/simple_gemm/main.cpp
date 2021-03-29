#include <stdio.h>
#include <stdint.h>
#include <starpu.h>
#include <time.h>
#include <Ktt.h>
#include <memory>
#include <fcntl.h>
#include <atomic>
#include <thread>
#include <iostream>
#include <chrono>

using namespace std::chrono_literals;

struct gemmArgs {
    int matsize_a;
    int matsize_b;
    int matsize_c;
    int batch;
    ktt::Tuner* tuner;
};


void fillRandomBytes(void *dst, size_t bytes) {
    int fd = open("/dev/urandom", O_RDONLY);
    read(fd, dst, bytes);
}

void generate_data(void *buffers[], void *func_arg) {
    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    gemmArgs* arguments = (gemmArgs*)func_arg;
    int a = arguments->matsize_a;
    int b = arguments->matsize_b;
    int c = arguments->matsize_c;
    int batch = arguments->batch;
    printf("generate: %d %d %d %d\n", a, b, c, batch);

    fillRandomBytes(srcA, a*b*batch*sizeof(float));
    fillRandomBytes(srcB, a*c*batch*sizeof(float));
}

void gemm_cpu(void *buffers[], void *func_arg)
{ 
    //batch size and matrix size can be in func_arg or retrieved from buffers

    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* result = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

    //uint32_t batch = STARPU_VECTOR_GET_NX(buffers[0]); 

    gemmArgs* arguments = (gemmArgs*)func_arg;
    const int a = arguments->matsize_a;
    const int b = arguments->matsize_b;
    const int c = arguments->matsize_c;
    const int batch = arguments->batch;
    printf("process: %d %d %d %d\n", a, b, c, batch);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < b; k++) {
		        float tmp = 0.0;
                for (int l = 0; l < a; l++) {
                    const size_t offsetA = i*a*b + k + l*b;
                    const size_t offsetB = i*c*a + l + j*a;
                    tmp += srcA[offsetA] * srcB[offsetB];
                }
                result[i*c*b + k + j*b] = tmp;
            }
        }
    }
}

extern "C" void gemm_batch(void *buffers[], void *_args);


struct Codelet {
    starpu_codelet generate;
    starpu_codelet gemm;
    Codelet();
};

Codelet::Codelet():gemm{0}, generate{0}{
    gemm.where = STARPU_CPU|STARPU_CUDA;
    gemm.cpu_funcs[0] = gemm_cpu;
    gemm.cuda_funcs[0] = gemm_batch;
    gemm.cuda_flags[0] = STARPU_CUDA_ASYNC;
    gemm.nbuffers = 3;
    gemm.modes[0] = STARPU_R;
    gemm.modes[1] = STARPU_R;
    gemm.modes[2] = STARPU_W;
    gemm.name="gemm codelet";

    generate.where = STARPU_CPU;
    generate.cpu_funcs[0] = generate_data;
    generate.nbuffers = 2;
    generate.modes[0] = STARPU_W;
    generate.modes[1] = STARPU_W;
    generate.name="matrix generation codelet";
}

Codelet codelet;

#define MAX_BYTES (512 * 1024 * 1024)

unsigned switchtime = 0;
int batchcount = 0;

std::atomic<int> running_batches = 0;


//right now, the callback function is useless
void callback_func(void *callback_arg)
{
	// struct starpu_task *finished_task = (struct starpu_task*) callback_arg;
    running_batches--;
    std::cout << "Concurrently running batches: " << running_batches << "\n";
}




int main(int argc, char **argv)
{
    starpu_init(NULL);

    constexpr size_t timeout = 5;
    switchtime = (unsigned)time(NULL) - timeout;

    gemmArgs arguments;
    size_t max_config = 5;
    for (int config_id=0; config_id < max_config;)
    {
         if (running_batches > 10) {
            std::this_thread::sleep_for(200ms);
        }
        running_batches++;
        unsigned currenttime = (unsigned)time(NULL);    

        if (currenttime - switchtime >= timeout)
        {
            printf("Processed batches: %i\n", batchcount);
            batchcount = 0;
            config_id++;
            switchtime = currenttime;

            auto *args = &arguments;
            args->matsize_a = 2+(float)(rand())*31 / RAND_MAX;
            args->matsize_b = 2+(float)(rand())*31 / RAND_MAX;
            args->matsize_c = 2+(float)(rand())*31 / RAND_MAX;
            args->batch = (MAX_BYTES / sizeof(float)) / ((args->matsize_a*args->matsize_b)+(args->matsize_c*args->matsize_a)+(args->matsize_c*args->matsize_b));
            printf("Switching matrix size: A = %i, B = %i, C = %i., batch=%i\n", args->matsize_a, args->matsize_b, args->matsize_c, args->batch);
        }

        auto *args = new gemmArgs(arguments);
        batchcount++;
        starpu_data_handle_t matrixA = {0};
        starpu_vector_data_register(&matrixA, -1, 0, args->matsize_a*args->matsize_b*args->batch, sizeof(float));
        starpu_data_set_name(matrixA, "Matrix A");

        starpu_data_handle_t matrixB = {0};
        starpu_vector_data_register(&matrixB, -1, 0, args->matsize_c*args->matsize_a*args->batch, sizeof(float));
        starpu_data_set_name(matrixB, "Matrix B");

        struct starpu_task *generateDataTask = starpu_task_create();
        generateDataTask->handles[0] = matrixA;
        generateDataTask->handles[1] = matrixB;
        generateDataTask->cl_arg = args;
        generateDataTask->cl_arg_size = sizeof(gemmArgs);
        generateDataTask->cl = &codelet.generate;
        generateDataTask->name = "Generate data task";
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(generateDataTask), "starpu_task_submit generateDataTask");

        starpu_data_handle_t resultBuffer = {0};
        starpu_vector_data_register(&resultBuffer, -1, 0, args->matsize_c*args->matsize_b*args->batch, sizeof(float));
        starpu_data_set_name(resultBuffer, "Result matrix");

        struct starpu_task *newtask = starpu_task_create();
        newtask->handles[0] = matrixA;
        newtask->handles[1] = matrixB;
        newtask->handles[2] = resultBuffer;
        newtask->cl = &codelet.gemm;
        newtask->cl_arg = args;
        newtask->cl_arg_size = sizeof(gemmArgs);
        newtask->callback_func = callback_func;
        newtask->name = "Compute GEMM task";
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(newtask), "starpu_task_submit newtask");  

        starpu_data_unregister_submit(matrixA);
        starpu_data_unregister_submit(matrixB);
        starpu_data_unregister_submit(resultBuffer);
    }	
	
    starpu_shutdown();

    return 0;
}
