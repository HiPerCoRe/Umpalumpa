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
    const ktt::KernelId* kernel;
    const ktt::KernelDefinitionId* kernelDefinition;
	bool tuningStep;
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

    fillRandomBytes(srcA, a*b*batch*sizeof(float));
    fillRandomBytes(srcB, a*c*batch*sizeof(float));
}


void gemm_cuda(void *buffers[], void *_args)
{	
    float* A = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* B = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* C = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

    gemmArgs* arguments = (gemmArgs*)_args;
    int matsize_a = arguments->matsize_a;
    int matsize_b = arguments->matsize_b;
    int matsize_c = arguments->matsize_c;
    int batch = arguments->batch;
    ktt::Tuner* tuner = arguments->tuner;
    const ktt::KernelId* kernel = arguments->kernel;
	const ktt::KernelDefinitionId* kernelDefinition = arguments->kernelDefinition;
	bool tuningStep = arguments->tuningStep;

    printf("CUDA processing: %d %d %d %d\n", matsize_a, matsize_b, matsize_c, batch);

    int bufferSizeA = matsize_a*matsize_b*batch;
    int bufferSizeB = matsize_c*matsize_a*batch;
    int bufferSizeC = matsize_c*matsize_b*batch;
    const ktt::ArgumentId aId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(A), bufferSizeA,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId bId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(B), bufferSizeB,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId resultId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(C), bufferSizeC,
        ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Host);

    const ktt::ArgumentId matsize_a_Id = tuner->AddArgumentScalar(matsize_a);
    const ktt::ArgumentId matsize_b_Id = tuner->AddArgumentScalar(matsize_b);
    const ktt::ArgumentId matsize_c_Id = tuner->AddArgumentScalar(matsize_c);
    const ktt::ArgumentId batchId = tuner->AddArgumentScalar(batch);

	tuner->SetArguments(*kernelDefinition, {aId, bId, resultId, batchId});
	

	tuner->SetLauncher(*kernel, [kernelDefinition, batch, matsize_c](ktt::ComputeInterface& interface)
	{
		auto config = interface.GetCurrentConfiguration().GetPairs();
		size_t padd_c = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "PADD_C");
		size_t group_size_y = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "GROUP_SIZE_Y");
		size_t group_size_z = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "GROUP_SIZE_Z");
		
		const ktt::DimensionVector newGlobalSize(batch/group_size_z,1,1); 
		const ktt::DimensionVector newlocalSize(matsize_c+padd_c,group_size_y,group_size_z);
		
		interface.RunKernel(*kernelDefinition, newGlobalSize, newlocalSize);
	});
	
	if (tuningStep)
	{
		tuner->TuneKernelIteration(*kernel, {ktt::BufferOutputDescriptor(resultId, C)});
		printf("TUNING\n");
	}
	else
	{
		auto bestConfig = tuner->GetBestConfiguration(*kernel);
		tuner->RunKernel(*kernel, {bestConfig}, {ktt::BufferOutputDescriptor(resultId, C)});
		printf("RUNNING\n");
	}
	
}

void gemm_cpu(void *buffers[], void *func_arg)
{ 

    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* result = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

    gemmArgs* arguments = (gemmArgs*)func_arg;
    const int a = arguments->matsize_a;
    const int b = arguments->matsize_b;
    const int c = arguments->matsize_c;
    const int batch = arguments->batch;

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

struct Codelet {
    starpu_codelet generate;
    starpu_codelet gemm;
    Codelet();
};

Codelet::Codelet():gemm{0}, generate{0}{
    gemm.where = STARPU_CPU|STARPU_CUDA;
    // gemm.cpu_funcs[0] = gemm_cpu;
    gemm.cuda_funcs[0] = gemm_cuda;
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

void callback_func(void *callback_arg)
{
    running_batches--;
    std::cout << "Concurrently running batches: " << running_batches << "\n";
}

bool tuningStep = true;

int main(int argc, char **argv)
{
	int random_a = 2+(float)(rand())*31 / RAND_MAX; //<-- constant matrix/batch size, will be modified later
	int random_b = 2+(float)(rand())*31 / RAND_MAX;
	int random_c = 2+(float)(rand())*31 / RAND_MAX;
	int batch = (MAX_BYTES / sizeof(float)) / ((random_a*random_b)+(random_c*random_a)+(random_c*random_b));
	
    ktt::Tuner GPUtuner(0, 0, ktt::ComputeApi::CUDA);
    ktt::Tuner CPUtuner(0, 0, ktt::ComputeApi::OpenCL);

    ktt::DimensionVector ndRangeDimensions(batch);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelDefinitionId kernelDefinition = GPUtuner.AddKernelDefinitionFromFile("gemm_batch", "../../examples/simple_gemm/gemm_kernel.cu", ndRangeDimensions, workGroupDimensions);//"/home/jaro/umpalumpa/examples/simple_gemm/kernel.cu", "gemm_batch_kernel", ndRangeDimensions, workGroupDimensions);

    const ktt::KernelId kernel = GPUtuner.CreateSimpleKernel("Batch GEMM", kernelDefinition);

    GPUtuner.SetTimeUnit(ktt::TimeUnit::Microseconds);

/*
    std::vector<ktt::PlatformInfo> platformsCUDA = GPUtuner.GetPlatformInfo();

    for (size_t i = 0; i < platformsCUDA.size(); ++i)
    {
        std::cout << platformsCUDA[i].GetString() << std::endl;
        std::vector<ktt::DeviceInfo> devices = GPUtuner.GetDeviceInfo(static_cast<ktt::PlatformIndex>(i));

        for (const auto& device : devices)
        {
            std::cout << device.GetString() << std::endl;
        }
    }

    std::vector<ktt::PlatformInfo> platformsOpenCL = CPUtuner.GetPlatformInfo();

    for (size_t i = 0; i < platformsOpenCL.size(); ++i)
    {
        std::cout << platformsOpenCL[i].GetString() << std::endl;
        std::vector<ktt::DeviceInfo> devices = CPUtuner.GetDeviceInfo(static_cast<ktt::PlatformIndex>(i));

        for (const auto& device : devices)
        {
            std::cout << device.GetString() << std::endl;
        }
    }
*/
    int tunecounter = 0;

    starpu_init(NULL);

    constexpr size_t timeout = 5;
    switchtime = (unsigned)time(NULL) - timeout;

	bool firstround = true;

    gemmArgs arguments;
    size_t max_config = 5;
    for (int config_id=0; config_id < max_config;)
    {
		if (tunecounter <= 15)
		{
			tuningStep = true;
		}
		else 
		{
			tuningStep = false;
		}
		
		tunecounter++; // <-- this should only happen when a CUDA kernel is launched
		
        if (running_batches > 10) {
            std::this_thread::sleep_for(200ms);
        }
        running_batches++;
        unsigned currenttime = (unsigned)time(NULL);    

        //if (currenttime - switchtime >= timeout)  <-- switching batch sizes is disabled for now
		if (firstround)
        {
			firstround = false;
			
            printf("Processed batches: %i\n", batchcount);
            batchcount = 0;
            config_id++;
            switchtime = currenttime;

			tunecounter = 0;
			tuningStep = true;

            auto *args = &arguments;
			/*args->matsize_a = 2+(float)(rand())*31 / RAND_MAX;
			args->matsize_b = 2+(float)(rand())*31 / RAND_MAX;
			args->matsize_c = 2+(float)(rand())*31 / RAND_MAX;
			
            args->batch = (MAX_BYTES / sizeof(float)) / ((args->matsize_a*args->matsize_b)+(args->matsize_c*args->matsize_a)+(args->matsize_c*args->matsize_b));
			*/
			args->matsize_a = random_a;
			args->matsize_b = random_b;
			args->matsize_c = random_c;
			
			args->batch = batch;
			
			
			GPUtuner.AddParameter(kernel, "SIZE_A", std::vector<uint64_t>{(size_t)args->matsize_a});
			GPUtuner.AddParameter(kernel, "SIZE_B", std::vector<uint64_t>{(size_t)args->matsize_b});
			GPUtuner.AddParameter(kernel, "SIZE_C", std::vector<uint64_t>{(size_t)args->matsize_c});
			GPUtuner.AddParameter(kernel, "GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
			GPUtuner.AddParameter(kernel, "GROUP_SIZE_Z", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
			GPUtuner.AddParameter(kernel, "CACHING_STRATEGY", std::vector<uint64_t>{0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
			GPUtuner.AddParameter(kernel, "PADD_AA", std::vector<uint64_t>{0, 1});
			GPUtuner.AddParameter(kernel, "PADD_AB", std::vector<uint64_t>{0, 1});
			if (args->matsize_c % 4 == 0)
				GPUtuner.AddParameter(kernel, "PADD_C", std::vector<uint64_t>{0});
			else
				GPUtuner.AddParameter(kernel, "PADD_C", std::vector<uint64_t>{0, 4-(args->matsize_c % 4)});
			GPUtuner.AddParameter(kernel, "DIRECT_WRITE", std::vector<uint64_t>{0, 1});
			GPUtuner.AddParameter(kernel, "UNROLL_K", std::vector<uint64_t>{0, 1});

			auto parallelismConstraint = [](const std::vector<size_t>& v) {return v[0] <= v[1];};
			GPUtuner.AddConstraint(kernel, {"GROUP_SIZE_Y", "SIZE_B"}, parallelismConstraint);
			auto paddConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0 && v[1] == 0 && v[2] == 0) || (v[3] > 0);};
			GPUtuner.AddConstraint(kernel, {"PADD_AA", "PADD_AB", "PADD_C", "CACHING_STRATEGY"}, paddConstraint);
			auto dwConstraint = [](const std::vector<size_t>& v) {return (v[0] == 1) || (v[1] > 0);};
			GPUtuner.AddConstraint(kernel, {"DIRECT_WRITE", "CACHING_STRATEGY"}, dwConstraint);
			auto unrollkConstraint = [](const std::vector<size_t>& v) {return (v[0] == 0) || (v[1] == 2);};
			GPUtuner.AddConstraint(kernel, {"UNROLL_K", "CACHING_STRATEGY"}, unrollkConstraint);
		#define SHARED_PER_BLOCK (49152/4)
			auto memConstraint = [](const std::vector<size_t>& v) {size_t a = v[1]; size_t b = v[2]; size_t c = v[3]; return (v[0] == 1 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && v[5] == 1 && ((a+v[7])*(b+v[8])+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK) || (v[0] == 2 && ((a+v[7])*(b+v[8])+c*a+(1-v[4])*(c*b))*v[6] < SHARED_PER_BLOCK);};
			GPUtuner.AddConstraint(kernel, {"CACHING_STRATEGY", "SIZE_A", "SIZE_B", "SIZE_C", "DIRECT_WRITE", "GROUP_SIZE_Y", "GROUP_SIZE_Z", "PADD_AA", "PADD_AB"}, memConstraint);
		#define MAX_BLOCK_SIZE 1024
			auto blockConstraint = [](const std::vector<size_t>&v) {return ((v[0]+v[2])*v[1]*v[3] < MAX_BLOCK_SIZE) && ((v[0]+v[2])*v[1]*v[3] >= 32);};
			GPUtuner.AddConstraint(kernel, {"SIZE_C", "GROUP_SIZE_Y", "PADD_C", "GROUP_SIZE_Z"}, blockConstraint);
			
            args->tuner = &GPUtuner;
			args->kernel = &kernel;
            args->kernelDefinition = &kernelDefinition;

            printf("Switching matrix size: A = %i, B = %i, C = %i., batch=%i\n", args->matsize_a, args->matsize_b, args->matsize_c, args->batch);
        }

        auto *args = new gemmArgs(arguments);
		args->tuningStep = tuningStep;
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

        printf("->>>>>>>>>>Yay\n");
        break;
    }	
	
    starpu_shutdown();

    return 0;
}
