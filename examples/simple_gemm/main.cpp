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
#include <fstream>
#include <string>
#include <sstream>
#include <cuda.h>

using namespace std::chrono_literals;

struct gemmArgs {
    int matsize_a;
    int matsize_b;
    int matsize_c;
    int batch;
    ktt::Tuner* tuner;
    ktt::KernelId kernel;
    ktt::KernelDefinitionId kernelDefinition;
	bool* tuningStep;
	int workerId;
	int taskId;
};


void fillRandomBytes(void *dst, size_t bytes, int fileSwitch) {
	char* fileName;
	if (fileSwitch == 0)
		fileName = "random.txt";
	else
		fileName = "random2.txt";
		
	std::ifstream infile(fileName);
    if (!infile.good())
		printf("INPUT FILE NOT FOUND!\n");	
		
	int fd = open(fileName, O_RDONLY);
    read(fd, dst, bytes);
}

void fillZeroBytes(void *dst, size_t bytes) {
    int fd = open("/dev/zero", O_RDONLY);
    read(fd, dst, bytes);
}

void generate_data(void *buffers[], void *func_arg) {
    float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
	float* results = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);
	long* kernelDuration = (long*)STARPU_VECTOR_GET_PTR(buffers[3]);
	
    gemmArgs* arguments = (gemmArgs*)func_arg;
    int a = arguments->matsize_a;
    int b = arguments->matsize_b;
    int c = arguments->matsize_c;
    int batch = arguments->batch;

    fillRandomBytes(srcA, a*b*batch*sizeof(float), 0);
    fillRandomBytes(srcB, a*c*batch*sizeof(float), 1);
	fillZeroBytes(results, c*b*batch*sizeof(float));
	fillZeroBytes(kernelDuration, sizeof(long));
}


void gemm_cuda(void *buffers[], void *_args)
{	

    float* A = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
    float* B = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
    float* C = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);

    gemmArgs* arguments = (gemmArgs*)_args;
	
	arguments->workerId = starpu_worker_get_id();
	
    int matsize_a = arguments->matsize_a;
    int matsize_b = arguments->matsize_b;
    int matsize_c = arguments->matsize_c;
    int batch = arguments->batch;
    ktt::Tuner* tuner = arguments->tuner;
    const ktt::KernelId kernel = arguments->kernel;
	const ktt::KernelDefinitionId kernelDefinition = arguments->kernelDefinition;
	bool* tuningStep = arguments->tuningStep;

    printf("CUDA processing: %d %d %d %d. Task ID: %d.\n", matsize_a, matsize_b, matsize_c, batch, arguments->taskId);

    int bufferSizeA = matsize_a*matsize_b*batch;
    int bufferSizeB = matsize_c*matsize_a*batch;
    int bufferSizeC = matsize_c*matsize_b*batch;
    const ktt::ArgumentId aId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(A), bufferSizeA,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId bId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(B), bufferSizeB,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Host);
    const ktt::ArgumentId resultId = tuner->AddArgumentVector<float>(reinterpret_cast<ktt::ComputeBuffer>(C), bufferSizeC,
        ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Host);

    const ktt::ArgumentId batchId = tuner->AddArgumentScalar(batch);

	tuner->SetArguments(kernelDefinition, {aId, bId, resultId, batchId});
	
	tuner->SetLauncher(kernel, [kernelDefinition, batch, matsize_c](ktt::ComputeInterface& interface)
	{
		auto config = interface.GetCurrentConfiguration().GetPairs();
		size_t padd_c = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "PADD_C");
		size_t group_size_y = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "GROUP_SIZE_Y");
		size_t group_size_z = ktt::ParameterPair::GetParameterValue<uint64_t>(config, "GROUP_SIZE_Z");
		
		const ktt::DimensionVector newGlobalSize(batch/group_size_z,1,1); 
		const ktt::DimensionVector newlocalSize(matsize_c+padd_c,group_size_y,group_size_z);
		
		interface.RunKernel(kernelDefinition, newGlobalSize, newlocalSize);
	});
	
	if (*tuningStep)
	{
		ktt::KernelResult kernelInfo = tuner->TuneIteration(kernel, {ktt::BufferOutputDescriptor(resultId, C)});
		long* kernelDuration = (long*)STARPU_VECTOR_GET_PTR(buffers[3]);
		
		long KTTInfo = (long)kernelInfo.GetKernelDuration();
		*kernelDuration = KTTInfo;
		printf("TUNING\n");
	}
	else
	{
		auto bestConfig = tuner->GetBestConfiguration(kernel);
		ktt::KernelResult kernelInfo = tuner->Run(kernel, {bestConfig}, {ktt::BufferOutputDescriptor(resultId, C)});
		
		long* kernelDuration = (long*)STARPU_VECTOR_GET_PTR(buffers[3]);
		long KTTInfo = (long)kernelInfo.GetKernelDuration();
		*kernelDuration = KTTInfo;
		
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

    arguments->workerId = starpu_worker_get_id();

	for (int i = 0; i < batch; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < b; k++) {
		        float tmp = 0.0;
                for (int l = 0; l < a; l++) {
                    const size_t offsetA = i*a*b + k*a + l;
                    const size_t offsetB = i*c*a + l*c + j;
                    tmp += srcA[offsetA] * srcB[offsetB];
                }
                result[i*c*b + k*c + j] = tmp;
            }
        }
    }
}

void check_results(void *buffers[], void *func_arg)
{
	gemmArgs* arguments = (gemmArgs*) func_arg;
	
	int workerId = arguments->workerId;
	
	if (workerId >= 0)
	{	
		float* srcA = (float*)STARPU_VECTOR_GET_PTR(buffers[0]);
        float* srcB = (float*)STARPU_VECTOR_GET_PTR(buffers[1]);
        float* results = (float*)STARPU_VECTOR_GET_PTR(buffers[2]);
		
		const int a = arguments->matsize_a;
		const int b = arguments->matsize_b;
		const int c = arguments->matsize_c;
		const int batch = arguments->batch;

		int numberOfMismatches = 0;

		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < c; j++) {
				for (int k = 0; k < b; k++) {
					float verificationResult = 0.0;
					for (int l = 0; l < a; l++) {
						const size_t offsetA = i*a*b + k*a + l;
						const size_t offsetB = i*c*a + l*c + j;
						verificationResult += srcA[offsetA] * srcB[offsetB];
					}
					
					float diff = abs(results[i*c*b + k*c + j] - verificationResult);
					
					if (diff > 10)
					{
						numberOfMismatches++;
					}
				}
			}
		}

		if (numberOfMismatches == 0)
		{
			std::cout << "All results match for task #" << arguments->taskId << ".\n";
		}
		else
		{
			std::cout << "\n\n\n\n Oh no! Some results don't match for task #" << arguments->taskId << "! Number of mismatches: " << numberOfMismatches << ".\n\n\n\n";
		}
		
		long GPUPerformance = 10070; //GPU GFLOPS/s of GeForce RTX 2080
		long taskFlops = 2*a*b*c*batch; //the number of FLOPS needed to compute the task
		long actualRuntime = ((long*)STARPU_VECTOR_GET_PTR(buffers[3]))[0]; //actual runtime retrieved from KTT (GetKernelDuration)
		long expectedRuntimeComputeBound = taskFlops / GPUPerformance; //expected runtime in nanoseconds		
		
		long taskBytes = (a*b+b*c+a*c)*batch;
		long GPUBandwidth = 448/4;
		long expectedRuntimeMemoryBound = taskBytes/GPUBandwidth;
		
		if (expectedRuntimeMemoryBound > expectedRuntimeComputeBound)
		{
			std::cout << "Kernel duration: " << actualRuntime << ". Theoretical optimum should be " << expectedRuntimeMemoryBound << ", being memory-bound.\n";
		}
		else
		{
			std::cout << "Kernel duration: " << actualRuntime << ". Theoretical optimum should be " << expectedRuntimeComputeBound << ", being compute-bound.\n";
		}
			
		//GPU: memsize = 8GB, performance = 10.07 TFLOPs (= 10070 GFLOPs), memory bandwidth = 448 GB/s
		//number of memory operations: a+b+c;
		//number of FLOPs: 2*a*b*c(*batch);
		//word size: (a*b+b*c+a*c)*4;
	}
}

struct Codelet {
    starpu_codelet generate;
    starpu_codelet gemm;
	starpu_codelet check;
    Codelet();
};

Codelet::Codelet():gemm{0}, generate{0}, check{0}{
    gemm.where = STARPU_CUDA; //STARPU_CPU|STARPU_CUDA;
    gemm.cpu_funcs[0] = gemm_cpu;
    gemm.cuda_funcs[0] = gemm_cuda;
    gemm.cuda_flags[0] = STARPU_CUDA_ASYNC;
    gemm.nbuffers = 4;
    gemm.modes[0] = STARPU_R;
    gemm.modes[1] = STARPU_R;
    gemm.modes[2] = STARPU_RW;
	gemm.modes[3] = STARPU_RW;
	gemm.specific_nodes = 1;
	gemm.nodes[0] = STARPU_SPECIFIC_NODE_LOCAL;
	gemm.nodes[1] = STARPU_SPECIFIC_NODE_LOCAL;
	gemm.nodes[2] = STARPU_SPECIFIC_NODE_LOCAL;
	gemm.nodes[3] = STARPU_SPECIFIC_NODE_CPU;
    gemm.name="gemm codelet";

    generate.where = STARPU_CPU;
    generate.cpu_funcs[0] = generate_data;
    generate.nbuffers = 4;
    generate.modes[0] = STARPU_W;
    generate.modes[1] = STARPU_W;
	generate.modes[2] = STARPU_W;
	generate.modes[3] = STARPU_W;
    generate.name="matrix generation codelet";
	
	check.where = STARPU_CPU;
    check.cpu_funcs[0] = check_results;
    check.nbuffers = 4;
    check.modes[0] = STARPU_R;
    check.modes[1] = STARPU_R;
	check.modes[2] = STARPU_R;
	check.modes[3] = STARPU_R;
    check.name="result checking codelet";
}

Codelet codelet;

#define MAX_BYTES (128 * 1024 * 1024)

unsigned switchtime = 0;
int batchcount = 0;

std::atomic<int> running_batches = 0;

void callback_func(void *callback_arg)
{	
    running_batches--;
    std::cout << "Concurrently running batches: " << running_batches << "\n";
}


int main(int argc, char **argv)
{
	starpu_init(NULL);
//////////
/*
	//CUdevice device;
    //CUcontext context;
    //cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
	auto context = starpu_sched_ctx_get_context();
	auto stream = starpu_cuda_get_local_stream();
	
	//GPUtuner.SetKernelStream();
	
    // Create compute API initializer which specifies context and streams that will be utilized by the tuner.
    ktt::ComputeApiInitializer initializer(context, std::vector<ktt::ComputeQueue>{stream});
    auto tunerUnique = std::make_unique<ktt::Tuner>(ktt::ComputeApi::CUDA, initializer);

    // Utilize the tuner in the same way as in previous tutorials.
    auto& GPUtuner = *tunerUnique;
*/	
//////////
	

int checkCounter = 0;
/*	std::ofstream randomfile;
	randomfile.open("random.txt");
	
	float* numbers = (float*) malloc(135000001*sizeof(float));
	
	for (int i = 0; i < 135000000; i++)
	{
		numbers[i] = (float)(rand())*1000 / RAND_MAX;
	}
	
	randomfile.write((char*) numbers, sizeof(float)*135000000);
	
	free(numbers);
	
    randomfile.close();
	
	
	std::ofstream randomfile2;
	randomfile2.open("random2.txt");
	
	float* numbers2 = (float*) malloc(135000001*sizeof(float));
	
	for (int i = 0; i < 135000000; i++)
	{
		numbers2[i] = (float)(rand())*1000 / RAND_MAX;
	}
	
	randomfile2.write((char*) numbers2, sizeof(float)*135000000);
	
	free(numbers2);
	
    randomfile2.close();
*/

//    ktt::Tuner GPUtuner(0, 0, ktt::ComputeApi::CUDA);
	cuInit(0);

	int device;
	cudaGetDevice(&device);
	
	CUcontext context;
	cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
	
	auto stream = starpu_cuda_get_local_stream();
	
	ktt::Tuner GPUtuner(ktt::ComputeApi::CUDA, ktt::ComputeApiInitializer(context, std::vector<ktt::ComputeQueue>{ stream }));
    ktt::Tuner CPUtuner(0, 0, ktt::ComputeApi::OpenCL);
	
	std::vector<ktt::PlatformInfo> platforms = GPUtuner.GetPlatformInfo();

    for (size_t i = 0; i < platforms.size(); ++i)
    {
        std::cout << platforms[i].GetString() << std::endl;
        std::vector<ktt::DeviceInfo> devices = GPUtuner.GetDeviceInfo(static_cast<ktt::PlatformIndex>(i));

        for (const auto& device : devices)
        {
            std::cout << device.GetString() << std::endl;
        }
    }
	
    ktt::DimensionVector ndRangeDimensions(0);
    ktt::DimensionVector workGroupDimensions;
    ktt::KernelDefinitionId kernelDefinition = GPUtuner.AddKernelDefinitionFromFile("gemm_batch", "../../examples/simple_gemm/gemm_kernel.cu", ndRangeDimensions, workGroupDimensions);//"/home/jaro/umpalumpa/examples/simple_gemm/kernel.cu", "gemm_batch_kernel", ndRangeDimensions, workGroupDimensions);
    ktt::KernelId kernel = GPUtuner.CreateSimpleKernel("Batch GEMM", kernelDefinition);

    GPUtuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
	
	std::vector<bool*> tuningSwitches;
	bool t = true; bool* truepointer = &t; bool f = false; bool* falsepointer = &f;
//	tuningSwitches.push_back(truepointer);

    constexpr size_t timeout = 5;
    switchtime = (unsigned)time(NULL) - timeout;
	
    gemmArgs arguments;
	int taskCounter = 0;
    size_t max_config = 5;
		
	int tunecounter = 0;
	
    for (int config_id=0; config_id < max_config;)
    {
		/*if (tunecounter <= 15)
		{
			arguments.tuningStep = true;
		}
		else 
		{
			arguments.tuningStep = false;
		}*/
		
		if (tunecounter == 5)
		{
			tuningSwitches.back() = falsepointer;
		}
		
		tunecounter++; // <-- this should only happen when a CUDA kernel is launched
		
        if (running_batches > 1) {
            std::this_thread::sleep_for(200ms);
        }
        running_batches++;
        unsigned currenttime = (unsigned)time(NULL);    

        if (currenttime - switchtime >= timeout)
        {
			//starpu_task_wait_for_all();
	
			tuningSwitches.push_back(truepointer);
			tunecounter = 0;
			
            batchcount = 0;
            config_id++;
            switchtime = currenttime;

			arguments.matsize_a = 2+(float)(rand())*31 / RAND_MAX;
			arguments.matsize_b = 2+(float)(rand())*31 / RAND_MAX;
			arguments.matsize_c = 2+(float)(rand())*31 / RAND_MAX;
            arguments.batch = (MAX_BYTES / sizeof(float)) / ((arguments.matsize_a*arguments.matsize_b)+(arguments.matsize_c*arguments.matsize_a)+(arguments.matsize_c*arguments.matsize_b));
			
			arguments.batch = arguments.batch / 512;
			arguments.batch = arguments.batch * 512;
			
			//GPUtuner.RemoveKernel(kernel);
			//GPUtuner.RemoveKernelDefinition(kernelDefinition);
			
			std::ostringstream kernelNameBuilder;
			kernelNameBuilder << "Batch GEMM " << arguments.matsize_a << " " << arguments.matsize_b << " " << arguments.matsize_c;
			std::string newKernelName = kernelNameBuilder.str();
			kernel = GPUtuner.CreateSimpleKernel(newKernelName, kernelDefinition);
			
			GPUtuner.AddParameter(kernel, "SIZE_A", std::vector<uint64_t>{(size_t)arguments.matsize_a});
			GPUtuner.AddParameter(kernel, "SIZE_B", std::vector<uint64_t>{(size_t)arguments.matsize_b});
			GPUtuner.AddParameter(kernel, "SIZE_C", std::vector<uint64_t>{(size_t)arguments.matsize_c});
			GPUtuner.AddParameter(kernel, "GROUP_SIZE_Y", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
			GPUtuner.AddParameter(kernel, "GROUP_SIZE_Z", std::vector<uint64_t>{1, 2, 4, 8, 16, 32, 64});
			GPUtuner.AddParameter(kernel, "CACHING_STRATEGY", std::vector<uint64_t>{0, 1, 2}); /* 0 = implicit caching, 1 = local memory, 2 = private memory */
			GPUtuner.AddParameter(kernel, "PADD_AA", std::vector<uint64_t>{0, 1});
			GPUtuner.AddParameter(kernel, "PADD_AB", std::vector<uint64_t>{0, 1});
			if (arguments.matsize_c % 4 == 0)
				GPUtuner.AddParameter(kernel, "PADD_C", std::vector<uint64_t>{0});
			else
				GPUtuner.AddParameter(kernel, "PADD_C", std::vector<uint64_t>{0, 4-(arguments.matsize_c % 4)});
			GPUtuner.AddParameter(kernel, "DIRECT_WRITE", std::vector<uint64_t>{0});//, 1});
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
			
            arguments.tuner = &GPUtuner;
			arguments.kernel = kernel;
            arguments.kernelDefinition = kernelDefinition;
			
			//arguments.workerId = -1;
			//arguments.kernelDuration = -1;

            printf("\n\n\n\nSwitching matrix size: A = %i, B = %i, C = %i., batch=%i\n\n\n\n\n", arguments.matsize_a, arguments.matsize_b, arguments.matsize_c, arguments.batch);
        }

        auto *args = new gemmArgs(arguments);
		//args->tuningStep = tuningStep;
		arguments.tuningStep = tuningSwitches.back();
		args->taskId = taskCounter;
		taskCounter++;
		
        batchcount++;
		
        starpu_data_handle_t matrixA = {0};
		starpu_vector_data_register(&matrixA, -1, 0, args->matsize_a*args->matsize_b*args->batch, sizeof(float));
        starpu_data_set_name(matrixA, "Matrix A");
		
        starpu_data_handle_t matrixB = {0};
        starpu_vector_data_register(&matrixB, -1, 0, args->matsize_c*args->matsize_a*args->batch, sizeof(float));
        starpu_data_set_name(matrixB, "Matrix B");

        starpu_data_handle_t resultBuffer = {0};
        starpu_vector_data_register(&resultBuffer, -1, 0, args->matsize_c*args->matsize_b*args->batch, sizeof(float));
        starpu_data_set_name(resultBuffer, "Result matrix");
		
		starpu_data_handle_t taskInfoBuffer = {0};
	  //starpu_vector_data_register(&taskInfoBuffer, -1, 0, 1, sizeof(long));
		long* kernelDuration;
		starpu_malloc((void **)&kernelDuration, sizeof(long));
		//*kernelDuration = 0;
	  //starpu_variable_data_register(&taskInfoBuffer, STARPU_MAIN_RAM, (uintptr_t) kernelDuration, sizeof(long));
		starpu_vector_data_register(&taskInfoBuffer, -1, (uintptr_t) kernelDuration, 1, sizeof(long));
		starpu_data_set_name(taskInfoBuffer, "Buffer for task metadata");	
		

        struct starpu_task *generateDataTask = starpu_task_create();
        generateDataTask->handles[0] = matrixA;
        generateDataTask->handles[1] = matrixB;
		generateDataTask->handles[2] = resultBuffer;
		generateDataTask->handles[3] = taskInfoBuffer;
        generateDataTask->cl_arg = args;
        generateDataTask->cl_arg_size = sizeof(gemmArgs);
        generateDataTask->cl = &codelet.generate;
        generateDataTask->name = "Generate data task";
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(generateDataTask), "starpu_task_submit generateDataTask");


        struct starpu_task *newtask = starpu_task_create();
        newtask->handles[0] = matrixA;
        newtask->handles[1] = matrixB;
        newtask->handles[2] = resultBuffer;
		newtask->handles[3] = taskInfoBuffer;
        newtask->cl = &codelet.gemm;
        newtask->cl_arg = args;
        newtask->cl_arg_size = sizeof(gemmArgs);
        newtask->callback_func = callback_func;
		newtask->callback_arg = 0;
        newtask->name = "Compute GEMM task";
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(newtask), "starpu_task_submit newtask");  
		
		
	    struct starpu_task *checkDataTask = starpu_task_create();
        checkDataTask->handles[0] = matrixA;
        checkDataTask->handles[1] = matrixB;
        checkDataTask->handles[2] = resultBuffer;
		checkDataTask->handles[3] = taskInfoBuffer;
        checkDataTask->cl = &codelet.check;
        checkDataTask->cl_arg = args;
        checkDataTask->cl_arg_size = sizeof(gemmArgs);
        checkDataTask->name = "Check data task";
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(checkDataTask), "starpu_task_submit checkDataTask");  

		//*tunepointer = false;

		checkCounter++;
		if (checkCounter > 10)
		{
			checkCounter = 0;
			starpu_task_wait_for_all();
			starpu_data_acquire(taskInfoBuffer, STARPU_R);
			long kd = *kernelDuration;
			std::cout << "\n\n!\n" << kd << "!\n!\n\n";
			starpu_data_release(taskInfoBuffer);
		}
		starpu_free(kernelDuration);


        starpu_data_unregister_submit(matrixA);
        starpu_data_unregister_submit(matrixB);
        starpu_data_unregister_submit(resultBuffer);
		starpu_data_unregister_submit(taskInfoBuffer);

	/*		
		starpu_task_wait_for_all();
				
		starpu_data_unregister(matrixA);
		starpu_data_unregister(matrixB);
		starpu_data_unregister(resultBuffer);
		starpu_data_unregister(taskInfoBuffer);

		std::cout << *kernelDuration << "\n\n\n\n";
		starpu_free(&kernelDuration);
	*/
    }	
	
    starpu_shutdown();

    return 0;

}
