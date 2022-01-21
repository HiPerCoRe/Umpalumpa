#include "flexalign_cpu.hpp"
// #include "flexalign_cuda.hpp"
#include "flexalign_starpu.hpp"

int main(int argc, char **argv)
{
  size_t size_x,size_y,movie_size,batch_size,num_of_movies,downscale_factor = 0;  

  if (argc < 7){
    //std::cout << "Default parameter values will be used (2048x2048x40x1)." << endl;
    size_x = 2048;
	  size_y = 2048;
	  movie_size = 40;
	  batch_size = 5;
	  num_of_movies = 1;
	  downscale_factor = 2;
  }
  else{
    size_x = atoi(argv[1]);
	  size_y = atoi(argv[2]);
	  movie_size = atoi(argv[3]);
	  batch_size = atoi(argv[4]);
	  num_of_movies = atoi(argv[5]);
	  downscale_factor = atoi(argv[6]);
  }
	
  //auto size = umpalumpa::data::Size(2048, 2048, 1, 40);
  auto size = umpalumpa::data::Size(size_x, size_y, 1, movie_size);
  const auto program = []() -> std::unique_ptr<FlexAlign<float>> {
    if (true) { return std::make_unique<FlexAlignStarPU<float>>(); }
    return std::make_unique<FlexAlignCPU<float>>();
  }();
  program->Execute(size,batch_size,num_of_movies,downscale_factor);
}
