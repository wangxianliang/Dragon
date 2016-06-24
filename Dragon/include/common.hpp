# ifndef COMMON_HPP
# define COMMON_HPP

#pragma warning(disable:4251)

#include <cstdio>
#include <assert.h>
#include <string>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <boost/thread/thread.hpp>  //boost::thread
#include <boost/shared_ptr.hpp>		//boost::shared_ptr
#include <boost/thread/tss.hpp>		//boost::thread_specific_ptr
#include <boost/smart_ptr/weak_ptr.hpp>
#include <boost/thread/mutex.hpp>   //boost::mutex

#include "utils/math.hpp"
#include "utils/rng.hpp"
#include "utils/device.hpp"

using namespace std;
using namespace boost;

#define NOT_IMPLEMENTED LOG(FATAL)<<"Use non-implemented codes."

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#include <cublas.h>
#include <curand.h>
//#include <device_launch_parameters.h>
#include <cuda.h>
#endif

class Dragon{
public:
	Dragon();
	~Dragon();
	static Dragon& Get();
	enum Mode{ CPU, GPU };
	enum Arch{ NORMAL, DEVICE, PS};
	static Mode get_mode() { return Get().mode; }
	static Arch get_arch() { return Get().arch; }
	static void set_mode(Mode mode) {Get().mode = mode;}
	static void set_arch(Arch arch) { Get().arch = arch; }
	static int get_solver_count() { return Get().solver_count; }
	static void set_solver_count(int val) { Get().solver_count = val; }
	static bool get_root_solver() {return Get().root_solver;}
	static void set_root_solver(bool val) {Get().root_solver = val;}
	static void set_random_seed(unsigned int seed);
	static void set_device(const int device_id);
	static rng_t* get_rng(){
		if (!Get().random_generator){
			Get().random_generator.reset(new RNG());
		}
		rng_t* rng = Get().random_generator.get()->get_rng();
		return rng;
	}
	static unsigned int get_random_value(){
		rng_t* rng = get_rng();
		return (*rng)();
	}
	static int64_t cluster_seedgen();
	class RNG{
	public:
		RNG() { generator.reset(new Generator()); }
		RNG(unsigned int seed) {generator.reset(new Generator(seed));}
		rng_t* get_rng() { return generator->get_rng(); }
			class Generator{
			public:
				//using pid generators a simple seed to construct RNG
				Generator() :rng(new rng_t((uint32_t)Dragon::cluster_seedgen())) {} 
				//assign a specific seed to construct RNG
				Generator(unsigned int seed) :rng(new rng_t(seed)) {}
				rng_t* get_rng() { return rng.get(); }
			private:
				boost::shared_ptr<rng_t> rng;
			};
	private:
		boost::shared_ptr<Generator> generator;
	};
#ifndef CPU_ONLY
	static cublasHandle_t get_cublas_handle() { return Get().cublas_handle; }
	static curandGenerator_t get_curand_generator() {return Get().curand_generator;}
#endif

private:
	Mode mode;
	Arch arch;
	int solver_count;
	bool root_solver;
	//	use mpi rank to set gpu device 
	boost::shared_ptr<RNG> random_generator;
#ifndef CPU_ONLY
	cublasHandle_t cublas_handle;
	curandGenerator_t curand_generator;
#endif
};

//	MACRO: Instance a class
//	more info see http://bbs.csdn.net/topics/380250382
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>


//	instance for forward/backward in cu file
//	note that INSTANTIATE_CLASS is meaningless in NVCC complier
//	you must INSTANTIATE again
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
  template void classname<float>::forward_gpu( \
      const vector<Blob<float>*>& bottom, \
      const vector<Blob<float>*>& top); \
  template void classname<double>::forward_gpu( \
      const vector<Blob<double>*>& bottom, \
      const vector<Blob<double>*>& top);

#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
  template void classname<float>::backward_gpu( \
      const vector<Blob<float>*>& top, \
      const vector<bool> &data_need_bp, \
      const vector<Blob<float>*>& bottom); \
  template void classname<double>::backward_gpu( \
      const vector<Blob<double>*>& top, \
      const vector<bool> &data_need_bp, \
      const vector<Blob<double>*>& bottom)

#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \
  INSTANTIATE_LAYER_GPU_FORWARD(classname); \
  INSTANTIATE_LAYER_GPU_BACKWARD(classname)

# endif