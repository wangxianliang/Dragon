#include "layers/neuron/dropout_layer.hpp"

template<typename Dtype>
__global__ void DropoutForwardKernel(const int n, const Dtype* bottom_data,
	const unsigned int* mask, const unsigned int threshold, const Dtype scale,Dtype* top_data) {
	CUDA_KERNEL_LOOP(idx, n) {
		//	filter the value lower threshold as zero
		top_data[idx] = bottom_data[idx] * (mask[idx] > threshold)*scale;
	}
}

template<typename Dtype>
void DropoutLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	unsigned int* mask = rand_vec.mutable_gpu_data();
	const int count = bottom[0]->count();
	if (this->phase == TRAIN){
		//	gpu can not generate Bernoulli distribution directly
		//	firstly generate Uniform distribution(0~UNIT_MAX)
		//	then use threshold to filter them
		dragon_gpu_rng_uniform(count, mask);
		DropoutForwardKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
			count, bottom_data, mask, threshold, scale,top_data);
	}
	else if (this->phase == TEST){
		dragon_gpu_copy(count, top_data, bottom_data);
		if (!this->param.dropout_param().scale())
			dragon_gpu_scal(count, Dtype(1) - prob, top_data);
	}
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DropoutBackwardKernel(const int n, const Dtype* top_diff,
	const unsigned int* mask, const unsigned int threshold,const Dtype scale, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(idx, n) {
		bottom_diff[idx] = top_diff[idx] * (mask[idx] > threshold)*scale;
	}
}

template <typename Dtype>
void DropoutLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	if (data_need_bp[0]){
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const int count = bottom[0]->count();
		if (this->phase == TRAIN){
			const unsigned int* mask = rand_vec.gpu_data();
			DropoutBackwardKernel<Dtype> << <GET_BLOCKS(count), CUDA_NUM_THREADS >> >(
				count, top_diff, mask, threshold, scale,bottom_diff);
		}
		else if (this->phase == TEST){
			NOT_IMPLEMENTED;
		}
	}
	CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);