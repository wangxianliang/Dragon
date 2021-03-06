# ifndef RESHAPE_LAYER_HPP
# define RESHAPE_LAYER_HPP

#include "../../layer.hpp"


template <typename Dtype>
class ReshapeLayer :public Layer < Dtype > {
public:
	ReshapeLayer(const LayerParameter& param) :Layer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
	virtual void reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {}
	virtual void backward_cpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	virtual void forward_gpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top) {}
	virtual void backward_gpu(const vector<Blob<Dtype>*> &top, const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom) {}
	int infer_axis, constant_count;
};

# endif