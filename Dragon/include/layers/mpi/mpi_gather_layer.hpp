# ifndef MPI_GATHER_LAYER_HPP
# define MPI_GATHER_LAYER_HPP

#include "base_mpi_layer.hpp"
template <typename Dtype>
class MPIGatherLayer : public BaseMPILayer<Dtype> {
public:
	MPIGatherLayer(const LayerParameter& param) : BaseMPILayer<Dtype>(param) {}
	virtual void layerSetup(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	bool mpiSyncFlag(bool flag);
protected:
	virtual void forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& data_need_bp, const vector<Blob<Dtype>*>& bottom);
};

# endif