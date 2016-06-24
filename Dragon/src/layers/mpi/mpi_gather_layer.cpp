#ifndef NO_MPI
#include "layers/mpi/mpi_gather_layer.hpp"


//	gather all processors' blob in this layer
//	output n blobs (n equal to the num of processors)

template <typename Dtype>
bool MPIGatherLayer<Dtype>::mpiSyncFlag(bool flag){
	int temp = (int)flag;
	//	set all processors' flags as root
	MPI_Bcast(&temp, 1, MPI_INT, this->comm_root, this->comm);
	return temp;
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){

	//	only root processor need to forward
	if (this->comm_rank == this->comm_root){
		CHECK_EQ(this->comm_size, top.size());
		for (int i = 0; i < top.size(); i++) top[i]->reshapeLike(*bottom[0]);
	}
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){

	//	only root processor need to forward
	if (this->comm_rank == this->comm_root)
		for (int i = 0; i < top.size(); i++) top[i]->reshapeLike(*bottom[0]);
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){

	//	 gather all bottom blobs in root proecesor
	Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	int count = bottom[0]->count();

	if (this->comm_rank == this->comm_root){
		//	copy for itself
		dragon_copy(count, top[this->comm_rank]->mutable_cpu_data(), bottom_data);
		//	recv from ohters
		for (int i = 0; i < this->comm_size; i++){
			if (i == this->comm_root) continue;
			if (typeid(Dtype) == typeid(double))
				MPI_Recv(top[i]->mutable_cpu_data(), count, MPI_DOUBLE, i, 0, this->comm, MPI_STATUS_IGNORE);
			else MPI_Recv(top[i]->mutable_cpu_data(), count, MPI_FLOAT, i, 0, this->comm, MPI_STATUS_IGNORE);
		}
	}
	else{
		if (typeid(Dtype) == typeid(double))
			MPI_Send(bottom_data, count, MPI_DOUBLE, this->comm_root, 0, this->comm);
		else MPI_Send(bottom_data, count, MPI_FLOAT, this->comm_root, 0, this->comm);
	}
}

template <typename Dtype>
void MPIGatherLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
	int count = bottom[0]->count();
	if (this->comm_rank == this->comm_root){
		dragon_copy(count, bottom_diff, top[this->comm_rank]->mutable_cpu_diff());
		for (int i = 0; i < this->comm_size; i++){
			if (i == this->comm_root) continue;
			if (typeid(Dtype) == typeid(double))
				MPI_Send(top[i]->mutable_cpu_diff(), count, MPI_DOUBLE, i, 0, this->comm);
			else MPI_Send(top[i]->mutable_cpu_diff(), count, MPI_FLOAT, i, 0, this->comm);
		}
	}
	else{
		if (typeid(Dtype) == typeid(double))
			MPI_Recv(bottom_diff, count, MPI_DOUBLE, this->comm_root, 0, this->comm, MPI_STATUS_IGNORE);
		else MPI_Recv(bottom_diff, count, MPI_FLOAT,this->comm_root, 0, this->comm, MPI_STATUS_IGNORE);
	}
}

INSTANTIATE_CLASS(MPIGatherLayer);
#endif




