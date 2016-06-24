#ifndef NO_MPI
#include "layers/mpi/mpi_broadcast_layer.hpp"

//	MPI_Bcast is a unity of MPI_Send & MPI_Recv
//	if current rank ¡Ù root£¬it works as a MPI_Recv(Blocking)
//	and vice versa

//	it does not use P2P communication protocol
//	a Send Action(this->comm_root) must mathch some Recv Actions(this->comm)
//	strictly and orderly 


template <typename Dtype>
bool MPIBroadcastLayer<Dtype>::mpiSyncFlag(bool flag){
	DLOG(INFO) << this->comm_rank << " - " << flag;
	int temp = (int)flag;
	int* buffer=new int[this->comm_size];
	//	collect all flags and distribute to the buffer for all processors
	MPI_Allgather(&temp, 1, MPI_INT, buffer, 1, MPI_INT, this->comm);
	for (int i = 0; i < this->comm_size; i++) if (buffer[i]) return true;
	return false;
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::layerSetup(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	int dims[4];
	//	prepare dims for broadcast
	if (this->comm_rank == this->comm_root){
		dims[0] = bottom[0]->num();
		dims[1] = bottom[0]->channels();
		dims[2] = bottom[0]->height();
		dims[3] = bottom[0]->width();
	}
	//	different actions
	//	root processor(src): broadcast it's bottom shape
	//	sub processors(this->comm): receive root's bottom shape
	MPI_Bcast(dims, 4, MPI_INT, this->comm_root, this->comm);
	top[0]->reshape(dims[0], dims[1], dims[2], dims[3]);
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::reshape(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	layerSetup(bottom, top);
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>*> &bottom, const vector<Blob<Dtype>*> &top){
	//	root processor
	if (this->comm_rank == this->comm_root){
		Dtype* bottom_data = bottom[0]->mutable_cpu_data();
		int count = bottom[0]->count();
		//	broadcast bottom_data to others¡® top_data
		if (typeid(Dtype) == typeid(double))
			MPI_Bcast(bottom_data, count, MPI_DOUBLE, this->comm_root, this->comm);
		else MPI_Bcast(bottom_data, count, MPI_FLOAT, this->comm_root, this->comm);
		//	note that the root must copy for itself
		dragon_copy(count, top[0]->mutable_cpu_data(), bottom_data);
	}
	//	sub processors
	else{
		Dtype* top_data = top[0]->mutable_cpu_data();
		int count = top[0]->count();
		//	receive broadcast contents and fill in top_data
		//	note that the sub processors do not have bottom_data(!!do not use it)
		if (typeid(Dtype) == typeid(double))
			MPI_Bcast(top_data, count, MPI_DOUBLE, this->comm_root, this->comm);
		else MPI_Bcast(top_data, count, MPI_FLOAT, this->comm_root, this->comm);
	}
}

template <typename Dtype>
void MPIBroadcastLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>*> &top,
	const vector<bool> &data_need_bp, const vector<Blob<Dtype>*> &bottom){
	//	root processor
	if (this->comm_rank == this->comm_root){
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		int count = bottom[0]->count();
		//	note that the root must copy for itself
		dragon_copy(count, bottom_diff, top_diff);
		//	gather and sum others' diff
		//	use P2P method but not Bcast(tag=0)
		for (int i = 0; i < this->comm_size; i++){
			//	ignore root
			if (i == this->comm_root) continue;
			//	store the buffer in top_diff
			if (typeid(Dtype) == typeid(double))
				MPI_Recv(top_diff, count, MPI_DOUBLE, i, 0, this->comm, MPI_STATUS_IGNORE);
			else MPI_Recv(top_diff, count, MPI_FLOAT, i, 0, this->comm, MPI_STATUS_IGNORE);
			//	sum
			dragon_add(count, bottom_diff, top_diff, bottom_diff);
		}
	}
	//	sub processors
	else{
		//	send buffer to root processor
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		int count = top[0]->count();
		if (typeid(Dtype) == typeid(double))
			MPI_Send(top_diff, count, MPI_DOUBLE, this->comm_root, 0, this->comm);
		else MPI_Send(top_diff, count, MPI_FLOAT, this->comm_root, 0, this->comm);
	}
}

INSTANTIATE_CLASS(MPIBroadcastLayer);
#endif