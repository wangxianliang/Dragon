#ifndef NO_MPI
#include "parallel/parameter_server.hpp"
#include "common.hpp"

template <typename BlobType>
void ParameterServer<BlobType>::handle_message(int param_id, int node_id){
	MPI_Status status;
	boost::shared_ptr<BlobType> param = params[param_id];
	MPI_Recv((void*)param->mutable_cpu_diff(), param->count(), getType(Dtype(0.0)),
		node_id, param_id, MPI_COMM_WORLD, &status);
	DLOG(INFO) << "[Server]: receive diff " << status.MPI_TAG << " from " << status.MPI_SOURCE;
	update(param_id);
	//	feedback updated-data 
	MPI_Send((void*)param->cpu_data(), param->count(), getType(Dtype(0.0)),
		node_id, param_id, MPI_COMM_WORLD);
	DLOG(INFO) << "[Server]: feedback data " << param_id << " to " << node_id;

}

template <typename BlobType>
void ParameterServer<BlobType>::listen(bool locking = true){
	if (rank == 0){
		this->locking = locking;
		//	create locks for all params
		if (locking) update_locks.reset(new boost::mutex[params.size()]);
		//	create thread and run listening service
		listen_thr.reset(new boost::thread(&ParameterServer::do_listen, this));
	}
}

template <typename BlobType>
void ParameterServer<BlobType>::do_listen(){
	while (1){
		MPI_Status status;
		//	listen and wait for request
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		DLOG(INFO) << "[Server]: get request " << status.MPI_TAG << " from " << status.MPI_SOURCE;
		//	server may receive request from itself
		//	see de-constructor
		if (status.MPI_SOURCE == rank){
			int dummy;
			//	drag received data for server cache
			MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
			break;
		}
		//	handle request if it is from non-root nodes
		handle_message(status.MPI_TAG, status.MPI_SOURCE);
	}
}

template <typename BlobType>
void ParameterServer<BlobType>::update(int param_id){
	static int update_cnt = 0;
	MPI_Status status;
	boost::shared_ptr<BlobType> param = params[param_id];
	DLOG(INFO) << "[Public]: request update " << param_id << " from" << rank;
	//	trigger in the do_listen() or call by server itself
	if (rank == 0){
		if (locking) update_locks[param_id].lock();
		//	update params in ParameterServer(Root node)
		dragon_axpy(param->count(), Dtype(-1), param->cpu_diff(), param->mutable_cpu_data());
		if (locking) update_locks[param_id].unlock();
		DLOG(INFO) << "[Server]: finish update";
		update_cnt++;
		if (update_cnt % 100000 == 0)
			DLOG(INFO) << "[Server]: reports:  " << update_cnt << " updates";
	}
	//	call by Net
	else{
		//	seed request diff to ParameterServer(Root node)
		MPI_Send((void*)param->cpu_diff(), param->count(), getType(Dtype(0.0)),
			0, param_id, MPI_COMM_WORLD);
		DLOG(INFO) << "[Node]: " << rank << " send param diff" << param_id;
		MPI_Recv((void*)param->mutable_cpu_data(), param->count(), getType(0.0),
			0, param_id, MPI_COMM_WORLD, &status);
		DLOG(INFO) << "[Node]: " << rank << " receive param data" << param_id;
	}
}

template class ParameterServer < Blob<float> > ; 
template class ParameterServer < Blob<double> > ;
#endif