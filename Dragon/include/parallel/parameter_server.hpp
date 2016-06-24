# ifndef PARAMETER_SERVER_HPP
# define PARAMETER_SERVER_HPP

#ifndef NO_MPI
#include <mpi/mpi.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/shared_array.hpp>
#include <boost/thread/thread.hpp>  //boost::thread
#include <boost/shared_ptr.hpp>		//boost::shared_ptr
#include "blob.hpp"

template <typename BlobType>
class ParameterServer{
	typedef typename BlobType::element_type Dtype;
public:
	ParameterServer() { MPI_Comm_rank(MPI_COMM_WORLD, &rank); }
	virtual ~ParameterServer(){
		if (listen_thr){
			int dummy = 1;
			//	seed '1' to itself
			MPI_Send(&dummy, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			listen_thr->join();
		}
		if (rank != -1) MPI_Finalize(); //	terminate the MPI_THREAD
	}
	int getRank() {return rank;}
	//	call by boost::thread callback function
	void sleep(){
		boost::this_thread::sleep(boost::posix_time::milliseconds(100)); //	100ms
	}
	void sync() {MPI_Barrier(MPI_COMM_WORLD);}
	void set_params(const std::vector<boost::shared_ptr<BlobType> > &params) { this->params = params; }
	void handle_message(int param_id, int node_id);
	void do_listen();
	//	lock if rank0 device run as a solver also(default)
	void listen(bool locking=true);
	//	this function can be execute by following ways:
	//	1. trigger in the do_listen()  [automatical]
	//	2. call by Net				   [artifical]
	void update(int param_id);
private:
	int rank;
	boost::shared_ptr<boost::thread> listen_thr;
	std::vector<boost::shared_ptr<BlobType> > params;
	bool locking;
	boost::shared_array<boost::mutex> update_locks;
	MPI_Datatype getType(Dtype data){
		if (typeid(Dtype) == typeid(float)) return MPI_FLOAT;
		else return MPI_DOUBLE;
	}
};

#endif



# endif