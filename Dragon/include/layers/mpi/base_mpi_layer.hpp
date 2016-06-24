# ifndef BASE_MPI_LAYER_HPP
# define BASE_MPI_LAYER_HPP

#include "../../layer.hpp"
#include "mpi/mpi.h"

template <typename Dtype>
class BaseMPILayer : public Layer < Dtype > {
public:
	BaseMPILayer(const LayerParameter& param) :Layer<Dtype>(param){
		comm = (MPI_Comm)param.mpi_param().comm_id();
		group = (MPI_Group)param.mpi_param().group_id();
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		MPI_Comm_size(comm, &comm_size);
		MPI_Comm_rank(comm, &comm_rank);

		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		int old_src = param.mpi_param().root();
		MPI_Group_translate_ranks(world_group, 1, &old_src, group, &comm_root);

		CHECK(comm_root !=MPI_UNDEFINED)<< "MPI root is not included in layer group."; 
	}
protected:
	MPI_Comm comm;
	MPI_Group group;
	int comm_size, comm_rank, comm_root;
	int world_size, world_rank;
};



# endif