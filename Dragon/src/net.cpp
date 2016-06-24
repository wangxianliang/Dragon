#ifndef NO_MPI
#include <mpi/mpi.h>
#endif

#include "net.hpp"
#include "layer_factory.hpp"
#include "utils/insert_splits.hpp"
#include "utils/io.hpp"

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param):
	root_net(root_net){
	Init(param);
}

//	usually create a net from a prototxt file 
template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase):
	root_net(root_net){
	NetParameter param;
	readNetParamsFromTextFileOrDie(param_file, &param);
	param.mutable_state()->set_phase(phase);
	Init(param);
}

//	remove the use of blobs that need to get by MPI
template <typename Dtype>
void Net<Dtype>::adjustMPIRank(LayerParameter* param){
#ifndef NO_MPI
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (param->type() == "MPIBroadcast")
		if (param->mpi_param().root() != rank) param->clear_bottom();

	if (param->type() == "MPIGather")
		if (param->mpi_param().root() != rank) param->clear_top();
#endif NO_MPI
}

template <typename Dtype>
void Net<Dtype>::establishMPIComm(LayerParameter* param){
#ifndef NO_MPI
	//	init for all MPIxxxLayers
	if (param->type().find("MPI") != string::npos){
		MPI_Group world_group, local_group;
		MPI_Comm local_comm;
		//	all processors will locate at world_group
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		//	default use world_group
		local_group = world_group;

		int world_size, world_rank;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


		vector<int> vec; int err_code;

		//	check for include rules
		if (param->include_size() > 0){
			for (int i = 0; i < param->include_size(); i++)
				for (int j = 0; j < param->include(i).mpi_rank_size(); j++)
					vec.push_back(param->include(i).mpi_rank(j));
			//	
			if (vec.size()>0){
				int* ranks = new int[vec.size()];
				for (int i = 0; i < vec.size(); i++) ranks[i] = vec[i];
				err_code = MPI_Group_incl(world_group, vec.size(), ranks, &local_group);
				CHECK(err_code == MPI_SUCCESS)
					<< "[" << param->name() << "]: " << " create mpi groups failed.";
			}
		}

		//	check for exclude rules
		if (param->exclude_size() > 0){
			vec.clear();
			for (int i = 0; i < param->exclude_size(); i++)
				for (int j = 0; j < param->exclude(i).mpi_rank_size(); j++)
					vec.push_back(param->exclude(i).mpi_rank(j));

			if (vec.size()>0){
				int* ranks = new int[vec.size()];
				for (int i = 0; i < vec.size(); i++) ranks[i] = vec[i];
				err_code = MPI_Group_excl(world_group, vec.size(), ranks, &local_group);
				CHECK(err_code == MPI_SUCCESS)
					<< "[" << param->name() << "]: " << " create mpi groups failed.";
			}
		}


		//	create final group as a new comm
		//	note that if no include/exclude rules
		//	it will use the world_group(i.e. all the processors)
		//	and a special comm id(not MPI_WORLD_COMM)
		err_code = MPI_Comm_create(MPI_COMM_WORLD, local_group, &local_comm);
		CHECK(err_code == MPI_SUCCESS)
			<< "[" << param->name() << "]: " << " create mpi comm failed.";

		if (local_comm != MPI_COMM_NULL){
			MPI_Comm_size(local_comm, &world_size);
			LOG(INFO) << "Rank:" << world_rank << " /Layer:" << param->name()
				<< "successfully create MPI comm group of " << world_size << "members.";
		}

		param->mutable_mpi_param()->set_comm_id((google::protobuf::uint64)local_comm);
		param->mutable_mpi_param()->set_group_id((google::protobuf::uint64)local_group);
	}
#endif
}

template <typename Dtype>
bool Net<Dtype>::stateMeetRule(const NetState& state, const NetStateRule& rule, const string& name){
	//	check phase
	//	use for TRAIN/TEST DataLayer
	if (rule.has_phase()){
		if (rule.phase() != state.phase()){
			LOG_IF(INFO, Dragon::get_root_solver()) << "The NetState phase("
				<< state.phase() << ") differed from layer phase(" << rule.phase() << ").";
			return false;
		}
	}
#ifndef NO_MPI
	//	check mpi
	if (rule.mpi_rank_size() > 0){
		int rank;
		bool hasRank = false;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		for (int i = 0; i < rule.mpi_rank_size(); i++)
			if (rule.mpi_rank(i) == rank) hasRank = true;
		if (!hasRank) return false;
	}
#endif

	return true;
}

template <typename Dtype> 
void Net<Dtype>::filterNet(const NetParameter& param, NetParameter* filtered_param){
	NetState state(param.state());
	filtered_param->CopyFrom(param); 
	// remove all layer params and then filter
	filtered_param->clear_layer();
	for (int i = 0; i < param.layer_size(); i++){
		LayerParameter layer_param;
		layer_param.CopyFrom(param.layer(i));
		const string& layer_name = layer_param.name();
		//	usually a layer has not any include/exclude rules
		CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
			<< "Specify either include or exclude rules.";
		bool layer_included = (layer_param.include_size() == 0);
		//	assume 'included' and check if meet any excluded rules
		for (int j = 0; layer_included&&j < layer_param.exclude_size(); j++){
			if (stateMeetRule(state, layer_param.exclude(j), layer_name)){
				//	cancel 'included'
				layer_included = false;
			}
		}
		//	assume 'excluded' and check if meet any included rules
		for (int j = 0; !layer_included&&j < layer_param.include_size(); j++){
			if (stateMeetRule(state, layer_param.include(j), layer_name)){
				//	cancel 'excluded'
				layer_included = true;
			}
		}
		//	 copy the included layer to filtered_param
		if (layer_included){
			//	create MPI_COMM
			if (Dragon::get_arch() != Dragon::NORMAL){
				establishMPIComm(&layer_param);
				adjustMPIRank(&layer_param);
			}
			filtered_param->add_layer()->CopyFrom(layer_param);
		}
	}
}

template <typename Dtype>
void Net < Dtype >::appendTop(const NetParameter& param, const int layer_id, const int top_id,
	set<string>* available_blobs, map<string, int>* blob_name_to_idx){
	boost::shared_ptr<LayerParameter> layer_param(
		layer_id >= 0 ? new LayerParameter(param.layer(layer_id)) : NULL);
	//	use (layer_id//top_id) or (-1//top_id) to get a blob name
	const string& blob_name = layer_param ?
		(top_id<layer_param->top_size() ? layer_param->top(top_id) : "(automatic)") : param.input(top_id);
	//	in-place case (e.g:
	//	I0721 10:38 : 16.722070  4692 net.cpp : 84] relu1 <-conv1
	//	I0721 10:38 : 16.722082  4692 net.cpp : 98] relu1->conv1(in-place)
	//	check a blob whether at the same postion in both bottom and top
	if (blob_name_to_idx && layer_param && top_id < layer_param->bottom_size() 
		&& blob_name == layer_param->bottom(top_id)){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< layer_param->name() << "[Layer-Produce]->" << blob_name << " [Blob-Name] (in-place)";
		//	add into this layer's top blob using blob_name
		top_vecs[layer_id].push_back(blobs[(*blob_name_to_idx)[blob_name]].get());
		//	log the id
		top_id_vecs[layer_id].push_back((*blob_name_to_idx)[blob_name]);
	}
	else if (blob_name_to_idx && (*blob_name_to_idx).count(blob_name) ){
		LOG(FATAL) << "Top blob:" << blob_name << " propogate from multiple sources.";
	}
	// normal top blob stuffing
	else{
		//	debug info
		if (Dragon::get_root_solver()){
			if (layer_param) LOG(INFO) << layer_param->name() << "[Layer-Produce] ->" << blob_name << " [Blob-Name]";
			//	special case and only used when viewing a Net's structure
			//	because they need not specify data source and can not train or test
			//	virtual data input blobs do not belong to any layers
			//	see more in insert_splits.cpp/void InsertSplits() 
			else LOG(INFO) << "Input " << top_id << "[Blob-Code] -> " << blob_name << "[Blob - Name]";
		}
		//	allocate a null blob at first
		boost::shared_ptr<Blob<Dtype> > ptr_blob(new Blob<Dtype>());
		//	store global blob infos
		const int blob_id = blobs.size();
		blobs.push_back(ptr_blob);
		blobs_name.push_back(blob_name);
		blobs_need_backward.push_back(false);
		//	encode index number for a name
		//	which also represent this top blob is binded from a bottom
		//	check it before can know whether a top blob has multiple sources(Forbidden)
		if (blob_name_to_idx) (*blob_name_to_idx)[blob_name] = blob_id;
		//	reshape for virtual input blobs solely
		//	becaude they do not exist into a DataLayer(provide reshape/transfrom service)
		if (layer_id == -1){
			ptr_blob->reshape(param.input_shape(top_id));
			//	store solely for virtual input blobs
			net_input_blobs.push_back(ptr_blob.get());
			net_input_blob_indices.push_back(blob_id);
		}
		else{
			top_vecs[layer_id].push_back(ptr_blob.get());
			top_id_vecs[layer_id].push_back(blob_id);
		}
	}
	//	a set used for listing all exsiting top blobs
	if (available_blobs) available_blobs->insert(blob_name);
} 

template <typename Dtype>
int Net < Dtype >::appendBottom(const NetParameter& param, const int layer_id, const int bottom_id,
	set<string>* available_blobs, map<string, int>* blob_name_to_idx){
	const LayerParameter& layer_param = param.layer(layer_id);
	const string& blob_name = layer_param.bottom(bottom_id);
	if (!available_blobs->count(blob_name))
		LOG(FATAL) << "Unknown bottom blob: " << blob_name<< " at layer: " << layer_param.name() << ".";
	//	a bottom blob must share a top blob
	const int blob_id = (*blob_name_to_idx)[blob_name];
	LOG_IF(INFO, Dragon::get_root_solver())
		<< layer_param.name() << "[Layer-Accept] <- " << blob_name << " [Blob-Name]";
	bottom_vecs[layer_id].push_back(blobs[blob_id].get());
	bottom_id_vecs[layer_id].push_back(blob_id);
	//	ensure that a top blob must specify only one bottom blob
	//	SplitLayer can be used to shadow a top blob into several top blobs
	available_blobs->erase(blob_name);
	bool need_bp = true;
	//	default(TEST) is false
	bottoms_need_backward[layer_id].push_back(need_bp & blobs_need_backward[blob_id]);
	return blob_id;
}

template <typename Dtype>
void Net<Dtype>::appendParam(const NetParameter& param, const int layer_id, const int param_id){
	const LayerParameter& layer_param = param.layer(layer_id);
	Layer<Dtype>* layer = layers[layer_id].get();
	const int param_size = layer_param.param_size();
	//	default name="" (not set)
	string param_name = param_id<param_size? layer_param.param(param_id).name() : "";
	//	has name
	if (param_name.size()) param_display_names.push_back(param_name);
	//	set (layer_name,param_id) as name
	else{
		ostringstream display_name;
		display_name << "(";
		display_name << layer_param.name();
		display_name << ",";
		display_name << param_id;
		display_name << ")";
		param_display_names.push_back(display_name.str());
	}
	//	each param blob has a net id(both weight and bias)
	const int net_param_id = param_blobs.size();
	//	add param blob which can be used by a net id
	param_blobs.push_back(layer->getBlobs()[param_id]);
	//	store a net id
	//	param_id_vecs[layer_id][param_id] can get the net_param_id
	param_id_vecs[layer_id].push_back(net_param_id);
	//	store orginal id ( x_th layer/ y_th param )
	//	param_layer_indices[net_param_id] can get layer_id/param_id
	param_layer_indices.push_back(make_pair(layer_id, param_id));
	ParamSpec default_hyperparameter;
	const ParamSpec* hyperparameter = param_id < param_size ?
		&layer_param.param(param_id) : &default_hyperparameter;

	//	store as vector<learnable_params>
	const int learnable_param_id = learnable_params.size();
	learnable_params.push_back(param_blobs[net_param_id].get());
	learnable_param_ids.push_back(learnable_param_id); 
	has_params_lr.push_back(hyperparameter->has_lr_mult());
	has_params_decay.push_back(hyperparameter->has_decay_mult());
	params_lr.push_back(hyperparameter->lr_mult());
	params_decay.push_back(hyperparameter->decay_mult());
	
}
template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param){
	phase = in_param.state().phase();
	NetParameter filtered_param, param;
	//	filter for unqualified LayerParameters(e.g Test DataLayer)
	filterNet(in_param, &filtered_param);
	insertSplits(filtered_param, &param);
	name = param.name();
	LOG_IF(INFO, Dragon::get_root_solver()) << "Initialize net from parameters: ";
	/*<< endl << param.DebugString();*/

	map<string, int> blob_name_to_idx;
	set<string> available_blobs;
	CHECK_EQ(param.input_size(), param.input_shape_size())<< "input blob_shape must specify a blob.";
	memory_used = 0;
	//	check and stuff virtual input blobs firstly [Viewing Mode Only]
	for (int input_id=0; input_id < param.input_size(); input_id++){
		const int layer_id = -1;
		//	net_input.push_back(.....virtual blob.....)
		appendTop(param, layer_id, input_id, &available_blobs, &blob_name_to_idx);
	}
	//	stuff real blobs for each layer then [Traning/Testing/Viewing Mode]
	bottom_vecs.resize(param.layer_size());
	bottom_id_vecs.resize(param.layer_size());
	bottoms_need_backward.resize(param.layer_size());
	top_vecs.resize(param.layer_size());
	top_id_vecs.resize(param.layer_size());
	param_id_vecs.resize(param.layer_size());
	for (int layer_id = 0; layer_id < param.layer_size(); layer_id++){

		//	copy net phase to layer if not set
		if (!param.layer(layer_id).has_phase())
			param.mutable_layer(layer_id)->set_phase(phase);
		const LayerParameter& layer_param = param.layer(layer_id);


		//	use layer factory to create a pointer
		//	layer type is referred by layer_param->type()
		//	see more in layer_factory.hpp

		layers.push_back(LayerFactory<Dtype>::createLayer(layer_param));
		layer_names.push_back(layer_param.name());
		LOG_IF(INFO, Dragon::get_root_solver()) << "Create Layer: " << layer_param.name();
		bool need_bp = false;
		//	stuff bottom blobs
		for (int bottom_id = 0; bottom_id < layer_param.bottom_size(); bottom_id++){
			const int blob_id = appendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
			//	check whether a bottom need back propogation
			need_bp |= blobs_need_backward[blob_id];
		}
		//	stuff top blobs
		for (int top_id = 0; top_id < layer_param.top_size(); top_id++)
			appendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);

		Layer<Dtype>* layer = layers[layer_id].get();

		// setup for this layer
		layer->setup(bottom_vecs[layer_id], top_vecs[layer_id]);
		LOG_IF(INFO, Dragon::get_root_solver()) << "Setup Layer: " << layer_param.name();

		for (int top_id = 0; top_id < top_vecs[layer_id].size(); top_id++){
			//	extend size to max number of blobs if necessary
			if (blobs_loss_weight.size() <= top_id_vecs[layer_id][top_id])
				blobs_loss_weight.resize(top_id_vecs[layer_id][top_id] + 1, Dtype(0));
			//	store global loss weights from each layer each blob
			blobs_loss_weight[top_id_vecs[layer_id][top_id]] = layer->getLoss(top_id);
			LOG_IF(INFO, Dragon::get_root_solver())
				<< "Top shape: " << top_vecs[layer_id][top_id]->shape_string();
			if (layer->getLoss(top_id)) LOG_IF(INFO, Dragon::get_root_solver())
				<< "	with loss weight " << layer->getLoss(top_id);
			//	sum up for training parameter statistic
			memory_used += top_vecs[layer_id][top_id]->count();
		}

		//	search from all processors
		need_bp= layers[layer_id]->mpiSyncFlag(need_bp);

		const int param_size = layer_param.param_size();
		//	blobs_size will be set after layer->setup()
		const int param_blobs_size = layer->getBlobs().size();
		CHECK_LE(param_size, param_blobs_size)<< "Too many params specify for layer.";
		//	use if do not specify hyperparameter
		//	lr_mult=decay_mult=1.0
		ParamSpec default_hyperparameter;
		for (int param_id = 0; param_id < param_blobs_size; param_id++){
			const ParamSpec* hyperparameter = param_id < param_size ?
				&layer_param.param(param_id) : &default_hyperparameter;
			const bool param_need_bp = hyperparameter->lr_mult() != 0;
			//	check whether a param blob need back propogation [default=true]
			need_bp |= param_need_bp;
			layer->setParamNeedBp(param_id, param_need_bp);
		}
		//	stuff param blobs
		for (int param_id = 0; param_id < param_blobs_size; param_id++)
			appendParam(param, layer_id, param_id);

		layer_need_backward.push_back(need_bp);
		//	after checking all bottom blobs and param blobs
		if (need_bp)
			for (int top_id = 0; top_id < top_id_vecs[layer_id].size(); top_id++)
				blobs_need_backward[top_id_vecs[layer_id][top_id]] = true;
	}	//	end layer_id

	set<string> blobs_under_loss, blobs_skip_bp;
	for (int layer_id = layers.size()-1; layer_id >= 0; layer_id--){
		bool layer_contributes_loss = false;
		bool layer_skip_bp = true;
		Layer<Dtype>* layer = layers[layer_id].get();
		for (int top_id = 0; top_id < top_vecs[layer_id].size(); top_id++){
			const string& blob_name = blobs_name[top_id_vecs[layer_id][top_id]];
			if (layer->getLoss(top_id) || blobs_under_loss.count(blob_name))
				layer_contributes_loss = true;
			if (!blobs_skip_bp.count(blob_name)) layer_skip_bp = false;
			//	find any top blobs if affected by loss and do not force to skip bp
			if (layer_contributes_loss&&!layer_skip_bp) break;
		}

		//	search from all processors
		layer_contributes_loss = layers[layer_id]->mpiSyncFlag(layer_contributes_loss);
		layer_skip_bp = !layers[layer_id]->mpiSyncFlag(!layer_skip_bp);

		//	optimization trick:	set lr_mult but is not affected by loss
		if (layer_need_backward[layer_id] && layer_skip_bp){
			//	cancel layer
			layer_need_backward[layer_id] = false;
			//	cancel bottom
			for (int bottom_id = 0; bottom_id < bottom_vecs[layer_id].size(); bottom_id++){
				bottoms_need_backward[layer_id][bottom_id] = false;
			}
		}
		//	cancel directly if layer is not affected by loss
		if (!layer_contributes_loss) layer_need_backward[layer_id] = false;
		//	debug info
		if (Dragon::get_root_solver()||Dragon::get_arch()==Dragon::DEVICE){
			if (layer_need_backward[layer_id])
				cout << "Layer: " << layer_names[layer_id] << " need back-propogation." << endl;
			else cout << "Layer: " << layer_names[layer_id] << " does not need back-propogation." << endl;
		}
		//	if one top blob affected by loss
		//	all bottom blobs will be affected
		//	regard it as "loss back-affected"
		for (int bottom_id = 0; bottom_id < bottom_vecs[layer_id].size(); bottom_id++){
			const string& blob_name = blobs_name[bottom_id_vecs[layer_id][bottom_id]];
			if (layer_contributes_loss) blobs_under_loss.insert(blob_name);
			else bottoms_need_backward[layer_id][bottom_id] = false;
			//	use for optimization trick : skip all bottom blobs
			if (!bottoms_need_backward[layer_id][bottom_id]) blobs_skip_bp.insert(blob_name);
		}
	}	//	end layer id
	if (param.force_backward()){
		for (int layer_id = 0; layer_id < layers.size(); layer_id++){
			layer_need_backward[layer_id] = true;
			for (int bottom_id = 0; bottom_id < bottom_vecs[layer_id].size(); bottom_id++){
				//	set for bottoms
				bottoms_need_backward[layer_id][bottom_id] = 
					bottoms_need_backward[layer_id][bottom_id]||layers[layer_id]->allowForceBackward(bottom_id);
				//	set for blobs
				blobs_need_backward[bottom_id_vecs[layer_id][bottom_id]] = 
					blobs_need_backward[bottom_id_vecs[layer_id][bottom_id]]||bottoms_need_backward[layer_id][bottom_id];
			}
			//	set for params
			for (int param_id = 0; param_id < layers[layer_id]->getBlobs().size(); param_id++){
				layers[layer_id]->setParamNeedBp(param_id, true);
			}
		}
	}
	//	move un-used(declare top but not use as bottom) blobs into output blobs
	//	usually contain loss blobs
	for (set<string>::iterator i = available_blobs.begin(); i != available_blobs.end(); i++){
		LOG_IF(INFO, Dragon::get_root_solver())
			<< "Network produces output: " << *i;
		net_output_blobs.push_back(blobs[blob_name_to_idx[*i]].get());
		net_output_blob_indices.push_back(blob_name_to_idx[*i]);
	}
	//	store blob_name -> blob_ids
	blobs_name_idx = blob_name_to_idx;
	//	store layer_name -> layer_id
	for (size_t layer_id = 0; layer_id < layer_names.size(); layer_id++)
		layers_name_idx[layer_names[layer_id]] = layer_id;

	const vector<Blob<Dtype>*> p = getLearnableParams();
	int param_mem_used = 0;
	for (int i = 0; i < p.size(); i++) param_mem_used += p[i]->count();
	LOG_IF(INFO, Dragon::get_root_solver())
		<< "Memory required for Data: " << memory_used*sizeof(Dtype);
	LOG_IF(INFO, Dragon::get_root_solver())
		<< "Memory required for Param: " << param_mem_used*sizeof(Dtype);
	LOG_IF(INFO, Dragon::get_root_solver()) << "Network Initializion done.";
}

template <typename Dtype>
Dtype Net<Dtype>::forwardFromTo(int start, int end){
	CHECK_GE(start, 0);
	CHECK_LT(end, layers.size());
	Dtype tot_loss = 0;
	for (int i = start; i <= end; i++){
		Dtype layer_loss = layers[i]->forward(bottom_vecs[i], top_vecs[i]);
		tot_loss += layer_loss;
	}
	return tot_loss;
}

template <typename Dtype>
Dtype Net<Dtype>::forwardFrom(int start){
	return forwardFromTo(start, layers.size() - 1);
}

template <typename Dtype>
Dtype Net<Dtype>::forwardTo(int end){
	return forwardFromTo(0, end);
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::forward(Dtype* loss = NULL){
	if (loss != NULL) *loss = forwardFromTo(0, layers.size() - 1);
	else forwardFromTo(0, layers.size() - 1);
	//	usually return empty vector
	return net_output_blobs;
}

//	clear param diffs, used in Solver::step()
template <typename Dtype>
void Net<Dtype>::clearParamDiffs(){
	for (int i = 0; i < learnable_params.size(); i++){
		Blob<Dtype>* blob = learnable_params[i];
		switch (Dragon::get_mode()){
			case Dragon::CPU:
				dragon_set(blob->count(), (Dtype)0, blob->mutable_cpu_diff());
				break;
			case Dragon::GPU:
#ifndef CPU_ONLY
				dragon_gpu_set(blob->count(), (Dtype)0, blob->mutable_gpu_diff());
				break;
#endif
		}
	}
}

template <typename Dtype>
void Net<Dtype>::shareTrainedLayerWith(const Net* other){
	int num_source_layers = other->getLayers().size();
	for (int i = 0; i < num_source_layers; i++){
		Layer<Dtype>* source_layer = other->getLayers()[i].get();
		const string& source_layer_name = other->getLayerNames()[i];
		int target_layer_id = 0;
		//	search for the same layer name
		while (target_layer_id != layer_names.size() &&
			layer_names[target_layer_id] != source_layer_name){
			target_layer_id++;
		}
		//	ignore this layer
		if (target_layer_id == layer_names.size()) continue;
		//	need not use learnable_params
		//	shared blobs have been set through pointer in shareWeights()
		const vector < boost::shared_ptr<Blob<Dtype> > >& target_blobs = layers[target_layer_id]->getBlobs();
		const vector < boost::shared_ptr<Blob<Dtype> > >& source_blobs = source_layer->getBlobs();
		CHECK_EQ(target_blobs.size(), source_blobs.size())
			<< "Test net use layer: " << source_layer_name << " has incompatible number of blobs.";
		for (int j = 0; j < source_blobs.size(); j++){
			Blob<Dtype>* source_blob = source_blobs[j].get();
			Blob<Dtype>* target_blob = target_blobs[j].get();
			CHECK(source_blob->shape() == target_blob->shape())
				<< "Incompatible shape when sharing trained params.";
			target_blob->shareData(*source_blob);
		}
	}
}

template <typename Dtype>
void Net<Dtype>::copyTrainedLayerFrom(const NetParameter& param){
	int num_layers = param.layer_size();
	for (int i = 0; i < num_layers; i++){
		const LayerParameter& source_layer = param.layer(i);
		const string& source_layer_name = source_layer.name();
		int target_layer_id = 0;
		while (target_layer_id != layer_names.size() &&
			layer_names[target_layer_id] != source_layer_name){
			target_layer_id++;
		}
		if (target_layer_id == layer_names.size()) continue;
		const vector < boost::shared_ptr<Blob<Dtype> > >& target_blobs = layers[target_layer_id]->getBlobs();
		for (int j = 0; j < target_blobs.size(); j++){
			Blob<Dtype> source_blob;
			source_blob.FromProto(source_layer.blobs(j));
			Blob<Dtype>* target_blob = target_blobs[j].get();
			//cout << source_blob.shape_string() << endl;
			//cout << target_blob->shape_string() << endl;
			CHECK(source_blob.count() == target_blob->count())
				<< "Incompatible shape when sharing trained params.";
			target_blob->FromProto(source_layer.blobs(j), false);
		}
	}
}

template <typename Dtype>
void Net<Dtype>::copyTrainedLayerFrom(const string& filename){
	NetParameter net_param;
	readProtoFromBinaryFileOrDie(filename.c_str(), &net_param);
	copyTrainedLayerFrom(net_param);
}


template <typename Dtype>
void Net<Dtype>::backwardFromTo(int start, int end){
	CHECK_GE(end, 0);
	CHECK_LT(start, layers.size());
	for (int i = start; i >= end; i--){
		if (layer_need_backward[i])
			layers[i]->backward(top_vecs[i], bottoms_need_backward[i], bottom_vecs[i]);
	}
}

template <typename Dtype>
void Net<Dtype>::backwardFrom(int start){
	backwardFromTo(start, 0);
}

template <typename Dtype>
void Net<Dtype>::backwardTo(int end){
	backwardFromTo(layers.size() - 1, end);
}

template <typename Dtype>
void Net<Dtype>::backward(){
	backwardFromTo(layers.size() - 1, 0);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff = false) const{
	//	why not copy NetParameter
	param->Clear();
	param->set_name(this->name);
	for (int i = 0; i < layers.size(); i++){
		LayerParameter* layer_param = param->add_layer();
		layers[i]->ToProto(layer_param, write_diff);
	}
}


INSTANTIATE_CLASS(Net); 