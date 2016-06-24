#include "solvers/gradient_solver.hpp"

template <typename Dtype>
void AdaDeltaSolver<Dtype>::computeUpdateValue(int param_id, Dtype rate){
	Blob<Dtype>* net_param = this->net->getLearnableParams()[param_id];
	const Dtype lr_mult = this->net->getLrMults()[param_id];
	Dtype eps = this->param.delta();
	Dtype momentum = this->param.momentum();
	// adadelta will ignore base_lr
	Dtype lr = lr_mult;
	const int count = net_param->count();
	switch (Dragon::get_mode()){
	case Dragon::CPU:
		//	history store for E[g^2]
		//	update store for E[delta^2]
		//	history=momentum*history + (1-momentum)*(diff^2)
		//	1. compute diff^2 in temp
		dragon_powx<Dtype>(count, net_param->cpu_diff(), Dtype(2), this->temp[param_id]->mutable_cpu_data());
		//	2. compute history
		dragon_cpu_axpby<Dtype>(count, Dtype(1) - momentum, this->temp[param_id]->cpu_data(),
				momentum, this->history[param_id]->mutable_cpu_data());
		//	3. compute RMS[history] as denominator in temp
		dragon_set<Dtype>(count, eps, this->temp[param_id]->mutable_cpu_data());
		dragon_axpy<Dtype>(count, Dtype(1), this->history[param_id]->cpu_data(),this->temp[param_id]->mutable_cpu_data());
		dragon_powx<Dtype>(count, this->temp[param_id]->cpu_data(), Dtype(0.5), this->temp[param_id]->mutable_cpu_data());
		//	4. compute diff/RMS[history] in diff
		dragon_div<Dtype>(count, net_param->cpu_diff(), this->temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());
		//	5. compute RMS[update] as numerator in temp
		dragon_set<Dtype>(count, eps, this->temp[param_id]->mutable_cpu_data());
		dragon_axpy<Dtype>(count, Dtype(1), this->update[param_id]->cpu_data(), this->temp[param_id]->mutable_cpu_data());
		dragon_powx<Dtype>(count, this->temp[param_id]->cpu_data(), Dtype(0.5), this->temp[param_id]->mutable_cpu_data());
		//	6. compute diff*RMS[update] in diff
		dragon_mul<Dtype>(count, net_param->cpu_diff(), this->temp[param_id]->cpu_data(), net_param->mutable_cpu_diff());
		//	7. compute final diff^2 in temp
		dragon_powx<Dtype>(count, net_param->cpu_diff(), Dtype(2), this->temp[param_id]->mutable_cpu_data());
		//	8. compute update
		dragon_cpu_axpby<Dtype>(count, (1 - momentum), this->temp[param_id]->cpu_data(),
			momentum, this->update[param_id]->mutable_cpu_data());
		//	9. apply learning rate
	    dragon_scal<Dtype>(count, lr, net_param->mutable_cpu_diff());
		break;
	case Dragon::GPU:
#ifndef CPU_ONLY
		adadeltaUpdate(net_param->count(),net_param->mutable_gpu_diff(),
			this->history[param_id]->mutable_gpu_data(),this-> update[param_id]->mutable_gpu_data(),momentum, eps, lr);
#endif
		break;
	default:LOG(FATAL) << "Unknown mode: " << Dragon::get_mode();
	}
}

template <typename Dtype>
void AdaDeltaSolver<Dtype>::applyUpdate(){
	CHECK(Dragon::get_root_solver());
	Dtype rate = getLearningRate();
	//	AdaDelta do not need base lr
	if (this->param.display() && this->iter%this->param.display() == 0)
		cout << "Iteration " << this->iter << ", lr = AdaDelta";
	clipGradients();
	vector<Blob<Dtype>*> net_params = this->net->getLearnableParams();
	for (int i = 0; i < net_params.size(); i++){
		normalize(i);
		regularize(i);
		computeUpdateValue(i, rate);
		net_params[i]->update();
	}
}

INSTANTIATE_CLASS(AdaDeltaSolver);