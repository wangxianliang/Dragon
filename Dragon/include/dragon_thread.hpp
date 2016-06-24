#ifndef DRAGON_THREAD_HPP
#define DRAGON_THREAD_HPP

#include "common.hpp"

class DragonThread
{
public:
	DragonThread() {}
	virtual ~DragonThread();
	void initializeThread(int device, Dragon::Mode mode, Dragon::Arch arch,int rand_seed, int solver_count, bool root_solver);
	void startThread();
	void stopThread();
	//the interface implements for specific working task 
	virtual void interfaceKernel() {}
	bool is_start();
	bool must_stop();
	boost::shared_ptr<boost::thread> thread;
};
#endif

