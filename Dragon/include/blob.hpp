# ifndef BLOB_HPP
# define BLOB_HPP
#pragma warning(disable:4244)
#pragma warning(disable:4267)
#pragma warning(disable:4081)
#pragma warning(disable:4996)
#pragma warning(disable:4005)
#pragma warning(disable:4018)
#include <vector>
#include "common.hpp"
#include "protos/dragon.pb.h"
using namespace std;
using namespace boost;


#ifndef NO_MPI
template <typename BlobType>
class ParameterServer ;
#endif

class SyncedMemory;

template <typename Dtype>
class Blob{
public:
	typedef Dtype element_type;
	Blob():data_(),diff_(),count_(0), capacity_(0) {}
	Blob(const vector<int>& shape) :count_(0),capacity_(0) { reshape(shape); }
	void reshape(int num, int channels, int height, int width);
	void reshape(vector<int> shape);
	void reshape(const BlobShape& blob_shape);
	void reshapeLike(const Blob& blob);
	const Dtype* cpu_data() const;
	void set_cpu_data(Dtype *data);
	const Dtype *gpu_data() const;
	void set_gpu_data(Dtype *data);
	const Dtype* cpu_diff() const;
	const Dtype* gpu_diff() const;
	Dtype *mutable_cpu_data();
	Dtype *mutable_gpu_data();
	Dtype *mutable_cpu_diff();
	Dtype *mutable_gpu_diff();
	//	Normal Update
	void update();
	//	ParameterServer Update
	void update(int param_id);
	Dtype asum_data();
	Dtype sumsq_diff() const;
	void scale_diff(Dtype scale_factor);
	int num() const { return shape(0); }
	int channels() const { return shape(1); }
	int height() const { return shape(2); }
	int width() const { return shape(3); }
	int count() const{ return count_; }
	int count(int start_axis, int end_axis) const {
		CHECK_GE(start_axis, 0);
		CHECK_LE(start_axis, end_axis);
		CHECK_LE(start_axis, num_axes());
		CHECK_LE(end_axis, num_axes());
		int cnt = 1;
		for (int i = start_axis; i < end_axis; i++) cnt *= shape(i);
		return cnt;
	}
	int count(int start_axis) const{ return count(start_axis, num_axes()); }
	const vector<int> &shape() const{ return shape_; }
	int shape(int axis) const{ return shape_[canonicalAxisIndex(axis)]; }
	// debug info( e.g. Top shape: 1000 1000 1000 1000 (2333) )s
	string shape_string() const {
		ostringstream stream;
		for (int i = 0; i < shape_.size(); ++i) stream << shape_[i] << " ";
		stream << "(" << count_ << ")";
		return stream.str();
	}
	int offset(const int n, const int c = 0, const int h = 0,
		const int w = 0){
		CHECK_GE(n, 0);
		CHECK_LE(n, num());
		CHECK_GE(channels(), 0);
		CHECK_LE(c, channels());
		CHECK_GE(height(), 0);
		CHECK_LE(h, height());
		CHECK_GE(width(), 0);
		CHECK_LE(w, width());
		return ((n * channels() + c) * height() + h) * width() + w;
	}
	int offset(const vector<int>& vec){
		CHECK_LE(vec.size(), num_axes());
		int offset = 0;
		for (int i = 0; i < num_axes(); i++){
			offset *= shape(i);
			if (vec.size()>i) offset += vec[i];
		}
		return offset;
	}
	int num_axes() const { return shape_.size(); }
	// idx ranges [-axes,axes)
	// idx(-1) means the last axis
	int canonicalAxisIndex(int axis) const{
		//cout << shape_string() << endl;
		CHECK_GE(axis, -num_axes());
		CHECK_LT(axis, num_axes());
		if (axis < 0) return axis + num_axes();
		else return axis;
	}
	const boost::shared_ptr<SyncedMemory>& data() const { return data_; }
	const boost::shared_ptr<SyncedMemory>& diff() const { return diff_; }
	//	change the shared_ptr object and will recycle the memory if need
	void shareData(const Blob& blob) {
		CHECK_EQ(count(), blob.count());
		data_ = blob.data(); 
	}
	void shareDiff(const Blob& blob) {
		CHECK_EQ(count(), blob.count());
		diff_ = blob.diff();
	}
	void FromProto(const BlobProto& proto, bool need_reshape = true);
	void ToProto(BlobProto* proto, bool write_diff = false);
#ifndef NO_MPI
	void setParameterServer(const boost::shared_ptr < ParameterServer< Blob<Dtype> > > ps){
		this->ps = ps;
	}
#endif
protected:
	boost::shared_ptr<SyncedMemory> data_, diff_;
#ifndef NO_MPI
	boost::shared_ptr < ParameterServer< Blob<Dtype> > > ps;
#endif
	vector<int> shape_;
	int count_, capacity_;
};

template<typename Dtype>
class Batch{
public:
	Blob<Dtype> data, label;
};

# endif