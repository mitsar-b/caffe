#include <vector>

#include "caffe/layers/select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 2);
  CHECK_EQ(bottom.size(), 2);
  first_reshape_ = true;
  random_ = true;
}

template <typename Dtype>
void SelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  indices_to_forward_.clear();
  labels_.clear();
  int nsize = bottom[0]->shape(0);
  indices_to_forward_.resize(nsize,0);
  labels_.resize(nsize,0);//resize and set all element to 0
  switch (this->layer_param_.select_param().type()) {
    case SelectParameter_Type_Random:
      // Create random numbers
      caffe_rng_bernoulli(nsize, 0.5, &indices_to_forward_[0]);
      std::copy(indices_to_forward_.begin(),indices_to_forward_.end(),labels_.begin());
      break;
    case SelectParameter_Type_Cross:
      caffe_set(nsize,Dtype(0),&labels_[0]);
      std::fill(indices_to_forward_.begin(),indices_to_forward_.end(),1);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
  }

  //top data NxCxHxW
  int num_axes = bottom[0]->num_axes();
  vector<int> shape_top(num_axes);
  for (int ts = 0; ts < num_axes; ++ts)
    shape_top[ts] = bottom[0]->shape(ts);
  top[0]->Reshape(shape_top);

  //top label Nx1
  vector<int> shape_label_top(2);
  shape_label_top[0] = bottom[0]->shape(0);
  shape_label_top[1] = 1;
  top[1]->Reshape(shape_label_top);
}

template <typename Dtype>
void SelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // For all items in batch N
  for (int i = 0; i < bottom[0]->shape(0);i++) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* top_label_data = top[1]->mutable_cpu_data();
    const Dtype* bottom_data = bottom[indices_to_forward_[i]]->cpu_data();
    int dim = bottom[0]->count() / bottom[0]->shape(0);
    int data_offset = i * dim;
    caffe_copy(dim, bottom_data + data_offset,
          top_data + data_offset);
    top_label_data[i] = labels_[i];
  }

}

template <typename Dtype>
void SelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < bottom.size(); i++ ) {
        if (propagate_down[i]) {
          const int dim = top[0]->count() / top[0]->shape(0); //data dimensions
          for (int n = 0; n < bottom[i]->shape(0); n++) { //over all N
            int data_offset = n * dim;
            if (indices_to_forward_[n] != i) { //if data was not taken from this bottom layer
              caffe_set(dim, Dtype(0),
                  bottom[i]->mutable_cpu_diff() + data_offset); //set diff to 0
            } else {
                caffe_copy(dim, top[0]->mutable_cpu_diff() + data_offset,
                    bottom[i]->mutable_cpu_diff() + data_offset); //propagate diff
            }
          }
        }
  }
}

/*#ifdef CPU_ONLY
STUB_GPU(SelectLayer);
#endif*/

INSTANTIATE_CLASS(SelectLayer);
REGISTER_LAYER_CLASS(Select);

}  // namespace caffe
