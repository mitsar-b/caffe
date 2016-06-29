#ifndef CAFFE_SELECT_LAYER_HPP_
#define CAFFE_SELECT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 */
template <typename Dtype>
class SelectLayer : public Layer<Dtype> {
 public:
  explicit SelectLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Select"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 2+)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 //   const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the forwarded inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool first_reshape_;
  bool random_;
  vector<unsigned int> indices_to_forward_;
  vector<Dtype> labels_;
};

}  // namespace caffe

#endif  // CAFFE_SELECT_LAYER_HPP_
