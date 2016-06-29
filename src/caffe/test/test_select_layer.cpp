#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/select_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SelectLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SelectLayerTest()
      : blob_bottom_1_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_bottom_2_(new Blob<Dtype>(6, 12, 2, 3)),
        blob_top_0_(new Blob<Dtype>()),
        blob_top_1_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    ConstantFiller<Dtype> cfiller(filler_param);
    filler.Fill(this->blob_bottom_1_);
    cfiller.Fill(this->blob_bottom_2_);
    blob_top_vec_0_.push_back(blob_top_0_);
    blob_top_vec_0_.push_back(blob_top_1_);
    blob_top_vec_1_.push_back(blob_top_0_);
    blob_top_vec_1_.push_back(blob_top_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_1_);
    blob_bottom_vec_1_.push_back(blob_bottom_2_);
  }

 /* virtual void ReduceBottomBlobSize() {
    blob_bottom_->Reshape(4, 5, 2, 2);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
  }*/

  virtual ~SelectLayerTest() {
    delete blob_top_0_; delete blob_top_1_;
    delete blob_top_2_; delete blob_bottom_1_;
    delete blob_bottom_2_;
  }

  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  Blob<Dtype>* const blob_top_2_;
  vector<Blob<Dtype>*> blob_top_vec_0_, blob_top_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
};

TYPED_TEST_CASE(SelectLayerTest, TestDtypesAndDevices);

TYPED_TEST(SelectLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SelectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_0_);
  EXPECT_EQ(this->blob_bottom_1_->num(),this->blob_top_0_->num());
  EXPECT_EQ(this->blob_top_0_->num(), this->blob_top_1_->num());
  EXPECT_EQ(this->blob_bottom_1_->channels(), this->blob_top_0_->channels());
  EXPECT_EQ(this->blob_bottom_1_->height(), this->blob_top_0_->height());
  EXPECT_EQ(this->blob_bottom_1_->width(), this->blob_top_0_->width());
  EXPECT_EQ(this->blob_top_1_->channels(), 1);
  EXPECT_EQ(this->blob_top_1_->height(), 1);
  EXPECT_EQ(this->blob_top_1_->width(), 1);
}

TYPED_TEST(SelectLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SelectLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_1_, this->blob_top_vec_0_);
  layer.Forward(this->blob_bottom_vec_1_, this->blob_top_vec_0_);
  for (int n = 0; n < this->blob_top_0_->num(); ++n) {
    for (int c = 0; c < this->blob_top_0_->channels(); ++c) {
      for (int h = 0; h < this->blob_bottom_1_->height(); ++h) {
        for (int w = 0; w < this->blob_bottom_1_->width(); ++w) {
 //         std::cout<<this->blob_top_1_->data_at(n, 0, 0, 0)<<" "<<this->blob_bottom_1_->data_at(n, c, h, w)<<" "<<this->blob_bottom_2_->data_at(n, c, h, w)<<" "<<this->blob_top_0_->data_at(n, c, h, w)<<"\n";
          if (this->blob_top_1_->data_at(n, 0, 0, 0) == 0 )
          EXPECT_EQ(this->blob_bottom_1_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
          else
          EXPECT_EQ(this->blob_bottom_2_->data_at(n, c, h, w),
                    this->blob_top_0_->data_at(n, c, h, w));
        }
      }
    }
  }
}

}  // namespace caffe
