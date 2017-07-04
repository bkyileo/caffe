#include <vector>
#include <string>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/param.hpp"
#include "caffe/util/helper.hpp"


using std::vector;
using caffe::Blob;
using caffe::Net;
using caffe::FrcnnParam;
using caffe::Point4f;
using caffe::BBox;

class Detector {
public:
  Detector(std::string &proto_file, std::string &model_file){
    Set_Model(proto_file, model_file);
  }
  void Set_Model(std::string &proto_file, std::string &model_file);
  void predict(const cv::Mat &img_in, vector<BBox<float> > &results);
  void predict_original(const cv::Mat &img_in, vector<BBox<float> > &results);
  void predict_iterative(const cv::Mat &img_in, vector<BBox<float> > &results);
private:
  void preprocess(const cv::Mat &img_in, const int blob_idx);
  void preprocess(const vector<float> &data, const int blob_idx);
  vector<boost::shared_ptr<Blob<float> > > predict(const vector<std::string> blob_names);
  boost::shared_ptr<Net<float> > net_;
  float mean_[3];
  int roi_pool_layer;
};
inline void Set_Config(std::string default_config) {
    caffe::FrcnnParam::load_param(default_config);
    caffe::FrcnnParam::print_param();
}
