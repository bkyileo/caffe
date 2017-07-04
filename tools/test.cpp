#include <gflags/gflags.h>
#include <glog/logging.h>
#include "boost/algorithm/string.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/vis.hpp"
#include "caffe/util/detector.hpp"
#include "caffe/util/license_plate_division.h"
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <caffe/caffe.hpp>
//using namespace caffe;
using namespace std;
using namespace cv;

typedef std::pair<string, float> Prediction;
class Classifier {
 public:
  Classifier(const string& model_file,const string& trained_file,const string& label_file);
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);
 private:
  std::vector<float> Predict(const cv::Mat& img);
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);
 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  std::vector<string> labels_;
};
Classifier::Classifier(const string& model_file,const string& trained_file,const string& label_file) {
//#ifdef CPU_ONLY
//  Caffe::set_mode(Caffe::CPU);
//#else
//  Caffe::set_mode(Caffe::GPU);
//#endif
  /* Load the network. */
  net_.reset(new Net<float>(model_file, caffe::TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}
static bool PairCompare(const std::pair<float, int>& lhs,const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}
/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
     
  std::vector<int> result;
  for (int i = 0; i < N; ++i)result.push_back(pairs[i].second); 
  return result;
}         
/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);
  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }
  return predictions;
}
std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);
  net_->Forward();
  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)cv::resize(sample, sample_resized, input_geometry_);
  else sample_resized = sample;
  cv::Mat sample_float;
  if (num_channels_ == 3)sample_resized.convertTo(sample_float, CV_32FC3);
  else sample_resized.convertTo(sample_float, CV_32FC1);
  cv::split(sample_float, *input_channels);
  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
int get_cnn_result(Classifier classifier,Mat gray){
    std::vector<Prediction> predictions = classifier.Classify(gray);
  /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
      Prediction p = predictions[i];
      cout<<p.first;
    }
    return 0; 
}

inline std::string INT(float x) { char A[100]; sprintf(A,"%.1f",x); return std::string(A);};
inline std::string FloatToString(float x) { char A[100]; sprintf(A,"%.4f",x); return std::string(A);};
using namespace std;
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

int main(int argc, char** argv){
  //FLAGS_alsologtostderr = 1;
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));


  caffe::Caffe::SetDevice(0);
  caffe::Caffe::set_mode(caffe::Caffe::GPU);

  caffe::GlobalInit(&argc,&argv); 
  LicensePlate license_plate_division;
  return 0;
  //set_Config(default_config_file);
  std::string proto_cnn_file ="/home/bk/caffe/models/deploy.prototxt";
  std::string model_cnn_file = "/home/bk/caffe/models/cnn.caffemodel";
  std::string label_cnn_file = "/home/bk/caffe/models/label.txt";
  //DataPrepare data_load;
  //int count = 0;	
  Classifier classifier(proto_cnn_file, model_cnn_file,label_cnn_file);
  string dir="/home/bk/caffe/test/";
  vector<string>files=vector<string>();
  getdir(dir,files);
  for(unsigned int ct=0;ct<files.size();ct++){
    if(files[ct].size()>3){
    Mat image=imread("/home/bk/caffe/test/"+files[ct]);
    InitLicensePlate(image,license_plate_division);
    for(int i=0;i<image.rows;i++){
      for(int j=0;j<image.cols;j++){
        image.at<Vec3b>(i,j)[0]*=0.00390625;
      }
    }
    std::cout<<files[ct]<<std::endl;
    get_cnn_result(classifier,image);}
  }  
}
