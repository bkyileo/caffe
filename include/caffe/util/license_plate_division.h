/*******************************************
*Author:caoyu(1.0)
*Date:13/06/2017
*Mail:**@gmail.com
*Copyright:BUPT
********************************************/
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstring>

#ifndef MAIN_CONSOLE_H
#define MAIN_CONSOLE_H
using namespace cv;
using namespace std;
//256 decimal , 3 digits
enum PlateColor{
    Blue = 0,
    Yellow = 1,
    White = 2,
    Black = 3,
    Other = 4
};

class LicensePlate{
    private:
        Mat plate_color_data_;
        Mat plate_gray_data_;
        Mat plate_binary_data_;
        PlateColor color_plate_;
        vector<int> segment_;
        vector<Mat> sub_image_;
    public:
        //LicensePlate();
        void ShowImage(Mat image,string show_image_name);
        void SetPlateColorData(Mat image);
        void SetPlateGrayData(Mat image);
        void SetPlateBinaryData(Mat image);
        void SetColorPlate(PlateColor color);
        void SetSegment(vector<int> segment);
        void SetSubImage(vector<Mat> sub_image);
        PlateColor GetColorPlate();
        Mat GetPlateColorData();
        Mat GetPlateGrayData();
        Mat GetPlateBinaryData();
        vector<int> GetSegment();
        vector<Mat> GetSubImage();
};

void InitLicensePlate(Mat crop_out,LicensePlate& license_plate_data);
void TestColorLicensePlate(LicensePlate& license_plate_data);
void DeleteRivet(LicensePlate& license_plate_data);
void ReSetPosition(LicensePlate& license_plate_data);
void DivisionLicensePlate(LicensePlate& license_plate_data);
void WorkSubImage(LicensePlate& license_plate_data);

#endif
