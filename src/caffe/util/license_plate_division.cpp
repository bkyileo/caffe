/************************************************
*Author:caoyu(1.0)
*Date:14/06/2017
*Mail:**@gmail.com
*Copyright:BUPT
************************************************/
#include "caffe/util/license_plate_division.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstring>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define debug

void LicensePlate::ShowImage(Mat image_mat,string show_image_name){
    IplImage copy_image=image_mat;
    IplImage* image = &copy_image;
    cvNamedWindow(show_image_name.c_str());
    cvShowImage(show_image_name.c_str(),image);
    cvWaitKey(0);
}
void LicensePlate::SetPlateColorData(Mat image){
    plate_color_data_=image;    
}
void LicensePlate::SetPlateGrayData(Mat image){
    plate_gray_data_=image;
}
void LicensePlate::SetPlateBinaryData(Mat image){
    plate_binary_data_=image;
}
void LicensePlate::SetColorPlate(PlateColor color){
    color_plate_=color;
}
void LicensePlate::SetSegment(vector<int> segment){
    segment_=segment;
}
void LicensePlate::SetSubImage(vector<Mat> sub_image){
    sub_image_=sub_image;
}
PlateColor LicensePlate::GetColorPlate(){
    return color_plate_;
}
Mat LicensePlate::GetPlateColorData(){
    return plate_color_data_;
}
Mat LicensePlate::GetPlateGrayData(){
    return plate_gray_data_;
}
Mat LicensePlate::GetPlateBinaryData(){
    return plate_binary_data_;
}
vector<int> LicensePlate::GetSegment(){
    return segment_;
}
vector<Mat> LicensePlate::GetSubImage(){
    return sub_image_;
}
//hsv model from wikipedia
int GetColor(int r,int g,int b){
    int mx=max(max(r,g),b);
    int mi=min(min(r,g),b);
    double h,s,v=1.0*mx/255;
    if(mx==mi)h=0;
    else if(mx==r&&g>=b)h=60.0*(g-b)/(mx-mi);
    else if(mx==r&&g<b)h=60.0*(g-b)/(mx-mi)+360;
    else if(mx==g)h=60.0*(b-r)/(mx-mi)+120;
    else if(mx==b)h=60.0*(r-g)/(mx-mi)+240;
    if(mx==0)s=0;
    else s=1-1.0*mi/mx;
    if(v<0.2)return 3;
    if(s<0.3&&mi>=100)return 2;
    if(h>=20&&h<=100)return 1;
    if(h>=200&&h<=280)return 0;
    return 4;
}
void InitLicensePlate(Mat image_path,LicensePlate &license_plate_data){
    //Mat image_data=imread(image_path.c_str());
    Mat image_data = image_path;
    license_plate_data.SetPlateColorData(image_data);
    Mat image_gray;
    cvtColor(license_plate_data.GetPlateColorData(),image_gray,CV_BGR2GRAY);
    license_plate_data.SetPlateGrayData(image_gray);
    Mat image_binary;
    threshold(image_gray, image_binary, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    license_plate_data.SetPlateBinaryData(image_binary);
}
void TestColorLicensePlate(LicensePlate &license_plate_data){
    Mat data_copy=license_plate_data.GetPlateColorData();
    Mat data=data_copy(Rect_<double>(data_copy.cols*0.1,data_copy.rows*0.1,data_copy.cols*0.8,data_copy.rows*0.8));
    int num[5]={0};
    for(int i=0;i<data.rows;i++){
        for(int j=0;j<data.cols;j++){
           int b=data.at<cv::Vec3b>(i,j)[0];
           int g=data.at<cv::Vec3b>(i,j)[1];
           int r=data.at<cv::Vec3b>(i,j)[2];
           int color_type=GetColor(r,g,b);
           num[color_type]++;/*
           if(color_type==0){
               data.at<Vec3b>(i,j)[0]=255;
               data.at<Vec3b>(i,j)[1]=0;
               data.at<Vec3b>(i,j)[2]=0;
           }else if(color_type==2){
               data.at<Vec3b>(i,j)[0]=255;
               data.at<Vec3b>(i,j)[1]=255;
               data.at<Vec3b>(i,j)[2]=255;
           }*/
        }
    }
    //license_plate_data.ShowImage(data,"work");
    int white_num=num[2];
    int black_num=num[3];
    num[0]+=num[2];
    num[1]+=num[3];
    num[2]+=num[3];
    num[3]=num[2];
    int mx=0;
    int ip_mx=0;
    for(int i=0;i<5;i++){
        if(mx<num[i]){
            mx=num[i];
            ip_mx=i;
        }
    }
    if(ip_mx==2||ip_mx==3){
        mx=max(white_num,black_num);
        if(mx==white_num)ip_mx=2;
        else ip_mx=3;
    }
    string color_name[5]={"blue","yellow","white","black","other"};
    //cout<<"color:"<<color_name[ip_mx]<<endl;
    PlateColor color=(PlateColor)ip_mx;
    license_plate_data.SetColorPlate(color);
    if(ip_mx==1||ip_mx==2){
        data=license_plate_data.GetPlateBinaryData();
        for(int i=0;i<data.rows;i++){
            for(int j=0;j<data.cols;j++){
                data.at<uchar>(i,j)=255-data.at<uchar>(i,j);
            }
        }
        license_plate_data.SetPlateBinaryData(data);
    }
    return;
}
void DeleteRivet(LicensePlate &license_plate_data){
    Mat data=license_plate_data.GetPlateBinaryData();
    for(int i=0;i<data.rows;i++){
        int jump_num=0;
        int white_num=0;
        for(int j=1;j<data.cols;j++){
            if(data.at<char>(i,j)!=data.at<char>(i,j-1)){
                jump_num++;
            }
            if(data.at<uchar>(i,j)==255){
                white_num++;
            }
        }
        if(jump_num<=5||white_num>data.cols*0.6||white_num<data.cols*0.1){
            for(int j=1;j<data.cols;j++){
                data.at<char>(i,j)=0;
            }
        }
    }
    license_plate_data.SetPlateBinaryData(data);
}
void ReSetPosition(LicensePlate &license_plate_data){
    Mat data=license_plate_data.GetPlateBinaryData();
    int up_x=0,down_x=data.rows-1;
    for(int i=0;i<data.rows;i++){
        int white_num=0;
        for(int j=0;j<data.cols;j++){
            if(data.at<uchar>(i,j)==255){
                white_num++;
            }
        }
        if(white_num==0){
            if(i<data.rows/2){
                up_x=i;
            }else{
                down_x=i;
                break;
            }
        }
    }
    Mat data_copy=data(Rect_<double>(0,up_x,data.cols,down_x-up_x+1));
    //license_plate_data.ShowImage(data_copy,"binary_image");
}
void DivisionLicensePlate(LicensePlate &license_plate_data){
    Mat data=license_plate_data.GetPlateBinaryData();
    //cout<<"rows:"<<data.rows<<endl;
    //cout<<"cols:"<<data.cols<<endl;
    vector<int>white_num_cols;
    for(int i=0;i<data.cols;i++){
        int white_num=0;
        for(int j=0;j<data.rows;j++){
            if(data.at<uchar>(j,i)==255){
                white_num++;
            }
        }
        white_num_cols.push_back(white_num);
    }
    int len=white_num_cols.size();
    vector<int>segment_white_sum;
    for(int i=0;i<len;i++){
        int sum=white_num_cols[i];
        for(int j=1;j<=2;j++){
            if(i-j>=0)
                sum+=white_num_cols[i-j];
            if(i+j<len)
                sum+=white_num_cols[i+j];
        }
        segment_white_sum.push_back(sum);
    }
    vector<int>point;
    for(int i=0;i<len;i++){
        int flag=1;
        for(int j=1;j<=15&&flag;j++){
            if(i-j>=0&&segment_white_sum[i-j]<segment_white_sum[i]){
                flag=0;
            }
        }
        for(int j=1;j<=15&&flag;j++){
            if(i+j<len&&segment_white_sum[i+j]<segment_white_sum[i]){
                flag=0;
            }
        }
        if(flag){
            point.push_back(i);
        }
    }
    point.push_back(data.cols-1);
    len=point.size();
    vector<int>segment;
    for(int i=1;i<len;i++){
        if(point[i]-point[i-1]>7){
            segment.push_back(point[i-1]);
            segment.push_back(point[i]);
        }
    }
    license_plate_data.SetSegment(segment);
}
void WorkSubImage(LicensePlate &license_plate_data){
    vector<int> segment=license_plate_data.GetSegment();
    Mat data=license_plate_data.GetPlateBinaryData();
    int len=segment.size();
    vector<Mat>sub_image;
    for(int i=1;i<len;i+=2){
        Mat tmp_image=data(Rect_<double>(segment[i-1],0,segment[i]-segment[i-1]+1,data.rows));
        sub_image.push_back(tmp_image);
    }
    len=sub_image.size();
    while(len<7){
        vector<Mat>tmp_sub_image;
        int mx=0,ip=0;
        for(int i=0;i<len;i++){
            if(mx<sub_image[i].cols){
                ip=i;
                mx=sub_image[i].cols;
            }
        }
        for(int i=0;i<ip;i++){
            tmp_sub_image.push_back(sub_image[i]);
        }
        Mat tmp_image=sub_image[ip](Rect_<double>(0,0,mx/2,data.rows));
        tmp_sub_image.push_back(tmp_image);
        tmp_image=sub_image[ip](Rect_<double>(mx/2-1,0,mx/2,data.rows));
        tmp_sub_image.push_back(tmp_image);
        for(int i=ip+1;i<len;i++){
            tmp_sub_image.push_back(sub_image[i]);
        }
        sub_image=tmp_sub_image;
        len=sub_image.size();
    }
    if(len>7){
        int mi=100000,ip=0;
        for(int i=0;i<len;i++){
            if(mi>sub_image[i].cols){
                mi=sub_image[i].cols;
                ip=i;
            }
        }
        int mx=0;
        for(int i=0;i<sub_image[ip].rows;i++){
            int num=0;
            for(int j=0;j<sub_image[ip].cols;j++){
                if(sub_image[ip].at<uchar>(i,j)==0){
                    mx=max(mx,num);
                    num=0;
                }else{
                    num++;
                }
            }
        }
        if(2*mx<sub_image[ip].rows){
            vector<Mat>tmp;
            for(int i=0;i<ip;i++){
                tmp.push_back(sub_image[i]);
            }
            for(int i=ip+1;i<len;i++){
                tmp.push_back(sub_image[i]);
            }
            sub_image=tmp;
        }
    }
    len=sub_image.size();
    vector<Mat>resize_image;
    for(int i=0;i<len;i++){
	Mat crop_out;
	cv::resize(sub_image[i],crop_out,cv::Size(28,28),0,0,CV_INTER_LINEAR);
	resize_image.push_back(crop_out);
        //cout<<i<<":"<<sub_image[i].cols<<endl;
	string filename="";
	filename+=(i+'0');
	filename+=".jpg";
	//cout<<filename<<endl;
	//imwrite(filename,crop_out);
//        license_plate_data.ShowImage(sub_image[i],i+".jpg");
    }
    license_plate_data.SetSubImage(resize_image);
}
/*
int main(int argc,char* argv[]){
    string image_path=(string)argv[1];
    LicensePlate license_plate_data ;
    InitLicensePlate(image_path,license_plate_data);
    TestColorLicensePlate(license_plate_data);
//    DeleteRivet(license_plate_data);
    ReSetPosition(license_plate_data);
    DivisionLicensePlate(license_plate_data);
    WorkSubImage(license_plate_data); 
    return 0;
}
*/
