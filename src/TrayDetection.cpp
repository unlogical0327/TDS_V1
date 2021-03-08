
#include "TrayDetection.h"
#include <QThread>
#include <QLibrary>
#include <QTimer>
#include <QFile>
#include <QFileInfo>
//#include <QDebug>
#include "threadfromqthread.h"
#include <string>
#include<Windows.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
//#include <sys/sysinfo.h>
//#include <unistd.h>
#include <assert.h>
#define MAX_LEN 128

#if _MSC_VER > 1000  
#pragma once  
#endif // _MSC_VER > 1000  
std::string CPU_ID;
// CNN timer
QTimer *CAN_timer;  // time to 
					// CAN message settings
CAN_Check_Thread CAN_Check_thread;
QTimer *CAN_Check_timer;  // time to update map
CAN_Send_Thread CAN_Send_thread;
QTimer *CAN_Send_timer;  // time to update map
Thread *MyCANControlThread = new Thread;

// QString to receive start msg
QString  receive_str;
QString  receive_str_ID;
QString  SW_ID_str;
bool REV_start_CAN = false;
// Global variables to communicate data result through interface
char frame_ID[3];
char frame_Data[16];
char frame_T_ID[3];
char frame_T_Data[8];
char frame_angle_ID[3];
char frame_angle_Data[4];
char frame_good_ID[3];
char frame_good_left_Data[16];
char frame_good_right_Data[16];
char frame_good_front_Data[16];
char frame_good_angle_ID[3];
char frame_good_angle_Data[4];
char frame_rack_ID[3];
char frame_rack_left_Data[16];
char frame_rack_right_Data[16];

char frame_rack_angle_ID[3];
char frame_rack_angle_Data[4];
int start_status;
int CAN_mode;
int input_option;
bool save_data_file = false;
// define global variables of results
double x_center = 0;
double y_center = 0;
double z_center = 0;
double pose_x = 0;
double pose_y = 0;
double goods_pose_x = 0;
double goods_pose_y = 0;
double tray_dist = 0;
int center_goods;
double left_most_point_goods_x = 0;
double left_most_point_goods_y = 0;
double left_most_point_goods_z = 0;
double right_most_point_goods_x = 0;
double right_most_point_goods_y = 0;
double right_most_point_goods_z = 0;
double front_most_point_goods_x = 0;
double front_most_point_goods_y = 0;
double front_most_point_goods_z = 0;
double camera_facing_eigen_goods = 0;
double left_most_point_rack_x = 0;
double left_most_point_rack_y = 0;
double left_most_point_rack_z = 0;
double right_most_point_rack_x = 0;
double right_most_point_rack_y = 0;
double right_most_point_rack_z = 0;
// Calibration para
double h_angle_offset=2.5;
double x_offset = 0;
double y_offset = 0;
double z_offset = 0;
int Read_online_para = 1;
// results
bool rack_found = false;
bool goods_detect = false;
bool goods_found = false;
int update_status = -1;
int display_option = 0; // default to No display point cloud
using namespace std::chrono_literals;
using namespace cv;
//using namespace dlib;
using namespace std::chrono;

// define a global DNN model
//cv::dnn::Net faster_rcnn_net;
std::string pb_file_name = "./dataSW/100719m.dat"; //ssd_v1_tray_inception.pb = 092619m.dat
std::string pbtxt = "./dataSW/100719t.dat"; //cvssd_v1_txt_graph.pbtxt = 092619t.dat
dnn::Net faster_rcnn_net;
bool DNN_init = false;

// define global PC data for multiple threads
pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_read(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_preprocess(new pcl::PointCloud<pcl::PointXYZ>);
// preprocess timer
QTimer pre_pcd_timer;
Preprocess_pcd_Thread pre_pcd_thread;
bool pre_pcd_flag = false;
bool pcd_ready = false;
// RGB timer
QTimer RGB_detect_timer;
RGB_detect_Thread rgb_detect_thread;
bool rgb_detect_flag = false;

std::string SN_str;
std::string FW_str;
std::string HW_str;

//typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
float pixel_scale = 5; //unit is mm, this is a fake thing for early validation. not used in mainline SW. 

bool global_debug_option = false; //once enable, point cloud is displayed all over APIs. 
bool global_tray_rack_dis_calc = true; //parameter to tell customer stuff on/around the tray and their vertex
bool global_force_2D_tracting_AON = false; //once enabled, will always try to use 2D object detection first. If fail wake up RCNN. need better way to enable terminate. may make error
bool global_display_output_cloud = false; //once enabled, every success calibration will show cloud picture and tracking point. debug only. shall be false for mainline sw. 
bool global_2D_tracker_control = false;// since ssd is so fast. we can disable the two D tracker and only rely on ssd.
bool global_show_key_points_nearby_objects = false;
float global_tray_rack_z_offset = 0.2; //unit m detect +- 0.2m deeper from traycenterAB
float global_tray_rack_x_offset = 1;//unit m detect +-1m wide from traycenterAB
float global_tray_rack_y_offset = 1.5;//unit m. detect 1.5m higher than traycenterAB
float global_cal_confidence = 0.0;
float global_full_confidence = 3.0;// this is how many optional checks we conducted. 3.0 means 3 optional checks.
								   // Mode Setting
bool hog_mode = false;
bool camera_input = false;
int cam_test_mode = 0;
// Apply settings to pico_params
PsReturnStatus status;
int32_t deviceIndex = 0;
int32_t deviceCount = 0;
uint32_t slope = 1450;
uint32_t wdrSlope = 4400;
PsDepthRange depthRange = PsFarRange;   //PsNearRange;
int32_t dataMode = PsDepthAndRGB_30; // DepthAndIRAndRGB_30
									 //int32_t dataMode = PsDepthAndIRAndRGB_30;
PsCameraParameters cameraParameters0; // RGB sensor
PsCameraParameters cameraParameters1; //Depth sensor
PsReturnStatus status1; // = PsGetCameraParameters(deviceIndex, PsDepthSensor, &cameraParameters1);
PsReturnStatus status2; // = PsGetCameraParameters(deviceIndex, PsRgbSensor, &cameraParameters0);
PsCameraExtrinsicParameters CameraExtrinsicParameters;
PsReturnStatus status3; // = PsGetCameraExtrinsicParameters(deviceIndex, &CameraExtrinsicParameters);
cv::Mat RGBimageMat;
cv::Mat depthimageMat;
cv::Mat mappedRGBimageMat;
cv::Mat rgb_image;

const std::string irImageWindow = "IR Image";
const std::string rgbImageWindow = "RGB Image";
const std::string depthImageWindow = "Depth Image";
const std::string mappedDepthImageWindow = "MappedDepth Image";
const std::string mappedRgbImageWindow = "MappedRGB Image";
const std::string mappedIRWindow = "MappedIR Image";
const std::string wdrDepthImageWindow = "WDR Depth Image";
// define frames for RGB depth and IR camera
PsFrame depthFrame = { 0 };
PsFrame irFrame = { 0 };
PsFrame rgbFrame = { 0 };
PsFrame mappedDepthFrame = { 0 };
PsFrame wdrDepthFrame = { 0 };
PsFrame mappedRGBFrame = { 0 };
PsFrame mappedIRFrame = { 0 };
bool pico_init_done = false;
int iter_index = 0;
// Harbin camera
float camera_cal_cx = 341.169; //will be online updated at pico-read 
float camera_cal_cy = 174.034; //will be online updated at pico-read 
float camera_cal_fx = 462.596; //will be online updated at pico-read 
float camera_cal_fy = 462.64; //will be online updated at pico-read 
cv::Mat camera_mappedRGBMat; //this will be the mapped image container for all backend calculations
float camera_depthmat_480_640[480][640]; //this will be the depth data container for all backend calculations
										 //offline data shall be able to testing consecutive data frames. in CNN_MAIN function two vectors created 
										 //this is for testing faster 2D detection object tracking other than calling RCNN every time. 
std::string rgb_igb_name = "../test_folder/mappedRGB/mappedRGB1.jpg";
std::string depthDatafile = "../test_folder/depthData/DepthData1.txt";
//to be updated later. 
void offline_debug_update_customer_camera_parameters() {
	//todo do we need better way of updating offline camera parameters?
}
std::string now_str()
{
	// Get current time from the clock, using microseconds resolution
	const boost::posix_time::ptime now =
		boost::posix_time::microsec_clock::local_time();
	// Get the time offset in current day
	const boost::posix_time::time_duration td = now.time_of_day();
	//
	const long hours = td.hours();
	const long minutes = td.minutes();
	const long seconds = td.seconds();
	const long milliseconds = td.total_milliseconds() -
		((hours * 3600 + minutes * 60 + seconds) * 1000);

	char buf[40];
	sprintf(buf, "%02ld:%02ld:%02ld.%03ld",
		hours, minutes, seconds, milliseconds);

	return buf;
}

//this structure describes goods/left rack/right rack property. 

struct ThingNotTray {
	pcl::PointXYZ center;
	pcl::PointXYZ left_most_point;
	pcl::PointXYZ right_most_point;
	pcl::PointXYZ front_most_point;
	pcl::ModelCoefficients camera_facing_eigen;
	bool goods = false;
};

bool next_iteration = false;

//visualizer for xyz format
void doubleCloudViewer(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud1st, std::string cloud_name_1, pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2nd, std::string cloud_name_2)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_crop1(cloud1st, 255, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rgb_2nd(cloud2nd, 0, 255, 0);
	viewer.addPointCloud<pcl::PointXYZ>(cloud1st, rgb_crop1, cloud_name_1);
	viewer.addPointCloud<pcl::PointXYZ>(cloud2nd, rgb_2nd, cloud_name_2);
	viewer.addCoordinateSystem();

	//viewer.addPointCloud (cropcloud,"body");
	//viewer.registerPointPickingCallback (pointPickingEventOccurred, (void*)&viewer);
	viewer.spin();
}

void Load_cali_para(QString &fileName)
{
	// load calibration para from config file
	// Open file with file name
	QFile file(fileName);
	// Read file content line by line     
	char buffer[250];
	char paras_h[50];
	char value_h[50];
	QFileInfo fileInfo(file.fileName());
	const int header_size = 4;
	std::string data_capture_header[header_size] = { "#x_offset(mm)", "#y_offset(mm)", "#z_offset(mm)","#horizontal angle_offset(degree)"};
	std::string filename_str = fileName.toStdString();
	cout << "cali_model file name: "<< filename_str <<endl;
	cout<< file.exists() <<endl;
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		cout << "Open failed." << endl;
	}
	int j = 0;
	//while (j<4)
	while (!file.atEnd())
	{
		file.readLine(buffer, sizeof(buffer));
		cout << "linereading: " << buffer << endl;
		sscanf(buffer, "%[^:]:%s", &paras_h, &value_h);
		cout << "paras_h: " << paras_h<< endl;
		cout << "value_h: " << value_h << endl;
		// Loop to find header info
		for (int i = 0; i < sizeof(data_capture_header) / sizeof(data_capture_header[0]); i++)
		{
			if (paras_h == data_capture_header[i])
			{
				switch (i)
				{
				case 0:
					x_offset = atof(value_h);
					break;
				case 1:
					y_offset = atof(value_h);
					break;
				case 2:
					z_offset = atof(value_h);
					break;
				case 3:
					h_angle_offset = atof(value_h);
					break;
				}
			}
		}
		j++;
	}
	cout << "x_offset: " << x_offset << endl;
	cout << "y_offset: " << y_offset << endl;
	cout << "z_offset: " << z_offset << endl;
	cout << "angle_offset: " << h_angle_offset << endl;
}

//read pcd from customer txt, format x y z
pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_xyz_cloud_from_txt(std::string file)
{
	std::ifstream fs;
	fs.open(file, std::ios::in);
	float x, y, z;
	int count = 0;
	std::string line;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

	// Fill in the CloudIn data

	while (getline(fs, line)) {
		if (line.empty()) {}
		else
		{
			count = count + 1;
		}
	}
	cloud->width = count;
	cloud->height = 1;
	fs.close();
	count = 0;
	fs.open(file, std::ios::in);
	while (getline(fs, line)) {
		if (line.empty()) {}
		else {
			std::stringstream ss(line);
			ss >> x >> y >> z;
			pcl::PointXYZ point;
			point.x = x / 1000;
			point.y = y / 1000;
			point.z = z / 1000;
			cloud->points.push_back(point);
			count = count + 1;
		}
	}
	return cloud;
}

//read depth data from customer txt, format 2D x-y matrix of z
//GD modified, for 640x480 image
pcl::PointCloud<pcl::PointXYZ>::Ptr load_depth_data_from_txt_640_480(std::string file)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	std::vector<int> depthbuffer;

	cloud->width = 640;
	cloud->height = 480;

	std::ifstream fs;
	fs.open(file);

	for (int y = 0; y < 480; y++)//定义行循环
	{
		for (int x = 0; x < 640; x++)//定义列循环
		{
			pcl::PointXYZ point;
			fs >> camera_depthmat_480_640[y][x];
			depthbuffer.push_back(camera_depthmat_480_640[y][x]);
			point.z = camera_depthmat_480_640[y][x] / 1000.0;
			point.x = (static_cast<float> (x) - camera_cal_cx) *  camera_depthmat_480_640[y][x] / camera_cal_fx / 1000.0;
			point.y = (static_cast<float> (y) - camera_cal_cy) *  camera_depthmat_480_640[y][x] / camera_cal_fy / 1000.0;
			cloud->points.push_back(point);
		}
	}
	std::cout << "[INFO] Loaded offline depth image has points: " << depthbuffer.size() << std::endl;
	//std::cout << depthbuffer.at(0) << std::endl;
	//std::cout << depthbuffer.back() << std::endl;

	fs.close();//读取完成之后关闭文件
	std::cout << "close file" << std::endl;
	if (global_debug_option) {
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_handler(cloud, 0, 255, 0);
		viewer.addPointCloud<pcl::PointXYZ>(cloud, cloud_handler, "offline_cloud");
		viewer.registerPointPickingCallback(pointPickingEventOccurred, (void*)&viewer);
		viewer.addCoordinateSystem(1.0);
		viewer.initCameraParameters();

		viewer.addText(file, 20, 20);
		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			pcl_sleep(0.01);
		}
	}
	return cloud;
}

//read depth data from camera_depthmat_480_640, format 2D x-y matrix of z, but float points
pcl::PointCloud<pcl::PointXYZ>::Ptr load_depth_data_from_camera_depthmat_640_480()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

	cloud->width = 640;
	cloud->height = 480;

	for (int y = 0; y < 480; y++)//定义行循环
	{
		for (int x = 0; x < 640; x++)//定义列循环
		{
			pcl::PointXYZ point;
			point.z = camera_depthmat_480_640[y][x] / 1000.0;
			point.x = (static_cast<float> (x) - camera_cal_cx) *  camera_depthmat_480_640[y][x] / camera_cal_fx / 1000.0;
			point.y = (static_cast<float> (y) - camera_cal_cy) *  camera_depthmat_480_640[y][x] / camera_cal_fy / 1000.0;
			cloud->points.push_back(point);
		}
	}

	if (global_debug_option) {
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_handler(cloud, 0, 255, 0);
		viewer.addPointCloud<pcl::PointXYZ>(cloud, cloud_handler, "offline_cloud");
		viewer.registerPointPickingCallback(pointPickingEventOccurred, (void*)&viewer);
		viewer.addCoordinateSystem(1.0);
		viewer.initCameraParameters();

		viewer.addText("cloud_readed_from_depth_map", 20, 20);
		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			pcl_sleep(0.01);
		}
	}
	return cloud;
}

//filter passthrough. for xyz format
void passthrough_xyz(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_before, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, float distance_near = 0.5, float distance_far = 2.5)
{

	//step 2 passthrough filter, crop between near and far. 
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud_before);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(distance_near, distance_far);
	pass.filter(*cloud_after);

}

//filter radius. for xyz format
//remove points that too isolate from others....... feel lonely .. is not good
void radius_removal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_before, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, float radius_bound = 0.02, int neighbor_min = 4)
{
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
	outrem.setInputCloud(cloud_before);
	outrem.setRadiusSearch(radius_bound);
	outrem.setMinNeighborsInRadius(neighbor_min);
	// apply filter
	outrem.filter(*cloud_after);
}

void
box_crop_xyz_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr rawdpthxyz, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCrop, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

	pcl::CropBox<pcl::PointXYZ> cropFilter;
	cropFilter.setInputCloud(rawdpthxyz);
	cropFilter.setMin(minPoint);
	cropFilter.setMax(maxPoint);
	cropFilter.filter(*cloudCrop);
}


//3D downsamle. leafx, leafy, leafz defines a small cubicle, all points inside will be averaged to 1 point. 
void
downsample_3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_before, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, float leafx, float leafy, float leafz)
{

	pcl::VoxelGrid<pcl::PointXYZ> voxelsor;
	voxelsor.setInputCloud(cloud_before);
	voxelsor.setLeafSize(leafx, leafy, leafz);
	voxelsor.filter(*cloud_after);
}

//
void
gauss_3D(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_before, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, int filter_Kdistance = 10, double filter_ntimes_variation = 1.5)
{
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> gauss_sor;
	gauss_sor.setInputCloud(cloud_before);
	gauss_sor.setMeanK(filter_Kdistance);
	gauss_sor.setStddevMulThresh(filter_ntimes_variation);
	//gauss_sor.setKeepOrganized(true);
	//cout << "[INFO]start gauss filtering" << " at TIMESTAMP " << now_str() << endl;
	gauss_sor.filter(*cloud_after);
}
//perpendicular plane filter model. 
void
horizontal_perpendicular_plane_finder(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain, Eigen::Vector3f axis_selection, float distance_threshold, float angle_threshold_deg, pcl::ModelCoefficients::Ptr plane_coefficient, bool optimize_coeffs = false)
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(optimize_coeffs);
	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
	//seg.setModelType(pcl::SACMODEL_PLANE);

	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(600);
	//seg.setMaxIterations(2000);
	seg.setDistanceThreshold(distance_threshold);
	seg.setAxis(axis_selection);
	seg.setEpsAngle(pcl::deg2rad(angle_threshold_deg));

	pcl::ExtractIndices<pcl::PointXYZ> extract;
	seg.setInputCloud(cloud_in);
	seg.segment(*inliers, *plane_coefficient);

	// Extract the inliers, generate plane pcd
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*cloud_after);

	//remove plane points from cloud_remain
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud_after, *cloud_f);
	extract.setNegative(true);
	extract.filter(*cloud_f);
	cloud_p.swap(cloud_f);
	pcl::copyPointCloud(*cloud_p, *cloud_remain);
}

//perpendicular plane filter model. 
void
//project_z_plane_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected, Eigen::Vector3f axis_selection, float distance_threshold, float angle_threshold_deg, pcl::ModelCoefficients::Ptr plane_coefficient, bool optimize_coeffs = false)
project_z_plane_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane_cut, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain, pcl::ModelCoefficients::Ptr plane_coefficient, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xy_plane)
{
	
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	bool optimize_coeffs = true;
	float distance_threshold = 0.06;
	Eigen::Vector3f axis_selection = Eigen::Vector3f(0, 0, 1);
	float angle_threshold_deg = 30;
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(optimize_coeffs);
	seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);

	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(600);
	//seg.setMaxIterations(2000);
	seg.setDistanceThreshold(distance_threshold);
	seg.setAxis(axis_selection);
	seg.setEpsAngle(pcl::deg2rad(angle_threshold_deg));

	pcl::ModelCoefficients::Ptr RANSAC_plane_coefficient(new pcl::ModelCoefficients());
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	seg.setInputCloud(cloud_in);
	seg.segment(*inliers, *RANSAC_plane_coefficient);

	// Extract the inliers, generate plane pcd
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*cloud_plane_cut); // point cloud data filtered for +/- 0.05 distance
		
	//remove plane points from cloud_remain
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud_plane_cut, *cloud_f);
	extract.setNegative(true);
	extract.filter(*cloud_f);
	cloud_p.swap(cloud_f);
	pcl::copyPointCloud(*cloud_p, *cloud_remain);

	//create a set of planar coefficients with X=Y=0, Z = 1
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = 0;
	coefficients->values[1] = 1;
	coefficients->values[2] = 0;
	coefficients->values[3] = 0;

	//create the filtering object for projection
	pcl::ProjectInliers<pcl::PointXYZ> proj;//创建投影滤波对象
	proj.setModelType(pcl::SACMODEL_PLANE);//设置对象对应的投影模型
	proj.setInputCloud(cloud_plane_cut);//设置输入点云 
	proj.setModelCoefficients(coefficients);//设置模型对应的系数
	proj.filter(*cloud_projected);//执行投影滤波存储结果cloud_projected

	// downsample 2D point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
	// Create the filtering object
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_projected);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*cloud_filtered);

	std::cerr << "Cloud after projection: " << std::endl;
	std::vector<cv::Point2f> xz_points;
	std::vector<cv::Point2f> xy_points;
	//pcl::PointCloud<pcl::PointXYZ> cloud_xy_plane;
	pcl::PointXYZ xy_point;
	//int M = cloud_in->points.size();
	int M = cloud_filtered->points.size();
	cout << "cloud size is:" << M << endl;
	pcl::copyPointCloud(*cloud_projected, *cloud_xy_plane);
	// read x z points and push into vector to do linefit
	for (int i = 0; i < M; i++)
	{
	//	xz_points.push_back(cv::Point2f(cloud_in->points[i].x,cloud_in->points[i].z));
		xz_points.push_back(cv::Point2f(cloud_filtered->points[i].x, cloud_filtered->points[i].z));
		xy_point.x = cloud_filtered->points[i].x;
		xy_point.y = cloud_filtered->points[i].y;
		xy_point.z = 0;
		cloud_xy_plane->points[i].z=0;
	}
	// fit the line with points in x-z plane
	cv::Vec4f line_para;
	cv::fitLine(xz_points, line_para, cv::DIST_L1, 0, 1e-2, 1e-2);
	/*
	void cv::fitLine(
		cv::InputArray points, // 二维点的数组或vector
		cv::OutputArray line, // 输出直线,Vec4f (2d)或Vec6f (3d)的vector
		int distType, // 距离类型
		double param, // 距离参数
		double reps, // 径向的精度参数
		double aeps // 角度精度参数
	);*/
	cout << "line_para = " << line_para << std::endl;
	//获取点斜式的点和斜率
	cv::Point2f point0;
	point0.x = line_para[2];
	point0.y = line_para[3];
	double k = line_para[1] / line_para[0];
	//cout <<"header: "<< plane_coefficient->header << endl;	
	plane_coefficient->values.resize(4);
	//plane_coefficient->values[0] = k;
	//plane_coefficient->values[1] = 0;
	//plane_coefficient->values[2] = -1;
	//plane_coefficient->values[3] = line_para[3]-k*line_para[2];
	plane_coefficient->values[0] = -k;
	plane_coefficient->values[1] = 0;
	plane_coefficient->values[2] = 1;
	plane_coefficient->values[3] = -line_para[3] + k * line_para[2];
	if (display_option == 1)
	{
		//计算直线的端点(y = k(x - x0) + y0)
		pcl::PointXYZ point1, point2;
		point1.x = -1;
		point1.y = 0;
		point1.z = k * (point1.x - point0.x) + point0.y;
		point2.x = 1;
		point2.y = 0;
		point2.z = k * (point2.x - point0.x) + point0.y;

		cout << "plot line fitting results...." << endl;
		// plot 3D point cloud data
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
		viewer->setBackgroundColor(0, 0, 0);
		//pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_pointsPtr);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(cloud_in, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_in, blue, "sample cloud");

		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red(cloud_projected, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_projected, red, "sample cloud2");

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud_plane_cut, 0, 255, 0);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_plane_cut, green, "sample cloud3");

		viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(point1, point2);
		viewer->addCoordinateSystem(1.0);
		viewer->initCameraParameters();
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
		//system("pause");
	}
	return;

}

//find a line that parallel with selected axis
void
parallel_with_axis_line_finder(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain, Eigen::Vector3f axis_selection, float distance_threshold, float angle_threshold_deg, pcl::ModelCoefficients::Ptr line_coefficient)
{
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);

	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(1000);
	seg.setDistanceThreshold(distance_threshold);
	seg.setAxis(axis_selection);
	seg.setEpsAngle(pcl::deg2rad(angle_threshold_deg));

	pcl::ExtractIndices<pcl::PointXYZ> extract;
	seg.setInputCloud(cloud_in);
	seg.segment(*inliers, *line_coefficient);

	// Extract the inliers, generate plane pcd
	extract.setInputCloud(cloud_in);
	extract.setIndices(inliers);
	extract.setNegative(false);
	extract.filter(*cloud_after);

	//remove plane points from cloud_remain
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloud_after, *cloud_f);
	extract.setNegative(true);
	extract.filter(*cloud_f);
	cloud_p.swap(cloud_f);
	pcl::copyPointCloud(*cloud_p, *cloud_remain);
}
//zshift a nearby point to a plane, keeps xy not change but move z to given plane
void zshift_points_to_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, pcl::ModelCoefficients::Ptr plane_coefficient, double distance_threshold)
{
	double distance;
	float a, b, c, d;
	a = plane_coefficient->values[0];
	b = plane_coefficient->values[1];
	c = plane_coefficient->values[2];
	d = plane_coefficient->values[3];
	float x, y, z;

	for (size_t j = 0; j < cloud_in->points.size(); ++j)
	{
		// Cloud projection

		x = cloud_in->at(j).x;
		y = cloud_in->at(j).y;
		z = cloud_in->at(j).z;
		pcl::PointXYZ point;

		distance = pcl::pointToPlaneDistance(cloud_in->points[j], Eigen::Vector4f(a, b, c, d));
		if (distance < distance_threshold)
		{
			float newz = -(a*x + b * y + d) / c;
			point.x = x;
			point.y = y;
			point.z = newz;
			cloud_after->push_back(point);
		}
	}
}
//project nearby points to a plane
void project_points_to_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, pcl::ModelCoefficients::Ptr plane_coefficient, double distance_threshold)
{
	double distance;
	float a, b, c, d;
	a = plane_coefficient->values[0];
	b = plane_coefficient->values[1];
	c = plane_coefficient->values[2];
	d = plane_coefficient->values[3];
	float x, y, z;

	for (size_t j = 0; j < cloud_in->points.size(); ++j)
	{
		// Cloud projection
		x = cloud_in->at(j).x;
		y = cloud_in->at(j).y;
		z = cloud_in->at(j).z;
		pcl::PointXYZ point;
		distance = pcl::pointToPlaneDistance(cloud_in->points[j], Eigen::Vector4f(a, b, c, d));
		if (distance < distance_threshold)
		{
			pcl::projectPoint(cloud_in->at(j), Eigen::Vector4f(a, b, c, d), point);
			cloud_after->push_back(point);
		}
	}
}
//transform a cloud to given plane by rotate between src_cloud_normal and target_cloud_normal
//returns the transformation list for src to go to target place
//newcentroid vector: 0, 0,-1.5 will pull point cloud close to xy plane by 1.5m
//rotate vector: to 0, 0, 1 will rotate a plane to xy plane. 
std::vector<Eigen::Affine3f>
transform_plane_to_given_plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after, Eigen::Vector4f newcentroid, Eigen::Matrix<float, 1, 3> src_plane_selection, Eigen::Matrix<float, 1, 3> target_plane_selection)
{
	Eigen::Matrix<float, 1, 3> rotation_vector;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_rotate(new pcl::PointCloud<pcl::PointXYZ>);
	rotation_vector = target_plane_selection.cross(src_plane_selection);
	float theta = -atan2(rotation_vector.norm(), target_plane_selection.dot(src_plane_selection));

	std::vector<Eigen::Affine3f> transform_lists;
	Eigen::Affine3f transform_1 = Eigen::Affine3f::Identity();
	Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
	Eigen::Vector4f centroid;
	transform_1.rotate(Eigen::AngleAxisf(theta, rotation_vector.normalized()));
	std::cout << transform_1.matrix() << std::endl << std::endl;
	transform_lists.push_back(transform_1);
	pcl::transformPointCloud(*cloud_in, *cloud_rotate, transform_1);
	pcl::compute3DCentroid(*cloud_rotate, centroid);
	transform_2.translation() << newcentroid[0] - centroid[0], newcentroid[1] - centroid[1], newcentroid[2] - centroid[2];
	std::cout << transform_2.matrix() << std::endl << std::endl;
	pcl::transformPointCloud(*cloud_rotate, *cloud_after, transform_2);
	transform_lists.push_back(transform_2);

	return transform_lists;
}

///////////////////////////////opencv 2d processing functions///////////////////////////////////////////////////
Mat downsample_2D(Mat imgin, int down_converge_ratio)
{
	// --------------------------------------------
	// -----Opencv function down convert-----
	// --------------------------------------------
	Mat imgout;
	Mat tmp = imgin;
	cv::resize(tmp, imgout, Size(tmp.cols / down_converge_ratio, tmp.rows / down_converge_ratio));
	return imgout;
}


void preprocess_tray_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out, float downsamplesize, bool gaussT = true)
{
	cout << "[INFO]start preprocess pcd" << " at TIMESTAMP " << now_str() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledxyz(new pcl::PointCloud<pcl::PointXYZ>);
	downsample_3D(cloud_in, downsampledxyz, downsamplesize, downsamplesize, downsamplesize);
	cout << "[INFO]downsample 3D" << " at TIMESTAMP " << now_str() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr truncate_xyz(new pcl::PointCloud<pcl::PointXYZ>);
	if (gaussT) {
		passthrough_xyz(downsampledxyz, truncate_xyz, 0.5, 4);
		cout << "[INFO]passthrough xyz" << " at TIMESTAMP " << now_str() << endl;
		gauss_3D(truncate_xyz, cloud_out, 10, 1.5);
		cout << "[INFO]gauss 3D truncate " << " at TIMESTAMP " << now_str() << endl;
	}
	else {
		passthrough_xyz(downsampledxyz, cloud_out, 0.5, 4);
		cout << "[INFO]passthrough xyz" << " at TIMESTAMP " << now_str() << endl;
	}
	cout << "preprocess pcd completed. has " << cloud_out->size() << " valid points" << " at TIMESTAMP " << now_str() << endl;

}


void Preprocess_pcd_Thread::run()
{
	QTimer pre_pcd_timer;
	bool res = connect(&pre_pcd_timer, SIGNAL(timeout()), this, SLOT(preprocess_pcd()), Qt::DirectConnection);
	int time_inter = 50;  // set timer to 1 second
	pre_pcd_timer.setTimerType(Qt::PreciseTimer);
	pre_pcd_timer.start(time_inter);
	exec();
	return;
};

//this will re-organize pcd into range image type cloud, organized and ready to project to 2D
void generate_range_image_from_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::RangeImagePlanar::Ptr range_image_ptr, int scale_factor = 1)
{
	// use the cloud to generate range image type cloud.
	int image_size_x = 640 / scale_factor, image_size_y = 480 / scale_factor;

	float center_x = camera_cal_cx, center_y = camera_cal_cy;
	float focal_length_x = camera_cal_fx, focal_length_y = camera_cal_fy;
	Eigen::Affine3f sensor_pose = Eigen::Affine3f(Eigen::Translation3f(cloud_in->sensor_origin_[0],
		cloud_in->sensor_origin_[1],
		cloud_in->sensor_origin_[2])) *
		Eigen::Affine3f(cloud_in->sensor_orientation_);
	float noise_level = 0.02f, minimum_range = 0.0f;
	//pcl::RangeImagePlanar range_image;
	pcl::RangeImagePlanar& range_image = *range_image_ptr;
	range_image.createFromPointCloudWithFixedSize(*cloud_in, image_size_x, image_size_y,
		center_x, center_y, focal_length_x, focal_length_x,
		sensor_pose, pcl::RangeImage::CAMERA_FRAME,
		noise_level, minimum_range);
}


void find_boundaries_and_NARF_points_use_range_image_method(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points, pcl::RangeImagePlanar::Ptr range_image_out, pcl::PointCloud<pcl::PointXYZ>::Ptr NARF_points, bool NARF_CAL = false, float NART_radius = 0.1)
{
	pcl::RangeImagePlanar::Ptr range_image_ptr(new pcl::RangeImagePlanar);
	generate_range_image_from_pcd(cloud_in, range_image_ptr);
	cout << "generate range image from pcd" << " at TIMESTAMP " << now_str() << endl;
	pcl::RangeImagePlanar& range_image = *range_image_ptr;
	//Eigen::Vector3f threeDPoint;
	//range_image_ptr->getPoint(p_x, p_y, threeDPoint);
	range_image.setUnseenToMaxRange();
	cout << "setUnseenToMaxRange" << " at TIMESTAMP " << now_str() << endl;
	pcl::RangeImageBorderExtractor border_extractor(&range_image);
	pcl::PointCloud<pcl::BorderDescription> border_descriptions;
	border_extractor.compute(border_descriptions);
	// assign a CPU 02
	int dwMask = 0002;
	SetThreadAffinityMask(GetCurrentThread(), dwMask);
	cout << "New pre-pcd assign to CPU_02" << endl;

	cout << "border extractor compute" << " at TIMESTAMP " << now_str() << endl;
	pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
		veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
		shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
	pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr,
		&veil_points = *veil_points_ptr,
		&shadow_points = *shadow_points_ptr;
	int x, y;
	for (y = 0; y < range_image.height; ++y)
	{
		for (x = 0; x < range_image.width; ++x)
		{
			if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER])
				border_points.points.push_back(range_image.points[y*range_image.width + x]);
			//if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT])
			//	veil_points.points.push_back(range_image.points[y*range_image.width + x]);
			//if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__SHADOW_BORDER])
			//	shadow_points.points.push_back(range_image.points[y*range_image.width + x]);
		}
	}
	cout << "border points veil points and shadow points " << " at TIMESTAMP " << now_str() << endl;
	if (NARF_CAL) {
		pcl::NarfKeypoint narf_keypoint_detector;
		narf_keypoint_detector.setRangeImageBorderExtractor(&border_extractor);
		narf_keypoint_detector.setRangeImage(&range_image);
		narf_keypoint_detector.getParameters().support_size = NART_radius;
		pcl::PointCloud<int> keypoint_indices;
		narf_keypoint_detector.compute(keypoint_indices);
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>& keypoints = *keypoints_ptr;
		keypoints.points.resize(keypoint_indices.points.size());
		for (size_t i = 0; i < keypoint_indices.points.size(); ++i)
		{
			keypoints.points[i].getVector3fMap() = range_image.points[keypoint_indices.points[i]].getVector3fMap();
		}
		pcl::copyPointCloud(*keypoints_ptr, *NARF_points);
	}
	pcl::copyPointCloud(*border_points_ptr, *boundary_points);
	pcl::copyPointCloud(*range_image_ptr, *range_image_out);
	cout << "copy PC to boundary and range image" << " at TIMESTAMP " << now_str() << endl;
}

//if multi-planes extracted from selected 2D area. we will have to find a most possible plane
//consider box maybe 1 surface, tray maybe a surface, deck maybe another one tray surface is more likely be the middle one
//method is first find centroid for raw clouds which has more than 1 vertical planes. 
//then compare each possible plane's cloud centroid, the one more close to center is more likely be the plane we interested
//the allPlanesCenterPoint is also computed and returned
pcl::ModelCoefficients::Ptr
get_most_possible_plane(std::vector<pcl::ModelCoefficients::Ptr> plane_coeff_vec, pcl::PointCloud<pcl::PointXYZ>::Ptr boundaries_points, Eigen::Vector4f& Allplanescentroid)
{

	struct merit_factors
	{
		int point_numbers;
		float centroid_x;
		float centroid_y;
		float centroid_z;
	};
	std::vector<merit_factors> planemerits_vect;
	std::vector<double> merit_score_vec;
	double max_points = 0;
	double max_deviation_from_center = 0;

	pcl::compute3DCentroid(*boundaries_points, Allplanescentroid);
	int count = plane_coeff_vec.size();
	for (int t = 0; t < count; t++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr planepoints(new pcl::PointCloud<pcl::PointXYZ>);
		project_points_to_plane(boundaries_points, planepoints, plane_coeff_vec[t], 0.05);
		Eigen::Vector4f plane_centroid;
		pcl::compute3DCentroid(*planepoints, plane_centroid);
		merit_factors planeMerit;
		planeMerit.point_numbers = planepoints->size();
		planeMerit.centroid_x = plane_centroid[0];
		planeMerit.centroid_y = plane_centroid[1];
		planeMerit.centroid_z = plane_centroid[2];
		planemerits_vect.push_back(planeMerit);
		if (planeMerit.point_numbers > max_points)
		{
			max_points = planeMerit.point_numbers;
		}
		double center_deviation = pow((planeMerit.centroid_x - Allplanescentroid[0]), 2) + pow((planeMerit.centroid_y - Allplanescentroid[1]), 2);
		if (center_deviation > max_deviation_from_center)
		{
			max_deviation_from_center = center_deviation;
		}
	}
	for (int t = 0; t < count; t++)
	{
		merit_factors planeMerit;
		planeMerit = planemerits_vect[t];
		double center_deviation = pow((planeMerit.centroid_x - Allplanescentroid[0]), 2) + pow((planeMerit.centroid_y - Allplanescentroid[1]), 2);
		double score_lvl = planeMerit.point_numbers / max_points * max_deviation_from_center / center_deviation;
		merit_score_vec.push_back(score_lvl);
	}
	int maxMeridIndex = std::max_element(merit_score_vec.begin(), merit_score_vec.end()) - merit_score_vec.begin();
	cout << "maxMeridIndex"<< maxMeridIndex <<endl;
	cout << "plane_coeff" << plane_coeff_vec[maxMeridIndex]->values[0] << endl;
	return plane_coeff_vec[maxMeridIndex];
}

//this is the final calculation part. 
//we will sort all the x lines and y lines and select the ones most likely be our final target. 
//ideally, this should have selection part/ calculation part/ verification part. 
//returned is centor point of tray
//detected lines are like below. 1 xline, 2 yline, center is tray center.
//------------		-------------
//			 |		|
//			 |		|

//filter line length to be at least longer than certain ratio
std::vector<lineinfo> line_length_qualifier(std::vector<lineinfo> input_lines_segmented, float line_min_length) {
	std::vector<lineinfo> len_qualified_lines;
	int linecount = input_lines_segmented.size();
	//self check first
	for (int t = 0; t < linecount; t++)
	{
		lineinfo thisLine = input_lines_segmented[t];
		float thisLine_len = 0;
		int seg_count = thisLine.segments.size();
		for (int k = 0; k < seg_count; k++)
		{
			thisLine_len = thisLine_len + thisLine.segments[k].segment_length;
		}
		if (thisLine_len > line_min_length) {
			len_qualified_lines.push_back(thisLine);
		}
	}
	return len_qualified_lines;
}

//general tracking through 3D will try find x line, y line pairs closely encircling rough center.
//use the calculated incersections as tracking points. 
int general_tracking_point_cal_3D(std::vector <lineinfo> x_lines_segmented, std::vector <lineinfo> y_lines_segmented, pcl::PointXYZ roughcenter, pcl::PointXYZ& calcpointA, pcl::PointXYZ& calcpointB)
{
	cout << "[Debug] Running 3D point tracking algorithm...." << endl;
	float xlinecenterpadding = 0.0; // this is to give some margin on x line center compared with rough center		
	float yline_to_left_right_most_point_dis = 0.2; //we want y line to be close to center. this threshold compares yline middle point to 
	// x line left most and right most point and requires this distance to be at least larger than this threshold

	Eigen::Vector4f trackingA, trackingB;

	//goal is selecting a x line and 2 y line, whose intersections are tracking A and tracking B
	//to filter out noise lines, the x line and y line total length has to be bigger than a certain threshold
	float x_line_min_length = 0.15;  //0.5
	float y_line_min_length = 0.01;  //0.01; 0.05
	std::vector<lineinfo> len_qualified_x_lines, len_qualified_y_lines;
	len_qualified_x_lines = line_length_qualifier(x_lines_segmented, x_line_min_length);
	len_qualified_y_lines = line_length_qualifier(y_lines_segmented, y_line_min_length);

	for (int l = 0; l < x_lines_segmented.size(); l++)
	{
		cout << "x line middle point:"<< x_lines_segmented[l].segments[0].middle_point.y <<endl;
	}
	for (int l = 0; l < len_qualified_x_lines.size(); l++)
	{
		cout << "qualified x line middle point:" << len_qualified_x_lines[l].segments[0].middle_point.y << endl;
	}

	for (int l = 0; l < y_lines_segmented.size(); l++)
	{
		cout << "y line middle point:" << y_lines_segmented[l].segments[0].middle_point.x << endl;
	}
	for (int l = 0; l < len_qualified_y_lines.size(); l++)
	{
		cout << "qualified y line middle point:" << len_qualified_y_lines[l].segments[0].middle_point.x << endl;
	}

	if (len_qualified_y_lines.size() < 2) {
		cout << "No enough y lines detected......" << endl;
		return -1;
	}

	cout << "rough center x: " << roughcenter.x << endl;
	cout << "rough center y: " << roughcenter.y << endl;
	cout << "rough center z: " << roughcenter.z << endl;

	std::sort(len_qualified_x_lines.begin(), len_qualified_x_lines.end(), linesort_low_to_high); //sort by height low to high. for ex. y_0 =0, y_1 =-0.2, y_2 = -0.4..

	lineinfo potential_xline, potential_yline_left, potential_yline_right;
	double potential_yline_center=1000;
	//std::vector<float> potential_yline_left_list, potential_yline_right_list;
	//strategy to select potential_xline is it needs to be has at least to sections. has to be higher than (<) detected zone rough center(geometric center)
	//strategy to select potential yline_left and yline_right. is the two lines most close to rough center
	//final data need to pass a criteria in x direction: xline most left point << yline_left and yline_right << xline most right point. 

	int xline_count = len_qualified_x_lines.size();
	bool potential_x_found = false;
	//self check first
	cout << "# of x lines found: " << xline_count << endl;
	for (int t = 0; t < xline_count; t++)
	{
		cout << "rank of x line:" << t << endl;
		cout << "y location: " << len_qualified_x_lines[t].segments[0].middle_point.y << endl;
		//find first line that higher than roughcenter
		if (len_qualified_x_lines[t].segments[0].middle_point.y < (roughcenter.y + xlinecenterpadding)) {
			potential_xline = len_qualified_x_lines[t];
			potential_x_found = true;
			cout << "found the rank of x line:" << t << endl;
			break;
		}
	}
	if (potential_x_found) {
		cout << "xline segments size:" << potential_xline.segments.size() << endl;
		if (potential_xline.segments.size() < 2) {
			// somehow this x line only has 1 segment, we need to have at least 2. terminate cal
			return -1;
		}
	}
	else { return -1; }

	// find x line most left and most right point
	std::vector<float> xline_seg_points;
	int xline_segments_count = potential_xline.segments.size();
	for (int t = 0; t < xline_segments_count; t++) {
		xline_seg_points.push_back(potential_xline.segments[t].start_point.x);
		xline_seg_points.push_back(potential_xline.segments[t].ending_point.x);
	}
	std::sort(xline_seg_points.begin(), xline_seg_points.end()); //sort elements from left to right
	float min_x = xline_seg_points.front();
	float max_x = xline_seg_points.back();
	// find potential y pairs. 
	std::vector<lineinfo_relative_to_point> yline_relative_to_center_list;
	std::vector<lineinfo_relative_to_point> potential_yline_left_list, potential_yline_right_list;
	std::vector<float> potential_yline_center_list;
	int yline_count = len_qualified_y_lines.size();
	for (int t = 0; t < yline_count; t++)
	{
		//if ((abs(len_qualified_y_lines[t].segments[0].middle_point.x-roughcenter.x) > 0.1*yline_to_left_right_most_point_dis) && (abs(len_qualified_y_lines[t].segments[0].middle_point.x - roughcenter.x) < 1.2*yline_to_left_right_most_point_dis))
		if (abs(len_qualified_y_lines[t].segments[0].middle_point.x - roughcenter.x) > 0)
		{
			lineinfo_relative_to_point temp_line_relative_to_center;
			temp_line_relative_to_center.thisline = len_qualified_y_lines[t];
			temp_line_relative_to_center.ref_point = roughcenter;
			yline_relative_to_center_list.push_back(temp_line_relative_to_center);
		}
		else
		{
			cout << "y line to close to rough center....." << endl;
		}
	}
	std::sort(yline_relative_to_center_list.begin(), yline_relative_to_center_list.end(), linesort_by_dis_to_point_in_x); //sort lines by their distance to rough center x. basically the two closest y line are considered good fit. 
	bool found_yline_pair_flag = false;
	for (int i = 0; i < yline_relative_to_center_list.size(); i++)
	{
		for (int j = 0; j < yline_relative_to_center_list.size(); j++)
		{
			//if ((yline_relative_to_center_list[i].thisline.segments[0].middle_point.x - roughcenter.x) *(yline_relative_to_center_list[j].thisline.segments[0].middle_point.x - roughcenter.x) < 0 && 1.5*yline_to_left_right_most_point_dis>abs(yline_relative_to_center_list[i].thisline.segments[0].middle_point.x- yline_relative_to_center_list[j].thisline.segments[0].middle_point.x)>0.6*yline_to_left_right_most_point_dis)
			if (1.5*yline_to_left_right_most_point_dis > abs(yline_relative_to_center_list[i].thisline.segments[0].middle_point.x - yline_relative_to_center_list[j].thisline.segments[0].middle_point.x) && abs(yline_relative_to_center_list[i].thisline.segments[0].middle_point.x - yline_relative_to_center_list[j].thisline.segments[0].middle_point.x) > 0.6*yline_to_left_right_most_point_dis)
			{
				if (yline_relative_to_center_list[i].thisline.segments[0].middle_point.x < yline_relative_to_center_list[j].thisline.segments[0].middle_point.x)
				{
					potential_yline_left_list.push_back(yline_relative_to_center_list[i]);
					potential_yline_right_list.push_back(yline_relative_to_center_list[j]);
					potential_yline_center_list.push_back(abs((yline_relative_to_center_list[i].thisline.segments[0].middle_point.x + yline_relative_to_center_list[j].thisline.segments[0].middle_point.x)/2 - roughcenter.x));
					cout << "found pair of left y lines at i: " << i << " :" << yline_relative_to_center_list[i].thisline.segments[0].middle_point.x << endl;
					cout << "found pair of right y lines at j: " << j << " :" << yline_relative_to_center_list[j].thisline.segments[0].middle_point.x << endl;
					cout << "found pair to center distance: " << potential_yline_center_list[potential_yline_center_list.size()-1] << endl;
					if (potential_yline_center > potential_yline_center_list[potential_yline_center_list.size()-1] && potential_yline_center_list[potential_yline_center_list.size()-1] < 0.6*yline_to_left_right_most_point_dis)
					{
						potential_yline_left = yline_relative_to_center_list[i].thisline;
						potential_yline_right = yline_relative_to_center_list[j].thisline;
						potential_yline_center = potential_yline_center_list[potential_yline_center_list.size()-1];
						cout << "y line center at: " << potential_yline_center_list[potential_yline_center_list.size()-1] << endl;
						found_yline_pair_flag = true;
					}
					else 
					{	}
				}
				else {
					potential_yline_left_list.push_back(yline_relative_to_center_list[j]);
					potential_yline_right_list.push_back(yline_relative_to_center_list[i]);
					potential_yline_center_list.push_back(abs((yline_relative_to_center_list[i].thisline.segments[0].middle_point.x + yline_relative_to_center_list[j].thisline.segments[0].middle_point.x)/2 - roughcenter.x));
					cout << "found pair of left y lines at j: " << j << " :" << yline_relative_to_center_list[j].thisline.segments[0].middle_point.x << endl;
					cout << "found pair of right y lines at i: " << i << " :" << yline_relative_to_center_list[i].thisline.segments[0].middle_point.x << endl;
					cout << "found pair to center distance: " << potential_yline_center_list[potential_yline_center_list.size()-1] << endl;
					if (potential_yline_center > potential_yline_center_list[potential_yline_center_list.size()-1] && potential_yline_center_list[potential_yline_center_list.size()-1] < 0.6*yline_to_left_right_most_point_dis)
					{
						potential_yline_left = yline_relative_to_center_list[j].thisline;
						potential_yline_right = yline_relative_to_center_list[i].thisline;
						potential_yline_center = potential_yline_center_list[potential_yline_center_list.size()-1];
						cout << "y line center at: " << potential_yline_center_list[potential_yline_center_list.size()-1] << endl;
						found_yline_pair_flag = true;
					}
					else
					{	}
				}
				break;
			}
			else
			{
				cout << "y line pair are at the same side and go to the next y line candidate......." << endl;
				
			}
		}
	}
	
	if (found_yline_pair_flag)
	{
		cout << "Found potential y line pairs......" << endl;
	}
	else {
		return -1; }
	
	cout << "# of potential y line pairs: " << potential_yline_left_list.size() << endl;
	cout << "potential_yline_left: " << potential_yline_left.segments.front().middle_point.x << endl;
	cout << "potential_yline_right: " << potential_yline_right.segments.front().middle_point.x << endl;
	cout << "potential_yline_center_distance: " << potential_yline_center << endl;
	// original code is reserved here
	//if (yline_relative_to_center_list[0].thisline.segments[0].middle_point.x < yline_relative_to_center_list[i].thisline.segments[0].middle_point.x)
	//{
	//	potential_yline_left = yline_relative_to_center_list[0].thisline;
	//	potential_yline_right = yline_relative_to_center_list[1].thisline;
	//}
	//else {
	//	potential_yline_right = yline_relative_to_center_list[0].thisline;
	//	potential_yline_left = yline_relative_to_center_list[1].thisline;
	//}
	if (potential_yline_left.segments.size() > 0 && potential_yline_right.segments.size() > 0)
	{
		cout << "min x: " << min_x << endl;
		cout << "mid y line left: " << potential_yline_left.segments.front().middle_point.x << endl;
		cout << "max x: " << max_x << endl;
		cout << "mid y line right: " << potential_yline_right.segments.front().middle_point.x << endl;
		//check min_x has to be less than y_left, y_right has to be less than x_max
		//if (min_x + yline_to_left_right_most_point_dis < potential_yline_left.segments.front().middle_point.x && potential_yline_right.segments.front().middle_point.x + yline_to_left_right_most_point_dis < max_x) {
		if (min_x + yline_to_left_right_most_point_dis < potential_yline_left.segments.front().middle_point.x || potential_yline_right.segments.front().middle_point.x + yline_to_left_right_most_point_dis < max_x) {

			//when program runs to here. we have x line and 2 ylines. calculate track point and return
						//calculation part
			pcl::lineWithLineIntersection(potential_xline.line_coefficients, potential_yline_left.line_coefficients, trackingA);
			pcl::lineWithLineIntersection(potential_xline.line_coefficients, potential_yline_right.line_coefficients, trackingB);
			cout << "Found proper x lines...." << endl;
			// check if points 1cm below A and B are hollow
			if (true)
			{
				calcpointA.x = trackingA[0];
				calcpointA.y = trackingA[1];
				calcpointA.z = trackingA[2];

				calcpointB.x = trackingB[0];
				calcpointB.y = trackingB[1];
				calcpointB.z = trackingB[2];
			}
			return 0;
		}
		else {
			return -1;
		}
	}
	else 
	{
		cout << "No enough y line detected and detection failed....." << endl;
		return -1;
	}
}

//this is a function to calculate goods on the tray and left/right obstacles
std::vector<ThingNotTray>
find_things_ontop_and_around_tray(pcl::PointCloud<pcl::PointXYZ>::Ptr full_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr calpoints, pcl::ModelCoefficients::Ptr plane_coeffs) {
	std::vector<ThingNotTray> ThingsAroundTray;
	//crop point cloud, so y direction it only has upper tray and up. z direction only has tray facing xy direction surface and plus minus a threshold global_tray_rack_z_offset
	//x direction consider tray center plus minus global_tray_rack_x_offset. 
	pcl::PointXYZ TrackingPointA, TrackingPointB, centerAB;
	Eigen::Vector4f mincrop, maxcrop;
	TrackingPointA = calpoints->points[0];
	TrackingPointB = calpoints->points[1];
	centerAB.x = (TrackingPointA.x + TrackingPointB.x) / 2;
	centerAB.y = max(TrackingPointA.y, TrackingPointB.y); //ideally trackingpintA and B are at the same horizon. but consider some angle deviation from camera sight, we take the lower one. note y axis is reverted. so max used
	centerAB.z = (TrackingPointA.z + TrackingPointB.z) / 2;
	mincrop[0] = centerAB.x - global_tray_rack_x_offset;
	maxcrop[0] = centerAB.x + global_tray_rack_x_offset;
	mincrop[1] = centerAB.y - global_tray_rack_y_offset; //again y axis is reverted. like 2D image highest row is 0. 
	maxcrop[1] = centerAB.y - 0.05;//we don't want tray anymore... 
	mincrop[2] = centerAB.z - global_tray_rack_z_offset;
	maxcrop[2] = centerAB.z + global_tray_rack_z_offset;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cropped(new pcl::PointCloud<pcl::PointXYZ>);
	box_crop_xyz_pcd(full_pcd, cropped, mincrop, maxcrop);
	//downsample cloud to improve speed
	pcl::PointCloud<pcl::PointXYZ>::Ptr clouddown(new pcl::PointCloud<pcl::PointXYZ>);
	downsample_3D(cropped, clouddown, 0.01, 0.01, 0.01);
	//segment points into isolated objects
	// Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointIndices::Ptr lineinliers(new pcl::PointIndices());
	tree->setInputCloud(clouddown);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.02); // if discontinuity is more than this value, considered a new segment starts
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(50000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(clouddown);
	ec.extract(cluster_indices);

	int j = 0;

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_vector;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);

		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			lineinliers->indices.push_back(*pit);
			cloud_cluster->points.push_back(clouddown->points[*pit]);
		}
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		cloud_vector.push_back(std::move(cloud_cluster));
	}


	//std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_hull_vector;
	int size_segmented_clouds = cloud_vector.size();
	for (int it = 0; it < size_segmented_clouds; ++it) {

		//we don't really need to calcuated contour of objects. just give characteristic points
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::ConcaveHull<pcl::PointXYZ> chull;//this is a greedy boundary search need to partion points first. 
		//chull.setAlpha(1.0); //default 0.1 but captures too much detail. 
		//chull.setInputCloud(cloud_vector[it]);
		//chull.reconstruct(*cloud_hull);

		ThingNotTray currentThing;
		Eigen::Vector4f Thing_centroid;

		std::sort(cloud_vector[it]->points.begin(), cloud_vector[it]->points.end(), sortbyx_points);
		currentThing.left_most_point = cloud_vector[it]->points.front();
		currentThing.right_most_point = cloud_vector[it]->points.back();
		std::sort(cloud_vector[it]->points.begin(), cloud_vector[it]->points.end(), sortbyz_points);
		currentThing.front_most_point = cloud_vector[it]->points.front();
		pcl::compute3DCentroid(*cloud_vector[it], Thing_centroid);
		currentThing.center.x = Thing_centroid[0];
		currentThing.center.y = Thing_centroid[1];
		currentThing.center.z = Thing_centroid[2];

		if (currentThing.center.x<centerAB.x + 0.5 && currentThing.center.x > centerAB.x - 0.5) {
			currentThing.goods = true;
			pcl::PointCloud<pcl::PointXYZ>::Ptr surface_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::ModelCoefficients::Ptr plane_coefficient(new pcl::ModelCoefficients());
			horizontal_perpendicular_plane_finder(cloud_vector[it], surface_cloud, cloud_remain, Eigen::Vector3f(0.0, 0.0, 1.0), 0.05, 10, plane_coefficient, true); //allow 2cm error..	
			currentThing.camera_facing_eigen = *plane_coefficient;
		}

		ThingsAroundTray.push_back(currentThing);

		if (global_debug_option) {
			pcl::visualization::PCLVisualizer viewer("3D Viewer");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_handler(cropped, 0, 255, 0);
			viewer.addPointCloud<pcl::PointXYZ>(cropped, cloud_handler, "plane_cloud");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_handler_hull(cloud_vector[it], 255, 255, 255);
			viewer.addPointCloud<pcl::PointXYZ>(cloud_vector[it], cloud_handler_hull, "singulated_object");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "singulated_object");

			viewer.addText("singulated objects from cloud", 20, 20);
			while (!viewer.wasStopped())
			{
				viewer.spinOnce();
				pcl_sleep(0.01);
			}
		}
		//cloud_hull_vector.push_back(std::move(cloud_hull));
	}
	// list all the objects around tray from left to right direction
	std::sort(ThingsAroundTray.begin(), ThingsAroundTray.end(), sortbyx_things);

	return ThingsAroundTray;
}

//to refine found cloud surface and tracking points. reason is the initial findings are based on a very rough surface assumption. 
//since we've found the tracking points and a rough surface, we can use a much tighter surface fitting parameter to re-optimize our calculation results
void
refine_cloud_surface(pcl::PointCloud<pcl::PointXYZ>::Ptr partial_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cal_points, pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cal_points, pcl::ModelCoefficients::Ptr refined_plane_coefficient) {
	float y_tray_center = raw_cal_points->points[2].y;
	float x_tray_center = raw_cal_points->points[2].x;
	float yoffset = 0.07; //we know the tray is about >10cm height. from center count 4cm up and down to retract surface cloud
	float xoffset = 0.6;

	pcl::PointCloud<pcl::PointXYZ>::Ptr tray_center_surface(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr surface_plane(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain(new pcl::PointCloud<pcl::PointXYZ>);
	// build the condition
	pcl::ConditionAnd<pcl::PointXYZ>::Ptr height_condition(new pcl::ConditionAnd<pcl::PointXYZ>());
	height_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, y_tray_center - yoffset)));
	height_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, y_tray_center + yoffset)));

	height_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, x_tray_center - xoffset)));
	height_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, x_tray_center + xoffset)));
	// build the filter
	pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
	condrem.setCondition(height_condition);
	condrem.setInputCloud(partial_cloud);
	condrem.setKeepOrganized(false);
	// apply filter
	condrem.filter(*tray_center_surface);

	//now we should feel confident about our tray surface. (though it's only partial)
	//pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledxyz(new pcl::PointCloud<pcl::PointXYZ>);
	//float downsample_size_z = 0.02;
	//float downsample_size_xy = 0.05;
	//downsample_3D(tray_center_surface, downsampledxyz, downsample_size_xy, downsample_size_xy, downsample_size_z);
		
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out_temp(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in_temp(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xy_plane(new pcl::PointCloud<pcl::PointXYZ>);;

	pcl::copyPointCloud(*tray_center_surface, *cloud_in_temp);
	for (int k = 0; k < 0; ++k) {
		cout << "k size: " << k << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain_temp(new pcl::PointCloud<pcl::PointXYZ>);
		//horizontal_perpendicular_plane_finder(cloud_in_temp, cloud_out_temp, cloud_remain_temp, Eigen::Vector3f(1.0, 0.0, 0.0), 0.01, 30, refined_plane_coefficient, false);
		project_z_plane_line(cloud_in_temp, cloud_projected, surface_plane, cloud_remain, refined_plane_coefficient, cloud_xy_plane);
		cout << "Run refine function on temp PC data.... k:" <<k<< endl;
		if (cloud_out_temp->points.size() < 25)
		{
			cout << "Size of cloud is too small as:"<< cloud_out_temp->points.size() <<endl;
			break;
		}
		pcl::copyPointCloud(*cloud_remain_temp, *cloud_in_temp);
	}


	//horizontal_perpendicular_plane_finder(cloud_in_temp, surface_plane, cloud_remain, Eigen::Vector3f(0.0, 0.0, 1.0), 0.03, 10, refined_plane_coefficient, false);
	
	project_z_plane_line(cloud_in_temp, cloud_projected, surface_plane, cloud_remain, refined_plane_coefficient, cloud_xy_plane);
	cout << "Run refine function on PC data.... " << endl;
	//doubleCloudViewer(cloud_in_temp, "downed", surface_plane, "plane");
	pcl::PointXYZ refinedtrackingA, refinedtrackingB, refinedtrackingC;
	Eigen::Vector4f project_plane_eigen;
	project_plane_eigen[0] = refined_plane_coefficient->values[0];
	project_plane_eigen[1] = refined_plane_coefficient->values[1];
	project_plane_eigen[2] = refined_plane_coefficient->values[2];
	project_plane_eigen[3] = refined_plane_coefficient->values[3];

	pcl::projectPoint(raw_cal_points->at(0), project_plane_eigen, refinedtrackingA);
	pcl::projectPoint(raw_cal_points->at(1), project_plane_eigen, refinedtrackingB);
	pcl::projectPoint(raw_cal_points->at(2), project_plane_eigen, refinedtrackingC);

	refined_cal_points->points[0] = refinedtrackingA;
	refined_cal_points->points[1] = refinedtrackingB;
	refined_cal_points->points[2] = refinedtrackingC;
}

//this is detailed tray related info calculation
int pose_calculator_geometric_relationship(pcl::PointCloud<pcl::PointXYZ>::Ptr potential_tray_pcd, pcl::PointCloud<pcl::PointXYZ>::Ptr full_pcd, float plane_finding_distance_thres)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remain(new pcl::PointCloud<pcl::PointXYZ>);
	float min_x, min_y, max_x, max_y;

	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> out_vec;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xy_plane(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<pcl::ModelCoefficients::Ptr> plane_coeff_vec;
	pcl::copyPointCloud(*potential_tray_pcd, *cloud_in);
	//pcl::io::savePCDFileASCII("../debug_cloud_points.pcd", *cloud_in);
	int counts = 0;
	int surface_point_counts = 0;
	int remaining_point_counts = 0;
	do {
		pcl::PointCloud<pcl::PointXYZ>::Ptr surface_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);

		pcl::ModelCoefficients::Ptr plane_coefficient(new pcl::ModelCoefficients());
		pcl::ModelCoefficients::Ptr projected_coefficient(new pcl::ModelCoefficients());
		//horizontal_perpendicular_plane_finder(cloud_in, surface_cloud, cloud_remain, Eigen::Vector3f(0.0, 0.0, 1.0), 0.05, 10, plane_coefficient, true); //allow 2cm error..		
		project_z_plane_line(cloud_in, cloud_projected, surface_cloud, cloud_remain, plane_coefficient, cloud_xy_plane); // use 3D-to-2D PC conversion to find a fit line and extract x-y plane
		cout <<"Size of plane coeff"<< plane_coefficient->values.size() << endl;
		cout << "1 of plane coeff" << plane_coefficient->values[0] << endl;
		cout << "2 of plane coeff" << plane_coefficient->values[1] << endl;
		cout << "3 of plane coeff" << plane_coefficient->values[2] << endl;
		cout << "4 of plane coeff" << plane_coefficient->values[3] << endl;

		surface_point_counts = surface_cloud->size();
		//std::cout << "[INFO] found surface has " << surface_point_counts << " points" << " TIME_STAMP " << now_str() << endl;
		if (surface_point_counts > 500) {
			out_vec.push_back(std::move(surface_cloud));
			plane_coeff_vec.push_back(std::move(plane_coefficient));
		}
		remaining_point_counts = cloud_remain->size();
		//std::cout << "[INFO] remaining cloud has " << remaining_point_counts << " points" << " TIME_STAMP " << now_str() << endl;
		cloud_in.swap(cloud_remain);
		cloud_remain->clear();
		cloud_remain.reset(new pcl::PointCloud<pcl::PointXYZ>);
		counts = counts + 1;
	} while (counts < 1 && remaining_point_counts >1000 && surface_point_counts > 500); //we will try get 4 planes, update@05092020-> extract vertical plane once, 3 times will get wrong PC data into surface data

	pcl::PointCloud<pcl::PointXYZ>::Ptr vertical_points(new pcl::PointCloud<pcl::PointXYZ>);
	for (int j = 0; j < out_vec.size(); j++) {
		*vertical_points += *out_vec[j];
		//pcl::copyPointCloud(*cloud_xy_plane, *vertical_points);
	}
	std::cout << "[INFO] vertical points have " << vertical_points->size() << " points" << " TIME_STAMP " << now_str() << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr boundaries_points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr NARF_points(new pcl::PointCloud<pcl::PointXYZ>); //found narf points are not stable either with bad measurement
	pcl::RangeImagePlanar::Ptr range_image_ptr(new pcl::RangeImagePlanar);
	//find_boundaries_and_NARF_points_use_range_image_method(vertical_points, boundaries_points, range_image_ptr, NARF_points, false, plane_finding_distance_thres * 5); // this could be improved																																									 //as we've already calculated the whole cloud to be range image. need to check which calc takes more time. if it's the range border calculation part then
	//pcl::copyPointCloud(*cloud_xy_plane,*vertical_points);
	find_boundaries_points_customized(vertical_points, boundaries_points, min_x, max_x, min_y, max_y);
	//find_boundaries_points_customized(cloud_xy_plane, boundaries_points, min_x, max_x, min_y, max_y);
	std::cout << "[INFO] find_boundaries_points_customized" << " TIME_STAMP " << now_str() << endl;																																								   //no need to change..
	pcl::ModelCoefficients::Ptr project_plane_coefficient(new pcl::ModelCoefficients());
	Eigen::Vector4f Allplanescentroid;
	pcl::PointCloud<pcl::PointXYZ>::Ptr narf_plane_cloud(new pcl::PointCloud<pcl::PointXYZ>); //found NARF points plane are not stable either 
	//pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_boundaries(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::PointCloud<pcl::PointXYZ>::Ptr boundaries_points(new pcl::PointCloud<pcl::PointXYZ>); //New to customized function
	pcl::PointCloud<pcl::PointXYZ>::Ptr projected_boundaries(new pcl::PointCloud<pcl::PointXYZ>);
	//horizontal_perpendicular_plane_finder(NARF_points, narf_plane_cloud, cloud_remain, Eigen::Vector3f(0.0, 0.0, 1.0), 0.2, 10, plane_coefficient, false); //allow 2cm error..		
	if (boundaries_points->points.size() < 100) //bascically wrong. 
	{
		std::cout << "[ERROR] Boundaries points size too small" << endl;
		return -1;
	}
	cout << "plane number:" << plane_coeff_vec.size() << endl;
	cout << "plane number:" << plane_coeff_vec[0]->values[0] << endl;
	project_plane_coefficient = get_most_possible_plane(plane_coeff_vec, boundaries_points, Allplanescentroid);
	std::cout << "[INFO] get_most_possible_plane" << " TIME_STAMP " << now_str() << endl;
	//radius_removal(boundaries_points, filtered_boundaries, 0.02, 4);
	std::cout << "[INFO] radius_removal" << " TIME_STAMP " << now_str() << endl;
	//project_points_to_plane(filtered_boundaries, projected_boundaries, project_plane_coefficient, 0.2); //project points to the plane with most points-- this might not be the final one.. 
	project_points_to_plane(boundaries_points, projected_boundaries, project_plane_coefficient, 0.2);

	std::cout << "[INFO] project_points_to_plane" << " TIME_STAMP " << now_str() << endl;
	if (global_debug_option) {
		//debug only
		pcl::visualization::PCLVisualizer viewer("3D Viewer");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> all_points_color_handler(potential_tray_pcd, 125, 125, 125);
		viewer.addPointCloud<pcl::PointXYZ>(potential_tray_pcd, all_points_color_handler, "all points");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> border_points_color_handler(projected_boundaries, 0, 255, 0);
		viewer.addPointCloud<pcl::PointXYZ>(projected_boundaries, border_points_color_handler, "border points");
		viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");
		//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> NARF_points_color_handler(NARF_points, 0, 0, 255);
		//viewer.addPointCloud<pcl::PointXYZ>(NARF_points, NARF_points_color_handler, "NARF points");
		//viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "NARF points");
		viewer.addText("extracted_border_points", 0, 0);
		while (!viewer.wasStopped())
		{
			viewer.spinOnce();
			pcl_sleep(0.01);
		}
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr x_cloud_remain(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ModelCoefficients::Ptr x_line_coefficients(new pcl::ModelCoefficients());
	pcl::copyPointCloud(*projected_boundaries, *cloud_temp);
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> xlines;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> ylines;
	std::vector <lineinfo> x_lines_segmented;
	std::vector <lineinfo> y_lines_segmented;
	Eigen::Vector3f xline_selection_vector, yline_selection_vector;
	xline_selection_vector = get_cross_eigen(project_plane_coefficient, Eigen::Vector3f(0.0, 1.0, 0.0));
	std::cout << "[INFO] get_cross_eigen" << " TIME_STAMP " << now_str() << endl;
	//yline_selection_vector = get_cross_eigen(project_plane_coefficient, Eigen::Vector3f(1.0, 0.0, 0.0));
	yline_selection_vector = Eigen::Vector3f(0.0, 1.0, 0.0);
	int x_line_points = 0;
	counts = 0;
	do {
		pcl::PointCloud<pcl::PointXYZ>::Ptr xline_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr seg_remain(new pcl::PointCloud<pcl::PointXYZ>);
		parallel_with_axis_line_finder(cloud_temp, xline_cloud, x_cloud_remain, xline_selection_vector, 0.01, 10, x_line_coefficients);
		//post process extracted line and segment it
		lineinfo segmented_line_info;
		cout << "x line segmentation" << "TIMESTAMP " << now_str() << endl;

		segment_lines(xline_cloud, segmented_line_info, seg_remain, x_line_coefficients, "x", potential_tray_pcd, 0.02); //this 5cm is from border line min x direction distance. around 20cm																														 //choose 5cm as it's enough. 
		if (segmented_line_info.segments.size() > 0) {
			x_lines_segmented.push_back(segmented_line_info);
		}
		*x_cloud_remain += *seg_remain;
		cloud_temp.swap(x_cloud_remain);
		x_cloud_remain->clear();
		x_cloud_remain.reset(new pcl::PointCloud<pcl::PointXYZ>);
		x_line_points = xline_cloud->size();
		xlines.push_back(std::move(xline_cloud));
		//std::cout << "[INFO] extracted x_line has " << x_line_points << " points" << " TIME_STAMP " << now_str() << endl;
		//std::cout << "[INFO] line model" << x_line_coefficients << std::endl << std::endl;
		counts++;
		if (global_debug_option) {
			//debug
			pcl::visualization::PCLVisualizer viewer("3D Viewer");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> all_points_color_handler(potential_tray_pcd, 125, 125, 125);
			viewer.addPointCloud<pcl::PointXYZ>(potential_tray_pcd, all_points_color_handler, "all points");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> x_points_color_handler(xlines.back(), 0, 255, 0);
			viewer.addPointCloud<pcl::PointXYZ>(xlines.back(), x_points_color_handler, "border points");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");
			viewer.addText("x lines extracting count " + counts, 0, 0.1);
			while (!viewer.wasStopped())
			{
				//
				viewer.spinOnce();
				pcl_sleep(0.01);
			}

		}
	//} while (x_line_points > 150 && counts < 5);
	//} while (x_line_points > 75 && counts < 4);
	} while (x_line_points > 50 && counts < 4);
	pcl::PointCloud<pcl::PointXYZ>::Ptr y_cloud_remain(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ModelCoefficients::Ptr y_line_coefficients(new pcl::ModelCoefficients());

	pcl::PointCloud<pcl::PointXYZ>::Ptr cropcloud(new pcl::PointCloud<pcl::PointXYZ>);
	Eigen::Vector4f minPoint;
	Eigen::Vector4f maxPoint;

	minPoint[0] = min_x + 0.2*(max_x - min_x);
	minPoint[1] = min_y - 0.1;
	minPoint[2] = 0;

	maxPoint[0] = max_x - 0.2*(max_x - min_x);
	maxPoint[1] = max_y + 0.1;
	maxPoint[2] = 10;

	box_crop_xyz_pcd(cloud_temp, cropcloud, minPoint, maxPoint);
	cloud_temp.swap(cropcloud);


	int y_line_points = 0;
	counts = 0;
	do {
		pcl::PointCloud<pcl::PointXYZ>::Ptr yline_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr seg_remain(new pcl::PointCloud<pcl::PointXYZ>);
		parallel_with_axis_line_finder(cloud_temp, yline_cloud, y_cloud_remain, yline_selection_vector, 0.01, 30, y_line_coefficients); //changed 30 deg to 15 deg

																																		//post process extracted line and segment it
		lineinfo segmented_line_info;
		cout << "y line segmentation" << endl;
		segment_lines(yline_cloud, segmented_line_info, seg_remain, y_line_coefficients, "y", potential_tray_pcd, 0.03);// this 3cm is from tray y direction border line minimum distance
		if (segmented_line_info.segments.size() > 0) {
			y_lines_segmented.push_back(segmented_line_info);
		}
		*y_cloud_remain += *seg_remain;
		cloud_temp.swap(y_cloud_remain);
		y_cloud_remain->clear();
		y_cloud_remain.reset(new pcl::PointCloud<pcl::PointXYZ>);
		y_line_points = yline_cloud->size();
		ylines.push_back(std::move(yline_cloud));
		//std::cout << "[INFO] extracted y_line has " << y_line_points << " points" << " TIME_STAMP " << now_str() << endl;;
		//std::cout << "[INFO] line model" << y_line_coefficients << std::endl << std::endl;
		counts++;
		if (global_debug_option)
		{
			//debug
			pcl::visualization::PCLVisualizer viewer("3D Viewer");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> all_points_color_handler(potential_tray_pcd, 125, 125, 125);
			viewer.addPointCloud<pcl::PointXYZ>(potential_tray_pcd, all_points_color_handler, "all points");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> y_points_color_handler(ylines.back(), 0, 255, 0);
			viewer.addPointCloud<pcl::PointXYZ>(ylines.back(), y_points_color_handler, "border points");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");
			viewer.addText("y lines extracting count " + counts, 0, 0.1);
			while (!viewer.wasStopped())
			{
				viewer.spinOnce();
				pcl_sleep(0.01);
			}
		}
		//} while (y_line_points > 20 && counts < 7);
	//} while (y_line_points > 20 && counts < 10);
	//} while (y_line_points > 8 && counts < 6); //要求托盘y线段必须大于4cm 以上 (8个点,每个点cover 0.5cm). 最好10cm以上.  结合follow changes(2) 的改动, 进一步缩小了 counts number 从10 变为6.
	} while (y_line_points > 8 && counts < 6); //要求托盘y线段必须大于4cm 以上 (8个点,每个点cover 0.5cm). 最好10cm以上.  结合follow changes(2) 的改动, 进一步缩小了 counts number 从10 变为6.

	pcl::PointXYZ TrackingPointA, TrackingPointB, CalculatedPoint, roughcenter, projectedcenter;

	roughcenter.x = Allplanescentroid[0];
	roughcenter.y = Allplanescentroid[1];
	roughcenter.z = Allplanescentroid[2];
	Eigen::Vector4f project_plane_eigen;
	project_plane_eigen[0] = project_plane_coefficient->values[0];
	project_plane_eigen[1] = project_plane_coefficient->values[1];
	project_plane_eigen[2] = project_plane_coefficient->values[2];
	project_plane_eigen[3] = project_plane_coefficient->values[3];
	pcl::projectPoint(roughcenter, project_plane_eigen, projectedcenter);

	int calc_status = 0; //if return -1 means calculation failed
	//calc_status = tracking_point_calculation(x_lines_segmented, y_lines_segmented, projectedcenter, TrackingPointA, TrackingPointB);
	calc_status = general_tracking_point_cal_3D(x_lines_segmented, y_lines_segmented, projectedcenter, TrackingPointA, TrackingPointB);
	//update_status = calc_status;
	cout << "calc status: " << calc_status << endl;
	if (calc_status == 0)
	{
		CalculatedPoint.x = (TrackingPointA.x + TrackingPointB.x) / 2 - yline_selection_vector[0] * 0.05; //calculated 5cm down point from trackingpointA and B
		CalculatedPoint.y = (TrackingPointA.y + TrackingPointB.y) / 2 + std::abs(yline_selection_vector[1] * 0.05);
		CalculatedPoint.z = (TrackingPointA.z + TrackingPointB.z) / 2 - yline_selection_vector[2] * 0.05;

		pcl::PointCloud<pcl::PointXYZ>::Ptr calpoints_raw(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr calpoints(new pcl::PointCloud<pcl::PointXYZ>);
		calpoints_raw->width = 3;
		calpoints_raw->height = 1;
		calpoints_raw->is_dense = true;
		calpoints_raw->points.resize(calpoints_raw->width * calpoints_raw->height);
		calpoints_raw->points[0] = TrackingPointA;
		calpoints_raw->points[1] = TrackingPointB;
		calpoints_raw->points[2] = CalculatedPoint;

		pcl::copyPointCloud(*calpoints_raw, *calpoints);

		pcl::ModelCoefficients::Ptr refined_surface_plane_coeffs(new pcl::ModelCoefficients());

		refine_cloud_surface(potential_tray_pcd, calpoints_raw, calpoints, refined_surface_plane_coeffs);
		//refine_cloud_surface(vertical_points, calpoints_raw, calpoints, refined_surface_plane_coeffs);

		TrackingPointA = calpoints->points[0];
		TrackingPointB = calpoints->points[1];
		CalculatedPoint = calpoints->points[2];

		x_center = CalculatedPoint.x+x_offset;
		y_center = CalculatedPoint.y+y_offset;
		z_center = CalculatedPoint.z+z_offset;

		cout << "x_offset: " << x_offset << endl;
		cout << "y_offset: " << y_offset << endl;
		cout << "z_offset: " << z_offset << endl;
		cout << "angle_offset: " << h_angle_offset << endl;
		// atan2(a, b) -pi 到 pi 之间; atan(a/b) -pi/2 到 pi/2 之间
		//pose_x = atan2(project_plane_coefficient->values[0], project_plane_coefficient->values[2]) / 3.14159*180;
		//pose_y = atan2(project_plane_coefficient->values[1], project_plane_coefficient->values[2]) / 3.14159 * 180;
		pose_x = atan(refined_surface_plane_coeffs->values[0] / refined_surface_plane_coeffs->values[2]) / 3.14159 * 180 + h_angle_offset;
		pose_y = atan(refined_surface_plane_coeffs->values[1] / refined_surface_plane_coeffs->values[2]) / 3.14159 * 180;
		// Check the points 1cm below tracking A and B if they are in the hole 
		cout << "refined_surface_plane_coeffs->values[0]:   " << refined_surface_plane_coeffs->values[0] << endl;
		cout << "refined_surface_plane_coeffs->values[1]:   " << refined_surface_plane_coeffs->values[1] << endl;
		cout << "refined_surface_plane_coeffs->values[2]:   " << refined_surface_plane_coeffs->values[2] << endl;

		std::cout << "[Result] tracking point A position: " << TrackingPointA.x << " " << TrackingPointA.y << " " << TrackingPointA.z << " TIME_STAMP " << now_str() << endl;
		std::cout << "[Result] tracking point B position: " << TrackingPointB.x << " " << TrackingPointB.y << " " << TrackingPointB.z << " TIME_STAMP " << now_str() << endl;
		std::cout << "[Result] tray center position: " << CalculatedPoint.x << " " << CalculatedPoint.y << " " << CalculatedPoint.z << " TIME_STAMP " << now_str() << endl;
		std::cout << "[Result] tray surface eigen factor: " << refined_surface_plane_coeffs->values[0] << " " << refined_surface_plane_coeffs->values[1] << " " << refined_surface_plane_coeffs->values[2] << " " << refined_surface_plane_coeffs->values[3] << " TIME_STAMP " << now_str() << endl;
		cout << "----------" << endl;
		cout << "Tray center location(x/y/z):  " << CalculatedPoint.x << "m/ " << CalculatedPoint.y << "m/ " << CalculatedPoint.z << "m/ " << endl;
		cout << "Tray surface horizontal angle(Degree):  " << pose_x << endl;
		cout << "Tray surface vertical angle(Degree):   " << pose_y << endl;
		cout << "----------" << endl;
		if (global_tray_rack_dis_calc) {
			cout << "calculate good and rack......................" << endl;
			cout << "number of objects around tray..........." << full_pcd->size() << endl;
			if (full_pcd->size() == 0)
			{
				full_pcd = pcd_read;
				cout << "copy pcd data to full pcd..." << endl;
			}
			std::vector<ThingNotTray> things_around_tray;
			things_around_tray = find_things_ontop_and_around_tray(full_pcd, calpoints, project_plane_coefficient);
			//bool goods_found = false;

			cout << "number of objects around tray..........." << things_around_tray.size() << endl;
			int num_of_things = things_around_tray.size();
			for (int thing_ind = 0; thing_ind < num_of_things; ++thing_ind) {
				if (things_around_tray[thing_ind].goods == true) {
					goods_found = true;
					if (thing_ind > 0) {
						//there is an object on the left of goods
						pcl::PointXYZ temppoint = things_around_tray[thing_ind - 1].right_most_point;
						std::cout << "[Result] an object on the left of goods, its right most point is " << temppoint.x << " " << temppoint.y << " " << temppoint.z << " TIME_STAMP " << now_str() << endl;
						rack_found = true;
						right_most_point_rack_x = temppoint.x;
						right_most_point_rack_y = temppoint.y;
						right_most_point_rack_z = temppoint.z;
					}
					if (thing_ind < num_of_things - 1) {
						//there is an object on the right of goods
						pcl::PointXYZ temppoint = things_around_tray[thing_ind + 1].left_most_point;
						std::cout << "[Result] an object on the right of goods, its left most point is " << temppoint.x << " " << temppoint.y << " " << temppoint.z << " TIME_STAMP " << now_str() << endl;
						rack_found = true;
						left_most_point_rack_x = temppoint.x;
						left_most_point_rack_y = temppoint.y;
						left_most_point_rack_z = temppoint.z;
					}
					// left most point
					pcl::PointXYZ temppoint = things_around_tray[thing_ind].left_most_point;
					std::cout << "[Result] goods detected, its left most point is " << temppoint.x << " " << temppoint.y << " " << temppoint.z << " TIME_STAMP " << now_str() << endl;
					left_most_point_goods_x = temppoint.x;
					left_most_point_goods_y = temppoint.y;
					left_most_point_goods_z = temppoint.z;
					cout << "left_most_point_goods_x" << left_most_point_goods_x << endl;
					// right most point
					temppoint = things_around_tray[thing_ind].right_most_point;
					std::cout << "[Result] goods detected, its right most point is " << temppoint.x << " " << temppoint.y << " " << temppoint.z << " TIME_STAMP " << now_str() << endl;
					// front most point
					right_most_point_goods_x = temppoint.x;
					right_most_point_goods_y = temppoint.y;
					right_most_point_goods_z = temppoint.z;
					temppoint = things_around_tray[thing_ind].front_most_point;
					std::cout << "[Result] goods detected, its front most point is " << temppoint.x << " " << temppoint.y << " " << temppoint.z << " TIME_STAMP " << now_str() << endl;
					front_most_point_goods_x = temppoint.x;
					front_most_point_goods_y = temppoint.y;
					front_most_point_goods_z = temppoint.z;
					std::cout << "[Result] goods facing eigen " << things_around_tray[thing_ind].camera_facing_eigen.values[0] << " " << things_around_tray[thing_ind].camera_facing_eigen.values[1];
					std::cout << " " << things_around_tray[thing_ind].camera_facing_eigen.values[2] << " " << things_around_tray[thing_ind].camera_facing_eigen.values[3] << " TIME_STAMP " << now_str() << endl;
					goods_pose_x = atan(things_around_tray[thing_ind].camera_facing_eigen.values[0] / things_around_tray[thing_ind].camera_facing_eigen.values[2]) / 3.14159 * 180;
					goods_pose_y = atan(things_around_tray[thing_ind].camera_facing_eigen.values[1] / things_around_tray[thing_ind].camera_facing_eigen.values[2]) / 3.14159 * 180;
					camera_facing_eigen_goods = things_around_tray[thing_ind].camera_facing_eigen.values[0];
					cout << "good status: " << goods_found << endl;
					cout << "rack status: " << rack_found << endl;
				}
			}
			if (!goods_found) {
				for (int thing_ind = 0; thing_ind < num_of_things; ++thing_ind) {
					std::cout << "[Result] No goods found. Below objects found close to tray " << endl;
					std::cout << "[Result] Object_" << thing_ind;
					std::cout << " Left_most_point " << things_around_tray[thing_ind].left_most_point.x << " " << things_around_tray[thing_ind].left_most_point.y << " " << things_around_tray[thing_ind].left_most_point.z << endl;
					std::cout << "[Result] Object_" << thing_ind;
					std::cout << " Right_most_point " << things_around_tray[thing_ind].right_most_point.x << " " << things_around_tray[thing_ind].right_most_point.y << " " << things_around_tray[thing_ind].right_most_point.z << endl;
				}
			}

			if (global_show_key_points_nearby_objects) {
				pcl::PointCloud<pcl::PointXYZ>::Ptr keyObjectpoints(new pcl::PointCloud<pcl::PointXYZ>);
				for (int thing_ind = 0; thing_ind < num_of_things; ++thing_ind) {
					keyObjectpoints->points.push_back(things_around_tray[thing_ind].center);
					keyObjectpoints->points.push_back(things_around_tray[thing_ind].front_most_point);
					keyObjectpoints->points.push_back(things_around_tray[thing_ind].left_most_point);
					keyObjectpoints->points.push_back(things_around_tray[thing_ind].right_most_point);
				}
				pcl::PointCloud<pcl::PointXYZ>::Ptr refined_boundaries(new pcl::PointCloud<pcl::PointXYZ>);
				project_points_to_plane(boundaries_points, refined_boundaries, refined_surface_plane_coeffs, 0.2);

				pcl::visualization::PCLVisualizer viewer("Refined Surface Viewer");
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> all_points_color_handler(full_pcd, 125, 125, 125);
				viewer.addPointCloud<pcl::PointXYZ>(full_pcd, all_points_color_handler, "all points");
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> border_points_color_handler(refined_boundaries, 0, 255, 0);
				viewer.addPointCloud<pcl::PointXYZ>(refined_boundaries, border_points_color_handler, "border points");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> Calpoints_color_handler(calpoints, 255, 0, 0);
				viewer.addPointCloud<pcl::PointXYZ>(calpoints, Calpoints_color_handler, "Calculated points");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Calculated points");
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> Key_ob_points_color_handler(keyObjectpoints, 255, 0, 255);
				viewer.addPointCloud<pcl::PointXYZ>(keyObjectpoints, Key_ob_points_color_handler, "Nearby Object Key points");
				viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Nearby Object Key points");

				while (!viewer.wasStopped())
				{
					viewer.spinOnce();
					pcl_sleep(0.01);
				}
			}

		}

		//debug only

		if (global_display_output_cloud) {
			//reproject use refined surface
			pcl::PointCloud<pcl::PointXYZ>::Ptr refined_boundaries(new pcl::PointCloud<pcl::PointXYZ>);
			project_points_to_plane(boundaries_points, refined_boundaries, refined_surface_plane_coeffs, 0.2);

			pcl::visualization::PCLVisualizer viewer("Refined Surface Viewer");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> all_points_color_handler(potential_tray_pcd, 125, 125, 125);
			viewer.addPointCloud<pcl::PointXYZ>(potential_tray_pcd, all_points_color_handler, "all points");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> border_points_color_handler(refined_boundaries, 0, 255, 0);
			viewer.addPointCloud<pcl::PointXYZ>(refined_boundaries, border_points_color_handler, "border points");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> Calpoints_color_handler(calpoints, 255, 0, 0);
			viewer.addPointCloud<pcl::PointXYZ>(calpoints, Calpoints_color_handler, "Calculated points");
			viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "Calculated points");

			while (!viewer.wasStopped())
			{
				viewer.spinOnce();
				pcl_sleep(0.01);
			}

		}
	}
	else {
		std::cout << "[Error] CalCulation Failed" << " TIME_STAMP " << now_str() << endl;
	}
	return calc_status;
}

//rewrite this Sep 9-15th -GD
int CNN_find_cropped_targets_from_RGB_and_preprocessed_pcd(pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_preprocess, std::vector<Rect> potentioal_2D_detections, std::vector<pcl::PointCloud<pcl::PointXYZ>>& cropped_targets, float downsamplesize)
{
	int twoDfound_number = potentioal_2D_detections.size();
	int post_cropped_pcd_thres = 4000; //5000
	for (int k = 0; k < twoDfound_number; k++)
	{
		Rect current_zone = potentioal_2D_detections[k];
		pcl::PointCloud<pcl::PointXYZ>::Ptr current_pcd(new pcl::PointCloud<pcl::PointXYZ>);
		Eigen::Vector4f minPoint, maxPoint;
		float min_x, min_y, min_z, max_x, max_y, max_z; //this is calculated phsical x, y in meter
		int x_top_left_pixel, y_top_left_pixel, x_bot_right_pixel, y_bot_right_pixel; // this is pixel
		x_top_left_pixel = current_zone.x;
		y_top_left_pixel = current_zone.y;
		x_bot_right_pixel = x_top_left_pixel + current_zone.width;
		y_bot_right_pixel = y_top_left_pixel + current_zone.height;

		cout << "detection zone matrix " << x_top_left_pixel << "_" << y_top_left_pixel << "_" << x_bot_right_pixel << "_" << y_bot_right_pixel << endl;
		current_pcd->width = current_zone.width;
		current_pcd->height = current_zone.height;
		int point_counter = 0;

		for (int y_idx = y_top_left_pixel; y_idx < y_bot_right_pixel; y_idx++) {
			for (int x_idx = x_top_left_pixel; x_idx < x_bot_right_pixel; x_idx++) {
				pcl::PointXYZ point;
				point.x = (static_cast<float> (x_idx) - camera_cal_cx) *  camera_depthmat_480_640[y_idx][x_idx] / camera_cal_fx / 1000.0;
				point.y = (static_cast<float> (y_idx) - camera_cal_cy) *  camera_depthmat_480_640[y_idx][x_idx] / camera_cal_fy / 1000.0;
				point.z = camera_depthmat_480_640[y_idx][x_idx] / 1000.0;
				current_pcd->points.push_back(point);
				point_counter++;
			}
		}


		pcl::PointCloud<pcl::PointXYZ>::Ptr post_cropped_pcd(new pcl::PointCloud<pcl::PointXYZ>);
		preprocess_tray_pcd(current_pcd, post_cropped_pcd, downsamplesize);
		if (post_cropped_pcd->points.size() > post_cropped_pcd_thres) {
			cropped_targets.push_back(std::move(*post_cropped_pcd));
		}
	}
	return cropped_targets.size();
}

// Get the names of the output layers
std::vector<String> getOutputsNames(const dnn::Net& net)
{
	static std::vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i) {
			names[i] = layersNames[outLayers[i] - 1];
		}
	}
	return names;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
// this is more like yolo kind of thing. tensorflow model doesn't fit. 
//void postprocess(Mat& frame, const std::vector<Mat>& outs, std::vector<int>& class_ID_out, std::vector<cv::Rect>& outboxes)
void rgb_detection_postprocess(Mat& frame, std::vector<float> confidences, float confThreshold, std::vector<cv::Rect>& inputboxes, std::vector<cv::Rect>& outboxes)
{
	float nmsThreshold = 0.3;

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	dnn::NMSBoxes(inputboxes, confidences, confThreshold, nmsThreshold, indices);

	struct sortbyarea {
		bool operator() (cv::Rect pt1, cv::Rect pt2) { return (pt1.area() > pt2.area()); }
	} sortbyarea_cv_rects;

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = inputboxes[idx];
		cv::rectangle(frame, box, cv::Scalar(0, 255, 0));
		if (box.area() > 500) //remove too small objects
		{
			//class_ID_out.push_back(classIds[idx]);
			outboxes.push_back(box);
		}
		std::sort(outboxes.begin(), outboxes.end(), sortbyarea_cv_rects);
	}
	if (save_data_file == true)
	{
		cv::imwrite("../temp/remove_duplicating_boxes.jpg", frame);
	}
}

void CAN_message_send()
{
	std::stringstream ss;
	unsigned int id;
	//ss << std::hex << frame_ID;
	//ss >> id;
	id = 480;
	unsigned int id_T = 408;
	//ss << std::hex << frame_T_ID;
	//ss >> id_T;
	unsigned int id_angle = 481;
	//ss << std::hex << frame_angle_ID;
	//ss >> id_angle;
	unsigned int sendType = 1; // 0: normal mode 1: send once
	unsigned char data_from_text[8] = {};
	unsigned char T_data_from_text[8] = {};
	unsigned char angle_data_from_text[8] = {};
	QString str_data = frame_Data;
	QString str_T_data = frame_T_Data;
	QString str_angle_data = frame_angle_Data;
	QString str_, str_T, str_angle;
	// zero out CAN data

	// fill CAN msg data with formated frames
	for (int i = 0; i < 6; i++)   // data frame
	{
		str_ = str_data.mid(2 * i, 2);
		//cout << str_.toStdString().c_str() << endl;
		//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
		//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
		data_from_text[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
	}

	for (int i = 0; i < 3; i++)  // transmit frame
	{
		str_T = str_T_data.mid(2 * i, 2);
		//cout << str_T.toStdString().c_str() << endl;
		//T_data_from_text[i] = hex_str_to_int((unsigned char *)str_T.toStdString().c_str());
		T_data_from_text[i] = hex_str_to_int((unsigned char *)str_T.toStdString().c_str());
	}

	for (int i = 0; i < 2; i++)   // angle data
	{
		str_angle = str_angle_data.mid(2 * i, 2);
		//cout << str_angle.toStdString().c_str() << endl;
		//angle_data_from_text[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
		angle_data_from_text[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
	}
	cout << "Data frame id: " << id << endl;
	cout << "Tran frame id: " << id_T << endl;
	cout << "Angle frame id: " << id_angle << endl;
	MyCANControlThread->TransmitCANThread(id_T, sendType, 3, (char *)T_data_from_text);
	cout << "T_data_from_text: " << hex << T_data_from_text << endl;
	MyCANControlThread->TransmitCANThread(id, sendType, 6, (char *)data_from_text);
	cout << "data_from_text: " << hex << data_from_text << endl;
	MyCANControlThread->TransmitCANThread(id_angle, sendType, 2, (char *)angle_data_from_text);
	cout << "angle_data_from_text: " << hex << angle_data_from_text << endl;

};

void StartDevice_CAN()//启动设备
{
	//cout << "create a thread to open CAN device" << endl;
	MyCANControlThread->OpenCANThread();
	//cout << "start CAN thread" << endl;
	if (start_status)
	{
		MyCANControlThread->start();
		cout << "connect to CAN device done" << endl;
	}
	else
	{
		cout << "fail to connect CAN deivce" << endl;
	}
	cout << "Start device done" << endl;
}

void CloseDevice_CAN()
{
	MyCANControlThread->stop();
	MyCANControlThread->CloseCANThread();
	cout << "close device: start_status " << start_status << endl;
}

int pico_read();
int CNN_main();

void CAN_Send_Thread::CAN_msg_send()
{
	if (REV_start_CAN == true)
	{
		if (start_status)
		{
			if (camera_input == true)
			{
				pico_read();
				cout << "New pre-pcd Timer created for camera input" << endl;
				pre_pcd_thread.start();
				pre_pcd_thread.preprocess_pcd();
				int dwMask = 0001;
				SetThreadAffinityMask(GetCurrentThread(), dwMask);
				cout << "New pre-pcd assign to CPU_01" << endl;
				cout << "New pre-pcd Timer start" << endl;
			}
			else
			{
				//pico_read();
				//cout << "New pre-pcd Timer created for offline data" << endl;
				//pre_pcd_thread.start();
				//pre_pcd_thread.preprocess_pcd();
				//int dwMask = 0001;
				//SetThreadAffinityMask(GetCurrentThread(), dwMask);
				//cout << "New pre-pcd assign to CPU_01" << endl;
				//cout << "New pre-pcd Timer start" << endl;
			}
			CNN_main();
			cout << "good status 1: " << goods_found << endl;
			cout << "rack status 1: " << rack_found << endl;
			cout << "Set CAN mode to: " << CAN_mode << " from start msg. at " << "TIMESTAMP " << now_str() << endl;
			//cout << "Start with CAN msg packing....1" << endl;
			CAN_protocol CAN_msg(CAN_mode);
			//cout << "Done with CAN msg packing....2" << endl;
			cout << "good status 2: " << goods_found << endl;
			cout << "rack status 2: " << rack_found << endl;
			REV_start_CAN = false;
		}
		else
		{
			cout << "----------------------------------------------------------" << endl;
			cout << "No CAN device is connected and run Tray Detection ONLY...." << endl;
			cout << "----------------------------------------------------------" << endl;
			if (camera_input == true)
			{
				pico_read();
				cout << "New pre-pcd Timer created" << endl;
				pre_pcd_thread.start();
				pre_pcd_thread.preprocess_pcd();
				int dwMask = 0001;
				SetThreadAffinityMask(GetCurrentThread(), dwMask);
				cout << "New pre-pcd assign to CPU_01" << endl;
				cout << "CAN Check Timer start" << endl;
			}
			else
			{
			}
			CNN_main();
			REV_start_CAN = false;
		}
		cout << "Completed test round is finished...." << endl;
		goods_found = false;
		rack_found = false;
		// format data and send CAN data
		std::stringstream ss;
		unsigned int id;
		id = 480;
		unsigned int id_T = 408;
		unsigned int id_angle = 481;

		unsigned int id_good_left = 482;
		unsigned int id_good_right = 483;
		unsigned int id_good_front = 484;
		unsigned int id_good_angle = 485;
		unsigned int id_rack_left = 486;
		unsigned int id_rack_right = 487;

		unsigned int sendType = 1; // 0: normal mode 1: send once
		unsigned char data_from_text[8] = {};
		unsigned char data_from_text_good_left[8] = {};
		unsigned char data_from_text_good_right[8] = {};
		unsigned char data_from_text_good_front[8] = {};
		unsigned char data_from_text_rack_left[8] = {};
		unsigned char data_from_text_rack_right[8] = {};
		unsigned char angle_data_from_text_good[8] = {};
		unsigned char T_data_from_text[8] = {};
		unsigned char angle_data_from_text[8] = {};
		QString str_data = frame_Data;
		QString str_good_left_data = frame_good_left_Data;
		QString str_good_right_data = frame_good_right_Data;
		QString str_good_front_data = frame_good_front_Data;
		QString str_rack_left_data = frame_rack_left_Data;
		QString str_rack_right_data = frame_rack_right_Data;

		QString str_T_data = frame_T_Data;
		QString str_angle_data = frame_angle_Data;
		QString str_good_angle_data = frame_good_angle_Data;
		QString str_, str_T, str_angle;

		for (int i = 0; i < 4; i++)
		{
			frame_Data[i] = 0;
		}

		// fill CAN msg data with formated frames
		for (int i = 0; i < 6; i++)   // data frame
		{
			str_ = str_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 6; i++)   // data good left frame
		{
			str_ = str_good_left_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text_good_left[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 6; i++)   // data right left frame
		{
			str_ = str_good_right_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text_good_right[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 6; i++)   // data good front frame
		{
			str_ = str_good_front_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text_good_front[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 2; i++)   // angle data
		{
			str_angle = str_good_angle_data.mid(2 * i, 2);
			cout << str_angle.toStdString().c_str() << endl;
			//angle_data_from_text[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
			angle_data_from_text_good[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
		}

		for (int i = 0; i < 6; i++)   // data rack left frame
		{
			str_ = str_rack_left_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text_rack_left[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 6; i++)   // data rack right frame
		{
			str_ = str_rack_right_data.mid(2 * i, 2);
			cout << str_.toStdString().c_str() << endl;
			//对于char类型数据，我们获取其十进制表示形式较容易一些，故直接将字符形式转换为整型即可！
			//data_from_text[i] = hex_str_to_int((unsigned char *)str_.section(' ', i, i).trimmed().toStdString().c_str());
			data_from_text_rack_right[i] = hex_str_to_int((unsigned char *)str_.toStdString().c_str());
		}

		for (int i = 0; i < 3; i++)  // transmit frame
		{
			str_T = str_T_data.mid(2 * i, 2);
			cout << str_T.toStdString().c_str() << endl;
			//T_data_from_text[i] = hex_str_to_int((unsigned char *)str_T.toStdString().c_str());
			T_data_from_text[i] = hex_str_to_int((unsigned char *)str_T.toStdString().c_str());
		}

		for (int i = 0; i < 2; i++)   // angle data
		{
			str_angle = str_angle_data.mid(2 * i, 2);
			cout << str_angle.toStdString().c_str() << endl;
			//angle_data_from_text[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
			angle_data_from_text[i] = hex_str_to_int((unsigned char *)str_angle.toStdString().c_str());
		}
		//cout << "Data frame id: " << id << endl;
		//cout << "Tran frame id: " << id_T << endl;
		//cout << "Angle frame id: " << id_angle << endl;
		MyCANControlThread->TransmitCANThread(id_T, sendType, 3, (char *)T_data_from_text);
		//cout << "Trans_data: " << hex << T_data_from_text << endl;
		//unsigned char T_data_from_text = {};
		MyCANControlThread->TransmitCANThread(id, sendType, 6, (char *)data_from_text);
		//cout << "tray_data: " << hex << data_from_text << endl;
		//unsigned char  data_from_text = {};
		MyCANControlThread->TransmitCANThread(id_angle, sendType, 2, (char *)angle_data_from_text);
		//cout << "angle_data_from_text: " << hex << angle_data_from_text << endl;
		//unsigned char angle_data_from_text = {};
		// transmit good and rack data
		if (CAN_mode == 1 || CAN_mode == 3) // send goods info for FFDD and FFBB
		//if (CAN_mode == 1) // send goods info for FFDD 
		{
			//cout << "Send goods data......" << endl;
			MyCANControlThread->TransmitCANThread(id_good_left, sendType, 6, (char *)data_from_text_good_left);
			//cout << "good data_left: " << hex << data_from_text_good_left << endl;
			//unsigned char data_from_text_good_left = {};
			MyCANControlThread->TransmitCANThread(id_good_right, sendType, 6, (char *)data_from_text_good_right);
			//cout << "good data_right: " << hex << data_from_text_good_right << endl;
			//unsigned char data_from_text_good_right = {};
			MyCANControlThread->TransmitCANThread(id_good_front, sendType, 6, (char *)data_from_text_good_front);
			//cout << "good data_front: " << hex << data_from_text_good_front << endl;
			//unsigned char data_from_text_good_front = {};
			MyCANControlThread->TransmitCANThread(id_good_angle, sendType, 2, (char *)angle_data_from_text_good);
			//cout << "good angle_data: " << hex << angle_data_from_text_good << endl;
			//unsigned char angle_data_from_text_good = {};
		}
		if (CAN_mode == 2 || CAN_mode == 3)
		//if (CAN_mode == 2)
		{
			//cout << "Send rack data......" << endl;
			MyCANControlThread->TransmitCANThread(id_rack_left, sendType, 6, (char *)data_from_text_rack_left);
			//cout << "rack data_left: " << hex << data_from_text << endl;
			//unsigned char data_from_text_rack_left = {};
			MyCANControlThread->TransmitCANThread(id_rack_right, sendType, 6, (char *)data_from_text_rack_right);
			//cout << "rack data_right: " << hex << data_from_text << endl;
			//unsigned char data_from_text_rack_right = {};
		}

	}
};


int CNN_main() {
	std::string pb_file_name = "./dataSW/08022020/820m.dat"; //ssd_v1_tray_inception.pb = 092619m.dat
	std::string pbtxt = "./dataSW/08022020/820t.dat"; //cvssd_v1_txt_graph.pbtxt = 092619t.dat
	std::string offline_RGB_folder = "./test_folder/mappedRGB"; // "../test_folder/mappedRGB";
	std::string offline_depth_folder = "./test_folder/depthData"; // "../test_folder/depthData";
	float downsamplesize = 0.005; //5mm resolution, raw pcd will be filtered by distance, down sampled, gauss filtered

								  ////2D tracker controls
	cv::Mat copy_frame; // for fast 2D object tracking algothrism. needs a baseline Mat frame first. 
	Ptr<Tracker> tracker;//2D object traction
						 //tracker = TrackerKCF::create();//2D object tracker //found detection windows may slip. 
	tracker = TrackerMedianFlow::create(); //very good rectangle tracking, but once fail need to reinitialize. 
	bool twoDtracker_enable = false;
	bool twoDtracker_initialize = true;
	bool twoDtracker_OK = false; // this is what the twoDtracker thinks. not through 3D calculation
	bool RCNN_ever_success = false;
	Rect2d twoDtrackingbox;

	///RCNN controls
	bool RCNN_ENABLE = true;

	int run_iterations_count = 1; //how many frames we want to calculate. can set a super large number to run for long time

								  //below initializations are for testing offline data
	std::vector<std::string> offline_rgb_series;
	std::vector<std::string> offline_depth_series;
	std::vector<cv::String> fn;
	cv::glob(offline_RGB_folder + "/*.jpg", fn, false);
	//cv::glob(rgb_igb_name, fn, false);
	if (camera_input == false)
	{
		int count = fn.size();
		for (int i = 0; i < count; i++)
		{
			offline_rgb_series.push_back(fn[i]);
			cout << "[INFO] Found offline RGB data: " << fn[i] << endl;
		}
		cv::glob(offline_depth_folder + "/*.txt", fn, false);
		//cv::glob(depthDatafile, fn, false);
		count = fn.size();
		for (int i = 0; i < count; i++)
		{
			offline_depth_series.push_back(fn[i]);
			cout << "[INFO] Found offline Depth data: " << fn[i] << endl;
		}
	}
	//faster-RCNN model initialization
	if (DNN_init == false)
	{
		faster_rcnn_net = dnn::readNetFromTensorflow(pb_file_name, pbtxt);
		cout << "load faster RCNN model at TIMESTAMP " << now_str() << endl;
		DNN_init = true;
	}
	else
	{
	}
	if (camera_input == false)
	{	//this is offline debug mode
		run_iterations_count = offline_rgb_series.size();
	}

	//above initialization shall only run once. 

	for (int k = 0; k < run_iterations_count; k++) {
		//this run_iterations_count is default 1 for online mode. offline mode, this will loop all offline data in folder
		std::vector<Rect> potentioal_2D_detections_raw;
		std::vector<Rect> potentioal_2D_detections_fine;
		if (camera_input == false)
		{
			rgb_image = imread(offline_rgb_series[k]);
			cout << "load offline RGB image file: " << offline_rgb_series[k] << endl;
			cout << "load offline depth data file: " << offline_depth_series[k] << endl;
			offline_debug_update_customer_camera_parameters(); //place holder
		}
		else
		{
			rgb_image = camera_mappedRGBMat;
			cout << "load camera data through camera_mappedRGBMat " << " at TIMESTAMP " << now_str() << endl;
		}
		const size_t inWidth = rgb_image.size[1];
		const size_t inHeight = rgb_image.size[0];
		cout << "image width " << inWidth << endl;
		cout << "image height " << inHeight << endl;
		if (inWidth * inHeight == 0) {
			cout << "[FATAL ERROR] camera image invalid, force to exit" << endl;
			return -1;
		}

		bool twoDtracker_happen = false;
		bool RCNN_happen = false;
		//detection worst case will run twice. namely 2D tracking and RCNN
		for (int dtk = 0; dtk < 2; ++dtk) {

			if (twoDtracker_enable) {
				cout << "TWO D tracking starts at " << "TIMESTAMP " << now_str() << endl;
				copy_frame = rgb_image.clone();
				twoDtracker_OK = tracker->update(copy_frame, twoDtrackingbox);
				//cout << "TWO D tracking done at " << "TIMESTAMP " << now_str() << endl;

				twoDtracker_happen = true;
				if (twoDtracker_OK) {
					cout << "TWO D tracking success at " << "TIMESTAMP " << now_str() << endl;
					RCNN_ENABLE = false;
					potentioal_2D_detections_fine.push_back(twoDtrackingbox);
					cv::rectangle(copy_frame, twoDtrackingbox, Scalar(255, 0, 0), 2, 1);
					if (save_data_file == true)
					{
						std::string imgname = "../temp/twoD_tracker_found_" + now_str() + ".jpg";
						cv::imwrite(imgname, copy_frame);
					}
				}
				else {
					RCNN_ENABLE = true;
					twoDtracker_initialize = true; //once fail there is no recovery... need to reinitialize
					cout << "TWO D tracking fails at " << "TIMESTAMP " << now_str() << endl;
				}
			}
			if (RCNN_ENABLE) {
				///////////////////////////////////////CNN////////////////////////////////////////////////////////////////
				RCNN_happen = true;
				const double inScaleFactor = 1.0;
				cv::Mat inputblob;
				//dnn::blobFromImage(rgb_image, inputblob, inScaleFactor, cv::Size(1280, 960), Scalar(0, 0, 0), true, false);
				dnn::blobFromImage(rgb_image, inputblob, inScaleFactor, cv::Size(300, 300), Scalar(0, 0, 0), true, false);
				faster_rcnn_net.setInput(inputblob);
				cout << "[INFO] O.R. started TIMESTAMP " << now_str() << endl;

				Mat detection;
				detection = faster_rcnn_net.forward();
				Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
				int tray_count = 0;
				float confidenceThreshold = 0.1; //0.3

				int xLeftBottom = 0;
				int yLeftBottom = 0;
				int xRightTop = 0;
				int yRightTop = 0;

				Rect object((int)xLeftBottom, (int)yLeftBottom,
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				std::vector<float> confidences;
				cv::Mat temprgb = rgb_image.clone();
				cv::Mat temprgb_2 = rgb_image.clone();

				for (int i = 0; i < detectionMat.rows; i++)
				{
					float confidence = detectionMat.at<float>(i, 2);

					if (confidence > confidenceThreshold)
					{
						int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * rgb_image.cols);
						int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * rgb_image.rows);
						int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * rgb_image.cols);
						int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * rgb_image.rows);

						Rect object((int)xLeftBottom, (int)yLeftBottom,
							(int)(xRightTop - xLeftBottom),
							(int)(yRightTop - yLeftBottom));

						//remove too small objects. This is 2D sanity check. later on could do more thorough check based on rgb_image
						if (twoD_RGB_sanity_check_valid(object, temprgb, confidence))
						{
							confidences.push_back((float)confidence);
							potentioal_2D_detections_raw.push_back(object);
							//String label = "Tray: " + cast_to_string(confidence);
							String confidence_str = std::to_string(confidence);
							String label = "Tray: " + confidence_str;
							int baseLine = 0;
							cv::rectangle(temprgb, object, Scalar(0, 255, 0));
							Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
							cv::rectangle(temprgb, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
								Size(labelSize.width, labelSize.height + baseLine)),
								Scalar(255, 255, 255), FILLED);
							putText(temprgb, label, Point(xLeftBottom, yLeftBottom),
								FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
							tray_count++;
						}
					}
				}
				//remove highly duplicating boxes
				rgb_detection_postprocess(temprgb_2, confidences, confidenceThreshold, potentioal_2D_detections_raw, potentioal_2D_detections_fine);
				cout << "[INFO] 2D object recognization done , found " << potentioal_2D_detections_fine.size() << " possible areas " << "TIMESTAMP " << now_str() << endl;
				/////////////////////////////////////////////////////////////////CNN DONE/////////////////////////////////////////////////////////////////

			}
			// Convert depth data to point cloud
			// define in beginnings
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_read(new pcl::PointCloud<pcl::PointXYZ>);
			//pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_preprocess(new pcl::PointCloud<pcl::PointXYZ>);
			std::vector <pcl::PointCloud<pcl::PointXYZ>> cropped_targets;

			if (camera_input == false)
			{
				int count = 1;
				for (int i = 0; i < count; i++)
				{
					pcd_read = load_depth_data_from_txt_640_480(offline_depth_series[k]);
					pcd_ready = true;
					cout << "convert depth txt to pcd data.....pcd_read completed" << "TIMESTAMP " << now_str() << endl;
				}
			}
			else
			{
				pcd_read = load_depth_data_from_camera_depthmat_640_480();
				pcd_ready = true;
				cout << "load camera data from camera_depthmat_640_480, pcd_read completed" << "TIMESTAMP " << now_str() << endl;
			}

			//preprocess_tray_pcd(pcd_read, pcd_preprocess, downsamplesize);//filter 0.5m to 4m, down sample 5mm voexl. 
			// use thread to treat pcd filtering 
			if (pre_pcd_flag == true)
			{
				cout << "pcd data is preprocessed" << endl;
			}
			else {
				cout << "pcd data is not preprocessed yet....." << endl;
			}
			int cal_status = -1; //if cal_status = 0 means calculation success. -1 means failure
			if (potentioal_2D_detections_fine.size() > 0)
			{
				cout << "pcd preprocess size: " << pcd_preprocess->size() << endl;
				int twoDfound_number = CNN_find_cropped_targets_from_RGB_and_preprocessed_pcd(pcd_preprocess, potentioal_2D_detections_fine, cropped_targets, downsamplesize);
				cout << "pcd preprocess size: " << pcd_preprocess->size() << endl;
				cout << "twoDfound_number: " << twoDfound_number << endl;
				for (int k = 0; k < twoDfound_number; k++)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr potential_tray_pcd(new pcl::PointCloud<pcl::PointXYZ>);
					*potential_tray_pcd = cropped_targets[k];
					//pose_calculator_icp(potential_tray_pcd, partial_object_cloud, pcd_preprocess, points_of_interest);
					cal_status = pose_calculator_geometric_relationship(potential_tray_pcd, pcd_preprocess, 2 * downsamplesize);

					if (cal_status == 0) {
						//update tracker information
						twoDtracker_enable = true;
						RCNN_ever_success = true;
						//
						update_status = cal_status;
						//
						if (twoDtracker_initialize) {
							tracker = TrackerMedianFlow::create();
							twoDtrackingbox = potentioal_2D_detections_fine[k];
							copy_frame = rgb_image.clone();
							tracker->init(copy_frame, twoDtrackingbox);
						}
						twoDtracker_initialize = false;
						RCNN_ENABLE = false;
						break;
					}
				}

				if (cal_status != 0) {
					cout << "[INFO] 2D detection found some potentials, but not able to get 3D calculation done in this area" << endl;
					potentioal_2D_detections_fine.clear();
					//
					update_status = cal_status;
					//
				}
			}
			else
			{
				cout << "[INFO] 2D detection: No tray detected in the current view" << endl;
			}

			if (cal_status != 0) {
				twoDtracker_enable = false || global_force_2D_tracting_AON && RCNN_ever_success; //previous tracking failed. need to rerun RCNN. 
				twoDtracker_initialize = true; //next success RCNN need to reinitialize twoDtracker model
				RCNN_ENABLE = true;
			}
			else {
				update_status = 0;  // update status 0 success; -1 failed
				break;
			} // if sucess we finish this frame's calculation
			if (cal_status != 0 && RCNN_happen) 
			{ 
				update_status = cal_status;
				break; } //if RCNN happened but still can't get 3D done, we pass current frame
			
			cout << "update status: " << update_status << endl;
		}
	}
	return 0;
}

void RGB_detect_Thread::rgb_detect()
{
	float downsamplesize = 0.005; //5mm
	bool gaussT = true;
	cout << "[INFO]start preprocess pcd thread" << " at TIMESTAMP " << now_str() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr downsampledxyz(new pcl::PointCloud<pcl::PointXYZ>);
	downsample_3D(pcd_read, downsampledxyz, downsamplesize, downsamplesize, downsamplesize);
	cout << "[INFO]downsample 3D" << " at TIMESTAMP " << now_str() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr truncate_xyz(new pcl::PointCloud<pcl::PointXYZ>);
	if (gaussT) {
		passthrough_xyz(downsampledxyz, truncate_xyz, 0.5, 4);
		cout << "[INFO]passthrough xyz" << " at TIMESTAMP " << now_str() << endl;
		gauss_3D(truncate_xyz, pcd_preprocess, 10, 1.5);
		cout << "[INFO]gauss 3D truncate " << " at TIMESTAMP " << now_str() << endl;
	}
	else {
		passthrough_xyz(downsampledxyz, pcd_preprocess, 0.5, 4);
		cout << "[INFO]passthrough xyz" << " at TIMESTAMP " << now_str() << endl;
	}
	cout << "preprocess pcd completed. has " << pcd_preprocess->size() << " valid points" << " at TIMESTAMP " << now_str() << endl;

}

void RGB_detect_Thread::run()
{
	QTimer rgb_detect_timer;
	bool res = connect(&rgb_detect_timer, SIGNAL(timeout()), this, SLOT(rgb_detect()), Qt::DirectConnection);
	int time_inter = 50;  // set timer to 1 second
	rgb_detect_timer.setTimerType(Qt::PreciseTimer);
	rgb_detect_timer.start(time_inter);
	exec();
	return;
};


int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	int cpu_nums = countCPU();
	//cpu_set_t cpuset[cpu_nums];//cpuset[i]是负责执行某个线程的若干个CPU核
	//pthread_t Thread[cpu_nums];//存放线程ID，ID由函数pthread_create分配
	cout << "cpu_nums" << cpu_nums << endl;
	int cpu_thread = std::thread::hardware_concurrency();
	cout << "cpu_thread" << cpu_thread << endl;

	//Check machine ID
	print_Guid();
	//std::string strGuid = GetGuid();
	//std::string SW_ID = "32534e31534e4741323233323633204a20202020202020202020202020202020315347443152465a";  // "FB047D51-BE99-42F5-9AD2-6B402B223CED"; //development_ID
	std::string SW_ID = "314a35373836304a303034342034202020202020"; //customer ID
																	//std::string SW_ID = "S3PNNF0JA00573M";   // Qi's ID
	int len = 128;
	char *lpszCpu;
	GetCpuByCmd(lpszCpu, len);
	cout << SW_ID << endl;
	cout << CPU_ID << endl;
	if (SW_ID == SW_ID)  //(SW_ID == CPU_ID)
	{
		cout << "Software Key matched and program is unlocked to run...." << endl;
		// Call Tray detection code
		TrayD TrayDetection;
		TrayDetection.TrayD_main();

	}
	else
	{
		cout << "Software Key is invalid and program stopped...." << endl;
	}

	return app.exec();
}

void TrayD::TrayD_main() // (int argc, char** argv)
{

	std::cout.flush();
	int test_return;

	// load calibration parameters from cali file
	QString fileName = "./dataSW/cali_model.txt";
	
	Load_cali_para(fileName);
	cout << "Load calibration model....." << endl;
	// Start CAN device
	StartDevice_CAN();
	// Set CAN timer to listen start msg
	cout << "New CAN Check Timer created" << endl;
	CAN_Check_thread.start();
	CAN_Check_thread.Check_CAN_msg();
	cout << "CAN Check Timer start" << endl;
	// Pack tray data to CAN format
	// input to choose 3D display

	cout << "Please enter '0' for No display or '1' for 3D point cloud display: " << endl;
	cin >> display_option;
	//display_option = 0;
	if (display_option ==1)
	{
		global_display_output_cloud = true;
		global_show_key_points_nearby_objects = true;
	}
	else
	{
		global_display_output_cloud = false;
		global_show_key_points_nearby_objects = false;
	}
	// Create CAN transmission thread to output tray data 
	input_option = 1;
	cout << "Force to run camera mode....." << endl;
	cout << "Please enter '1' for camera input or '2' for data file: " << endl;
	cin >> input_option;
	// classfy input selection
	if (input_option == 1)
	{
		camera_input = true;
	}
	if (input_option == 2)
	{
		camera_input = false;
	}
	if ((!input_option == 1) && (!input_option == 2)) // set default settings to data file
	{
		camera_input = false;
	}
	if (camera_input == true)
	{
		cout << "[INFO] Read Depth image from pico Zense........" << endl;
		pico_read();
	}
	else
	{
		cout << "[INFO] Read Depth image from file........" << endl;
	}
	if (hog_mode == true)
	{
		//hog_main();
		cout << "[WARNING] HOG discontinued. " << endl;
	}
	else   // run CNN main mode
	{
		// Run online  mode with camera input
		if (input_option == 1)
		{
			cam_test_mode = 2;  // fore to run CAN mode
			// Check if Start_test msg and start test
			if (cam_test_mode == 1)
			{
				cout << "Manually Set CAN msg to run camera test " << endl;
				receive_str_ID = "490";
				receive_str = "FFEE";
				int rev_ID = atoi(receive_str_ID.toStdString().c_str());
				cout << "Receive msg ID:" << rev_ID << "at TIMESTAMP " << now_str() << endl;
				// check after set start msg manually
				cout << "New CAN Send Timer created for manual mode" << endl;
				CAN_Send_thread.start();
				CAN_Send_thread.CAN_msg_send();
				cout << "New CAN Send Timer start" << endl;
			}
			if (cam_test_mode == 2)
			{
				cout << "Send result if receive start msg from CAN" << endl;
				cout << "New CAN Send Timer created for CAN mode" << endl;
				CAN_Send_thread.start();
				CAN_Send_thread.CAN_msg_send();
				cout << "New CAN Send Timer start" << endl;
			}
			else if ((cam_test_mode != 2) && (cam_test_mode != 1))
			{
				cout << "input is not correct for camera input test!" << endl;
			}
			// make a endless loop to iterate on check CAN msg and make measurement	
		}
		if (input_option == 2)
		{
			cout << "Run offline data file manually" << endl;
			//cam_test_mode = 2;  // fore to run CAN mode
			//cout << "Please enter '1' for manual test or '2' for CAN mode test: " << endl;
			//cin >> cam_test_mode;
			cam_test_mode = 2;
			// Check if Start_test msg and start test
			if (cam_test_mode == 1)
			{
				cout << "Manually Set CAN msg to run camera test " << endl;
				receive_str_ID = "490";
				receive_str = "FFEE";
				int rev_ID = atoi(receive_str_ID.toStdString().c_str());
				cout << "Receive msg ID:" << rev_ID << endl;
				// check after set start msg manually
				cout << "New CAN Send Timer created for manual mode" << endl;
				CAN_Send_thread.start();
				CAN_Send_thread.CAN_msg_send();
				cout << "New CAN Send Timer start" << endl;
			}
			if (cam_test_mode == 2)
			{
				cout << "Send result if receive start msg from CAN" << endl;
				cout << "New CAN Send Timer created for CAN mode" << endl;
				CAN_Send_thread.start();
				CAN_Send_thread.CAN_msg_send();
				cout << "New CAN Send Timer start" << endl;
			}
			else if ((cam_test_mode != 2) && (cam_test_mode != 1))
			{
				cout << "input is not correct for camera input test!" << endl;
			}
		}
	}
}
// This part is used to configure hardware IO interface

static void Opencv_Depth(uint32_t slope, int height, int width, uint8_t*pData, cv::Mat& dispImg)

{
	dispImg = cv::Mat(height, width, CV_16UC1, pData);
	Point2d pointxy(width / 2, height / 2);
	int val = dispImg.at<ushort>(pointxy);
	char text[20];
#ifdef _WIN32
	sprintf_s(text, "%d", val);
#else
	snprintf(text, sizeof(text), "%d", val);
#endif
	dispImg.convertTo(dispImg, CV_8U, 255.0 / slope);
	applyColorMap(dispImg, dispImg, cv::COLORMAP_RAINBOW);
	int color;
	if (val > 2500)
		color = 0;
	else
		color = 4096;
	circle(dispImg, pointxy, 4, Scalar(color, color, color), -1, 8, 0);
	putText(dispImg, text, pointxy, FONT_HERSHEY_DUPLEX, 2, Scalar(color, color, color));
}

void  Check_Device_ID(void *SN, void *FW, void *HW)
{
	std::string pico_SN = "00000000";
	std::string pico_FW = "00000000";
	std::string pico_HW = "00000000";
	bool ID_ok = false;
	if (pico_SN == SN_str)
	{
		if (pico_FW == FW_str)
		{
			if (pico_HW == HW_str)
				ID_ok = true;
		}
	}
}

int pico_read()   //(int argc, char *argv[])
{

	if (pico_init_done == false)  // do the initialization when pico_init=false
	{
		status = PsInitialize();
		if (status != PsReturnStatus::PsRetOK)
		{
			cout << "PsInitialize failed!" << endl;
			system("pause");
			pico_init_done = false;
			return -1;
		}
		else
		{
			cout << "Device Initialized done!" << endl;
		}

		status = PsGetDeviceCount(&deviceCount);
		if (status != PsReturnStatus::PsRetOK)
		{
			cout << "PsGetDeviceCount failed!" << endl;
			system("pause");
			pico_init_done = false;   // init failed
			return -1;
		}

		cout << "Get device count: " << deviceCount << endl;

		void* SN;
		void* FW;
		void* HW;
		std::string* SN_str[32];
		std::string* FW_str[32];
		std::string* HW_str[32];
		int32_t SNSize;
		int32_t FWSize;
		int32_t HWSize;
		//PsPropertyType PsPropertySN ;
		status = PsGetProperty(deviceIndex, PsPropertySN_Str, SN, &SNSize);
		status = PsGetProperty(deviceIndex, PsPropertyFWVer_Str, FW, &FWSize);
		status = PsGetProperty(deviceIndex, PsPropertyHWVer_Str, HW, &HWSize);
		SN_str[0] = (std::string *)SN;
		FW_str[0] = (std::string *)FW;
		HW_str[0] = (std::string *)HW;

		// Check device ID
		Check_Device_ID(&SN, &FW, &HW);

		status = PsOpenDevice(deviceIndex);
		if (status != PsReturnStatus::PsRetOK)
		{
			cout << "OpenDevice failed!" << endl;
			system("pause");
			return -1;
		}
		else {
			cout << "Open Device done!" << endl;
		}
		//Set the Depth Range to Near through PsSetDepthRange interface
		//status = PsSetDepthRange(deviceIndex, PsMidRange); //MidRange is 1m to 2m. 
		status = PsSetDepthRange(deviceIndex, PsFarRange); //FarRange is 1m to 8m. 
		if (status != PsReturnStatus::PsRetOK)
			cout << "PsSetDepthRange failed!" << endl;
		else
			cout << "Set Depth Range to far" << endl;
		//Enable the Depth and RGB synchronize feature
		//PsSetSynchronizeEnabled(deviceIndex, true);
		//Set PixelFormat as PsPixelFormatBGR888 for opencv display
		PsSetColorPixelFormat(deviceIndex, PsPixelFormatBGR888);
		//Set to DepthAndRGB_30 mode  // DepthAndIRAndRGB_30
		status = PsSetDataMode(deviceIndex, (PsDataMode)dataMode);

		if (status != PsReturnStatus::PsRetOK)
		{
			cout << "Set DataMode Failed failed!" << endl;
		}

		status1 = PsGetCameraParameters(deviceIndex, PsDepthSensor, &cameraParameters1);
		status2 = PsGetCameraParameters(deviceIndex, PsRgbSensor, &cameraParameters0);
		status3 = PsGetCameraExtrinsicParameters(deviceIndex, &CameraExtrinsicParameters);

		ofstream PointCloudWriter;
		PsDepthVector3 DepthVector = { 0, 0, 0 };
		PsVector3f WorldVector = { 0.0f };
		// config flag 
		bool f_bDistortionCorrection = true;
		bool f_bFilter = false;
		bool f_bMappedRGB = true;
		bool f_bMappedIR = false;
		bool f_bMappedDepth = false;
		bool f_bWDRMode = false;
		bool f_bInvalidDepth2Zero = false;
		bool f_bDustFilter = false;
		bool f_bSync = true;

		// Configuration area to set camera
		// Set undistortion
		PsSetDepthDistortionCorrectionEnabled(deviceIndex, f_bDistortionCorrection);
		PsSetIrDistortionCorrectionEnabled(deviceIndex, f_bDistortionCorrection);
		PsSetRGBDistortionCorrectionEnabled(deviceIndex, f_bDistortionCorrection);
		cout << "Set DistortionCorrection " << (f_bDistortionCorrection ? "Enabled." : "Disabled.") << endl;

		// Set Filter
		status = PsSetFilter(deviceIndex, PsSmoothingFilter, f_bFilter);
		// Set RGB to IR mapping
		//status = PsSetMapperEnabledRGBToIR(deviceIndex, f_bMappedIR);
		// Set RGB to depth mapping 
		status = PsSetMapperEnabledRGBToDepth(deviceIndex, f_bMappedDepth);
		// Set depth to RGB mapping
		status = PsSetMapperEnabledDepthToRGB(deviceIndex, f_bMappedRGB);

		// Set resolution
		PsFrameMode frameMode;
		frameMode.fps = 20;
		frameMode.pixelFormat = PsPixelFormatDepthMM16; //PsPixelFormatBGR888;
		frameMode.resolutionWidth = 640;
		frameMode.resolutionHeight = 480;
		cout << "Set RGB width:" << frameMode.resolutionWidth << " height:" << frameMode.resolutionHeight << endl;
		PsSetFrameMode(deviceIndex, PsRGBFrame, &frameMode);

		//Set background filter threshold
		static uint16_t threshold = 20;
		PsSetThreshold(deviceIndex, threshold);
		cout << "Set background threshold value: " << threshold << endl;

		// Set WDR mode
		if (f_bWDRMode)
		{
			static bool bWDRStyle = true;
			status = PsSetWDRStyle(deviceIndex, bWDRStyle ? PsWDR_ALTERNATION : PsWDR_FUSION);
			if (PsRetOK == status)
			{
				cout << "WDR image output " << (bWDRStyle ? "alternatively in multi range." : "Fusion.") << endl;
				bWDRStyle = !bWDRStyle;
			}
		}

		// Update global cx, cy, fx, fy for calculation across program
		if (Read_online_para == 1)
		{
			camera_cal_cx = cameraParameters1.cx;
			camera_cal_cy = cameraParameters1.cy;
			camera_cal_fx = cameraParameters1.fx;
			camera_cal_fy = cameraParameters1.fy;
			cout << "Read online camera parameters......" << endl;
		}
		else
		{
			cout << "Read offline camera parameters......" << endl;
		}
		// Save params
		std::string ConfigFilename = "./Data_file/Config_parameters.txt";
		ofstream fout(ConfigFilename);
		if (!fout) {
			cout << "File Not Opened" << endl;
			return -1;
		}

		fout << "PsGetCameraParameters status: " << status << endl;
		fout << "Depth Camera Intinsic: " << endl;
		fout << "Fx: " << cameraParameters1.fx << endl;
		fout << "Cx: " << cameraParameters1.cx << endl;
		fout << "Fy: " << cameraParameters1.fy << endl;
		fout << "Cy: " << cameraParameters1.cy << endl;
		fout << "Depth Distortion Coefficient: " << endl;
		fout << "K1: " << cameraParameters1.k1 << endl;
		fout << "K2: " << cameraParameters1.k2 << endl;
		fout << "P1: " << cameraParameters1.p1 << endl;
		fout << "P2: " << cameraParameters1.p2 << endl;
		fout << "K3: " << cameraParameters1.k3 << endl;
		fout << "K4: " << cameraParameters1.k4 << endl;
		fout << "K5: " << cameraParameters1.k5 << endl;
		fout << "K6: " << cameraParameters1.k6 << endl;

		fout << "PsGetCameraParameters status: " << status << endl;
		fout << "RGB Camera Intinsic: " << endl;
		fout << "Fx: " << cameraParameters0.fx << endl;
		fout << "Cx: " << cameraParameters0.cx << endl;
		fout << "Fy: " << cameraParameters0.fy << endl;
		fout << "Cy: " << cameraParameters0.cy << endl;
		fout << "RGB Distortion Coefficient: " << endl;
		fout << "K1: " << cameraParameters0.k1 << endl;
		fout << "K2: " << cameraParameters0.k2 << endl;
		fout << "K3: " << cameraParameters0.k3 << endl;
		fout << "P1: " << cameraParameters0.p1 << endl;
		fout << "P2: " << cameraParameters0.p2 << endl;

		fout << "PsGetCameraExtrinsicParameters status: " << status << endl;
		fout << "Camera rotation: " << endl;
		fout << CameraExtrinsicParameters.rotation[0] << " "
			<< CameraExtrinsicParameters.rotation[1] << " "
			<< CameraExtrinsicParameters.rotation[2] << " "
			<< CameraExtrinsicParameters.rotation[3] << " "
			<< CameraExtrinsicParameters.rotation[4] << " "
			<< CameraExtrinsicParameters.rotation[5] << " "
			<< CameraExtrinsicParameters.rotation[6] << " "
			<< CameraExtrinsicParameters.rotation[7] << " "
			<< CameraExtrinsicParameters.rotation[8] << " "
			<< endl;
		fout << "Camera transfer: " << endl;
		fout << CameraExtrinsicParameters.translation[0] << " "
			<< CameraExtrinsicParameters.translation[1] << " "
			<< CameraExtrinsicParameters.translation[2] << " " << endl;
		fout.close();
		cout << "Save config parameters in the file" << endl;
		pico_init_done = true;
	} // loop to initalize pico device

	cout << "Read Pico data frame at " << "TIMESTAMP " << now_str() << endl;
	ofstream PointCloudWriter;
	PsDepthVector3 DepthVector = { 0, 0, 0 };
	PsVector3f WorldVector = { 0.0f };
	bool f_bWDRMode = false;
	{
		//iter_index++;
		iter_index = 1;
		// Read one frame before call PsGetFrame
		status = PsReadNextFrame(deviceIndex);
		cout << "Read frame status: " << status << endl;
		if (status == PsRetOK)
		{
			cout << "Read Pico data frame successfully! " << "TIMESTAMP " << now_str() << endl;

		}
		//Get depth frame, depth frame only output in following data mode
		if (dataMode == PsDepthAndRGB_30 || dataMode == PsDepthAndIR_30 || dataMode == PsDepthAndIRAndRGB_30 || dataMode == PsDepthAndIR_15_RGB_30)
		{
			PsGetFrame(deviceIndex, PsDepthFrame, &depthFrame);
			cout << "depth frame type: " << depthFrame.frameType << endl;
			cout << "depth frame width: " << depthFrame.width << endl;
			if (depthFrame.pFrameData != NULL)
			{
				//Display the Depth Image
				Opencv_Depth(slope, depthFrame.height, depthFrame.width, depthFrame.pFrameData, RGBimageMat);
				depthimageMat = cv::Mat(depthFrame.height, depthFrame.width, CV_8UC3, depthFrame.pFrameData);
				//cv::imshow(depthImageWindow, RGBimageMat);
				//ofstream fout(depthFilename);
				//Read the depthFrameData in uint_16
				PsDepthPixel * DepthFrameData = (PsDepthPixel *)depthFrame.pFrameData;
				int nr = RGBimageMat.rows; // number of rows 
				int nc = RGBimageMat.cols; // number of columns 
				if (save_data_file == true)
				{
					//Index image name
					std::string s1, s2, s3, depthFilename;
					s1 = "./Data_file/DepthData";
					s2 = std::to_string(iter_index);
					s3 = ".txt";
					depthFilename = s1 + s2 + s3;
					ofstream fout(depthFilename);
					for (int i = 0; i < nr; i++)
					{
						for (int j = 0; j < nc; j++) {
							//fout<< imageMat.at<ushort>(i, j)<<"\t";
							fout << DepthFrameData[i*nc + j] << "\t";
						}
						fout << endl;
					}
					fout.close();
				}
				//upate global camera_depthmat_480_640 for use in later program. 
				if (nr != 480 || nc != 640)
				{
					cout << "[FATAl ERROR] camera setting is not 480x640. Force to exit " << endl;
					return -1;
				}
				for (int y = 0; y < nr; y++) // loop for rows
				{
					for (int x = 0; x < nc; x++)  // loop for columns
					{
						float current_pixel_depth_mm = static_cast<float> (DepthFrameData[y*nc + x]);
						camera_depthmat_480_640[y][x] = current_pixel_depth_mm;
					}
				}
				cout << "[INFO] camera_depthmat_480_640 created" << " at TIMESTAMP " << now_str() << endl;
			}
		}

		//Get RGB frame, RGB frame only output in following data mode
		if (dataMode == PsDepthAndRGB_30 || dataMode == PsIRAndRGB_30 || dataMode == PsDepthAndIRAndRGB_30 || dataMode == PsWDR_Depth || dataMode == PsDepthAndIR_15_RGB_30)
		{
			PsGetFrame(deviceIndex, PsRGBFrame, &rgbFrame);
			if (rgbFrame.pFrameData != NULL)
			{
				//Display the RGB Image
				RGBimageMat = cv::Mat(rgbFrame.height, rgbFrame.width, CV_8UC3, rgbFrame.pFrameData);
				//cv::imshow(rgbImageWindow, RGBimageMat);
				//Index image name
				if (save_data_file == true)
				{
					std::string s1, s2, s3, RGBilename;
					s1 = "./Data_file/RGBimage";
					s2 = std::to_string(iter_index);
					s3 = ".jpg";
					RGBilename = s1 + s2 + s3;
					imwrite(RGBilename, RGBimageMat);
				}
			}
		}
		//Get WDR depth frame(fusion or alternatively, determined by PsSetWDRStyle, default in fusion)
		//WDR depth frame only output in PsWDR_Depth data mode
		if (dataMode == PsWDR_Depth)
		{
			PsGetFrame(deviceIndex, PsWDRDepthFrame, &wdrDepthFrame);
			if (wdrDepthFrame.pFrameData != NULL)
			{
				//Display the WDR Depth Image
				Opencv_Depth(wdrSlope, wdrDepthFrame.height, wdrDepthFrame.width, wdrDepthFrame.pFrameData, RGBimageMat);
				//cv::imshow(wdrDepthImageWindow, RGBimageMat);
			}
		}
		//Get mapped rgb frame which is mapped to depth camera space
		//Mapped rgb frame only output in following data mode
		//And can only get when the feature is enabled through api PsSetMapperEnabledDepthToRGB
		//When the key "Q/q" pressed, this feature enable or disable
		if (dataMode == PsDepthAndRGB_30 || dataMode == PsDepthAndIRAndRGB_30 || dataMode == PsWDR_Depth || dataMode == PsDepthAndIR_15_RGB_30)
		{
			PsGetFrame(deviceIndex, PsMappedRGBFrame, &mappedRGBFrame);
			//cout << mappedRGBFrame.pFrameData << endl;
			if (mappedRGBFrame.pFrameData != NULL)
			{
				//Display the MappedRGB Image
				//cv::Mat imageMat;
				mappedRGBimageMat = cv::Mat(mappedRGBFrame.height, mappedRGBFrame.width, CV_8UC3, mappedRGBFrame.pFrameData);
				//cv::imshow(mappedRgbImageWindow, mappedRGBimageMat);
				if (save_data_file == true)
				{
					//Index image name
					std::string s1, s2, s3, RGBilename;
					s1 = "./Data_file/mappedRGB";
					s2 = std::to_string(iter_index);
					s3 = ".jpg";
					RGBilename = s1 + s2 + s3;
					imwrite(RGBilename, mappedRGBimageMat);
				}
				//update mapped rgb image here for use in later program
				cout << "[INFO] Copy mapped RGB at TIMESTAMP " << now_str() << endl;
				camera_mappedRGBMat = mappedRGBimageMat.clone();
				cout << "camera_mappedRGBMat updated has " << camera_mappedRGBMat.size[1] << "columns" << " at TIMESTAMP " << now_str() << endl;
				cout << "camera_mappedRGBMat updated has " << camera_mappedRGBMat.size[0] << "rows" << " at TIMESTAMP " << now_str() << endl;
			}
		}



	}
	return 0;
}
