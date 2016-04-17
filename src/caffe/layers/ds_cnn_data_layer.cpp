#include <stdint.h>
//#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/ds_cnn_data_layer.hpp"

using std::string;
using std::map;
using std::pair;
using std::min;

#define ALPHA_CHANNEL 1

double round(double r);

namespace caffe {

template <typename Dtype>
void Com2DualSourceDataLayer<Dtype>::MirrorJointLabels(vector<Dtype>& labels) {
		vector<Dtype> tmp = labels;
		CHECK_EQ(num_labels_, 15);
		for(int j=0; j<num_labels_; j++) {
			labels[mirror_joints_map_[j]]=tmp[j];
		}
	}

template <typename Dtype>
void* Com2DualSourceDataLayerPrefetch(void* layer_pointer) {
	Com2DualSourceDataLayer<Dtype>* layer =  reinterpret_cast<Com2DualSourceDataLayer<Dtype>*>(layer_pointer);

	Dtype* top_local_data = layer->prefetch_local_data_->mutable_cpu_data(); 
	Dtype* top_global_data = layer->prefetch_global_data_->mutable_cpu_data(); 
  	Dtype* top_local_label = layer->prefetch_local_label_->mutable_cpu_data();
	Dtype* top_2d_pos = layer->prefetch_2d_pos_->mutable_cpu_data();
	Dtype* top_local_window = layer->prefetch_local_window_->mutable_cpu_data();
	Dtype* top_pose_rec =  layer->prefetch_pose_rec_->mutable_cpu_data();

	const Dtype scale = layer->layer_param_.pose_window_data_param().scale();
	const bool test_one_image = layer->layer_param_.pose_window_data_param().test_one_image();
  	const int batch_size = (layer->layer_param_.pose_window_data_param().batch_size());
  	const int crop_size = layer->layer_param_.pose_window_data_param().crop_size();
  	const int context_pad = layer->layer_param_.pose_window_data_param().context_pad();
  	const bool mirror = layer->layer_param_.pose_window_data_param().mirror();
	const float fg_fraction = layer->layer_param_.pose_window_data_param().fg_fraction();
	const int num_fg = static_cast<int>(static_cast<float>(batch_size)*fg_fraction);
	const Dtype* mean = layer->data_mean_.cpu_data();
	const int mean_off = (layer->data_mean_.width() - crop_size) / 2;
  	const int mean_width = layer->data_mean_.width();
  	const int mean_height = layer->data_mean_.height();
  	cv::Size cv_crop_size(crop_size, crop_size);
	cv::Size cv_global_crop_size(crop_size, crop_size);
  	const string& crop_mode = layer->layer_param_.pose_window_data_param().crop_mode();
	bool use_square = (crop_mode == "square") ? true : false;
	const float area_upper_threshold = layer->layer_param_.pose_window_data_param().area_upper_threshold();
	const float area_threshold = layer->layer_param_.pose_window_data_param().area_threshold();

	// zero out batch
	memset(top_local_data, 0, sizeof(Dtype)*layer->prefetch_local_data_->count());
	memset(top_global_data, 0, sizeof(Dtype)*layer->prefetch_global_data_->count());

	int num_samples[2];
	num_samples[0]=(batch_size - num_fg);
	num_samples[1]=num_fg;
	if(test_one_image==true){
		CHECK_EQ(num_samples[0], 0); // all foreground
		CHECK_EQ(mirror, false);
		printf("fg: %d/%d\n", (layer->fg_cnt_total_ % layer->fg_windows_.size()), layer->fg_windows_.size());
	}

	CHECK_EQ(layer->fg_labels_.size(), layer->fg_windows_.size());
	CHECK_EQ(layer->bg_labels_.size(), layer->bg_windows_.size());

	int item_id = 0;
	vector<Dtype> img_size;

	for (int is_fg = 0; is_fg < 2; ++is_fg) {
		for (int dummy = 0; dummy < num_samples[is_fg]; ++dummy) {
			// sample a window
			unsigned int rand_index = layer->PrefetchRand();
			unsigned int rand_index_global = layer->PrefetchRand();
			vector<Dtype> window;
			vector<Dtype> global_window;

			if(test_one_image==true){
				CHECK_EQ(is_fg, 1);
				window = (is_fg) ?
          				layer->fg_windows_[layer->fg_cnt_total_ % layer->fg_windows_.size()] :
          				layer->bg_windows_[layer->bg_cnt_total_ % layer->bg_windows_.size()];
					// full-body window
					vector<vector<Dtype> >  global_windows = layer->full_body_database_[window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX]];
			}else{
				window = (is_fg) ?
	          				layer->fg_windows_[rand_index % layer->fg_windows_.size()] :
	          				layer->bg_windows_[rand_index % layer->bg_windows_.size()];

				//printf("(%d,%d)%f\t%f\t%f\t%f\n", is_fg, rand_index % layer->bg_windows_.size(), window[JointDetPose2dDualSourceDataLayer<Dtype>::X1], window[JointDetPose2dDualSourceDataLayer<Dtype>::Y1], window[JointDetPose2dDualSourceDataLayer<Dtype>::X2], window[JointDetPose2dDualSourceDataLayer<Dtype>::Y2]);

				// full-body window
				vector<vector<Dtype> >  global_windows = layer->full_body_database_[window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX]]; //
				if(is_fg==1){
					global_window = global_windows[rand_index % global_windows.size()];
				}else{
					global_window = global_windows[0];
				}
			}

			bool do_mirror = false;
			if (mirror && layer->PrefetchRand() % 2) {
        			do_mirror = true;
      			}

			// load the image containing the window
			pair<std::string, vector<Dtype> > image =
          			layer->image_database_[window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX]];

			img_size = image.second;
			cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
			if (!cv_img.data) {
        			LOG(ERROR) << "Could not open or find file " << image.first;
        			return reinterpret_cast<void*>(NULL);
 			}
			const int channels = cv_img.channels();
			const Dtype torso = img_size[3], torso_area=torso*torso;

			// crop window out of image and warp it
      			int x1 = window[Com2DualSourceDataLayer<Dtype>::X1];
      			int y1 = window[Com2DualSourceDataLayer<Dtype>::Y1];
      			int x2 = window[Com2DualSourceDataLayer<Dtype>::X2];
      			int y2 = window[Com2DualSourceDataLayer<Dtype>::Y2];
			int x1g = global_window[Com2DualSourceDataLayer<Dtype>::X1];
      			int y1g = global_window[Com2DualSourceDataLayer<Dtype>::Y1];
      			int x2g = global_window[Com2DualSourceDataLayer<Dtype>::X2];
      			int y2g = global_window[Com2DualSourceDataLayer<Dtype>::Y2];

			int pad_w = 0;
      			int pad_h = 0;
			int pad_w_g = 0;
      			int pad_h_g = 0;

			if(context_pad > 0 || use_square) {
								
				// scale factor by which to expand the original region
       				// such that after warping the expanded region to crop_size x crop_size
        			// there's exactly context_pad amount of padding on each side
				Dtype context_scale = static_cast<Dtype>(crop_size)/static_cast<Dtype>(crop_size - 2*context_pad);

				//--------------global-------------------------------------
				// Compute the expanded region
				Dtype half_height_g = static_cast<Dtype>(y2g-y1g+1)/2.0;
        			Dtype half_width_g = static_cast<Dtype>(x2g-x1g+1)/2.0;
        			Dtype center_x_g = static_cast<Dtype>(x1g) + half_width_g;
        			Dtype center_y_g = static_cast<Dtype>(y1g) + half_height_g;
				if(use_square){
					if (half_height_g > half_width_g) {
						half_width_g = half_height_g;
					}else{
						half_height_g = half_width_g;
					}
				}
				x1g = static_cast<int>(round(center_x_g - half_width_g*context_scale));
        			x2g = static_cast<int>(round(center_x_g + half_width_g*context_scale));
        			y1g = static_cast<int>(round(center_y_g - half_height_g*context_scale));
        			y2g = static_cast<int>(round(center_y_g + half_height_g*context_scale));

				int unclipped_height_g = y2g-y1g+1;
        			int unclipped_width_g = x2g-x1g+1;
        			int pad_x1_g = std::max(0, -x1g);
        			int pad_y1_g = std::max(0, -y1g);
        			int pad_x2_g = std::max(0, x2g - cv_img.cols + 1);
        			int pad_y2_g = std::max(0, y2g - cv_img.rows + 1);

				// clip bounds
				x1g = x1g + pad_x1_g;
        			x2g = x2g - pad_x2_g;
        			y1g = y1g + pad_y1_g;
        			y2g = y2g - pad_y2_g;
				CHECK_GT(x1g, -1);
        			CHECK_GT(y1g, -1);
        			CHECK_LT(x2g, cv_img.cols);
        			CHECK_LT(y2g, cv_img.rows);

				int clipped_height_g = y2g-y1g+1;
        			int clipped_width_g = x2g-x1g+1;
				

				//--------------local-------------------------------------
				// Compute the expanded region
				Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        			Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        			Dtype center_x = static_cast<Dtype>(x1) + half_width;
        			Dtype center_y = static_cast<Dtype>(y1) + half_height;
				if(use_square){
					if (half_height > half_width) {
						half_width = half_height;
					}else{
						half_height = half_width;
					}
				}
				x1 = static_cast<int>(round(center_x - half_width*context_scale));
        			x2 = static_cast<int>(round(center_x + half_width*context_scale));
        			y1 = static_cast<int>(round(center_y - half_height*context_scale));
        			y2 = static_cast<int>(round(center_y + half_height*context_scale));

				// the expanded region may go outside of the image
       				// so we compute the clipped (expanded) region and keep track of
        			// the extent beyond the image
        			int unclipped_height = y2-y1+1;
        			int unclipped_width = x2-x1+1;
        			int pad_x1 = std::max(x1, x1g) - x1;
        			int pad_x2 = x2 - std::min(x2, x2g);
        			int pad_y1 = std::max(y1, y1g) - y1;
        			int pad_y2 = y2 - std::min(y2, y2g);
				// clip bounds
				x1 = x1 + pad_x1;
        			x2 = x2 - pad_x2;
        			y1 = y1 + pad_y1;
        			y2 = y2 - pad_y2;
				CHECK_GT(x1, -1);
        			CHECK_GT(y1, -1);
        			CHECK_LT(x2, cv_img.cols);
        			CHECK_LT(y2, cv_img.rows);

				int clipped_height = y2-y1+1;
        			int clipped_width = x2-x1+1;

				//---------------Local scaling-----------------------
				// scale factors that would be used to warp the unclipped
        			// expanded region
        			Dtype scale_x =
            				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        			Dtype scale_y =
            				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

				// size to warp the clipped exanded region to
				cv_crop_size.width =
            				static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        			cv_crop_size.height =
            				static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
				pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        			pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        			pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        			pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

				pad_h = pad_y1; 
				// if we're mirroring, we mirror the padding too (to be pedantic)
				if (do_mirror) {
          				pad_w = pad_x2;
        			} else {
          				pad_w = pad_x1;
        			}

				// ensure that the warped, clipped region plus the padding fits in the
        			// crop_size x crop_size image (it might not due to rounding)
        			if (pad_h + cv_crop_size.height > crop_size) {
          				cv_crop_size.height = crop_size - pad_h;
        			}
        			if (pad_w + cv_crop_size.width > crop_size) {
          				cv_crop_size.width = crop_size - pad_w;
        			}

				//---------------Global scaling-----------------------
				Dtype scale_x_g =
            				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width_g);
        			Dtype scale_y_g =
            				static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height_g);

				// size to warp the clipped exanded region to
				cv_global_crop_size.width =
            				static_cast<int>(round(static_cast<Dtype>(clipped_width_g)*scale_x_g));
        			cv_global_crop_size.height =
            				static_cast<int>(round(static_cast<Dtype>(clipped_height_g)*scale_y_g));
				pad_x1_g= static_cast<int>(round(static_cast<Dtype>(pad_x1_g)*scale_x_g));
        			pad_x2_g= static_cast<int>(round(static_cast<Dtype>(pad_x2_g)*scale_x_g));
        			pad_y1_g= static_cast<int>(round(static_cast<Dtype>(pad_y1_g)*scale_y_g));
        			pad_y2_g= static_cast<int>(round(static_cast<Dtype>(pad_y2_g)*scale_y_g));

				pad_h_g = pad_y1_g;
				// if we're mirroring, we mirror the padding too (to be pedantic)
				if (do_mirror) {
          				pad_w_g = pad_x2_g;
        			} else {
          				pad_w_g = pad_x1_g;
        			}

				if (pad_h_g + cv_global_crop_size.height > crop_size) {
          				cv_global_crop_size.height = crop_size - pad_h_g;
        			}
        			if (pad_w_g + cv_global_crop_size.width > crop_size) {
          				cv_global_crop_size.width = crop_size - pad_w_g;
        			}
			}

			//-----------Transfer local region------------------
			cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      			cv::Mat cv_cropped_img = cv_img(roi);
      			cv::resize(cv_cropped_img, cv_cropped_img, cv_crop_size, 0, 0, cv::INTER_LINEAR);

			// horizontal flip at random
      			if (do_mirror) {
        			cv::flip(cv_cropped_img, cv_cropped_img, 1);
      			}

			// copy the warped window into top_data
			for (int c = 0; c < channels; ++c) {
        			for (int h = 0; h < cv_cropped_img.rows; ++h) {
          				for (int w = 0; w < cv_cropped_img.cols; ++w) {
            					Dtype pixel = static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

						top_local_data[((item_id * channels + c) * crop_size + h + pad_h) * crop_size + w + pad_w] 
							= (pixel - mean[(c * mean_height + h + mean_off + pad_h)* mean_width + w + mean_off + pad_w])*scale;
          				}
        			}
			}
			//-----------Transfer global region------------------
			cv::Rect groi(x1g, y1g, x2g-x1g+1, y2g-y1g+1);
      			cv::Mat cv_global_cropped_img = cv_img(groi);
			cv::resize(cv_global_cropped_img, cv_global_cropped_img, cv_global_crop_size, 0, 0, cv::INTER_LINEAR);

			// horizontal flip at random
			if(do_mirror) {
				cv::flip(cv_global_cropped_img, cv_global_cropped_img, 1);
			}

			// copy the warped window into top_data
			for (int c = 0; c < channels; ++c) {
        			for (int h = 0; h < cv_global_cropped_img.rows; ++h) {
          				for (int w = 0; w < cv_global_cropped_img.cols; ++w) {
						Dtype pixel = static_cast<Dtype>(cv_global_cropped_img.at<cv::Vec3b>(h, w)[c]);
#if ALPHA_CHANNEL==1
						top_global_data[((item_id * (channels+1) + c) * crop_size + h + pad_h_g) * crop_size + w + pad_w_g]
#else
						top_global_data[((item_id * (channels+0) + c) * crop_size + h + pad_h_g) * crop_size + w + pad_w_g]
#endif
							= (pixel - mean[(c * mean_height + h + mean_off + pad_h_g)* mean_width + w + mean_off + pad_w_g])*scale;
          				}
        			}
			}
			// Cal sub_image locations in global window
			Dtype g_scale_x = static_cast<Dtype>(cv_global_crop_size.width)/static_cast<Dtype>(x2g-x1g+1);
			Dtype g_scale_y = static_cast<Dtype>(cv_global_crop_size.height)/static_cast<Dtype>(y2g-y1g+1);
			Dtype x_off, y_off, sx1, sx2, sy1, sy2;
			int tmp_x;
			
			x_off = (x1 - x1g)*g_scale_x;  
			y_off = (y1 - y1g)*g_scale_y;
			if(do_mirror){
				x_off = cv_global_crop_size.width - 1- x_off;
			}
			x_off+=pad_w_g;
			y_off+=pad_h_g;
			//sx1, sy1: rounding to be valid
			sx1 = round((x_off<pad_w_g)?pad_w_g:(x_off>=(cv_global_crop_size.width+pad_w_g)?(cv_global_crop_size.width+pad_w_g-1):x_off));
			sy1 = round((y_off<pad_h_g)?pad_h_g:(y_off>=(cv_global_crop_size.height+pad_h_g)?(cv_global_crop_size.height+pad_h_g-1):y_off));

			x_off = (x2 - x1g)*g_scale_x;  
			y_off = (y2 - y1g)*g_scale_y;
			if(do_mirror){
				x_off = cv_global_crop_size.width - 1- x_off;
			}
			x_off+=pad_w_g;
			y_off+=pad_h_g;
			// sx2, sy2: rounding to be valid
			sx2 = round((x_off<pad_w_g)?pad_w_g:(x_off>=(cv_global_crop_size.width+pad_w_g)?(cv_global_crop_size.width+pad_w_g-1):x_off));
			sy2 = round((y_off<pad_h_g)?pad_h_g:(y_off>=(cv_global_crop_size.height+pad_h_g)?(cv_global_crop_size.height+pad_h_g-1):y_off));

			if(do_mirror){
				tmp_x = sx1;
				sx1 = sx2;
				sx2 = tmp_x;
			}

			// sx1, sy1, sx2, sy2 have taken the pad into account, so no pad here
#if ALPHA_CHANNEL==1
			for (int h = 0; h < crop_size; ++h) {
          			for (int w = 0; w < crop_size; ++w) {
					if(h>=sy1 && h<=sy2 && w>=sx1 && w<=sx2) {
						top_global_data[((item_id * (channels+1) + channels) * crop_size + h) * crop_size + w] = 127;
					}else{
						top_global_data[((item_id * (channels+1) + channels) * crop_size + h) * crop_size + w] = -128;
					}
          			}
			}
#endif

			/////////////////////////////////////
			// Record window info
			int win_str = Com2DualSourceDataLayer<Dtype>::NUM-2+2; // -OVERLAP-CLOSEST_JOINT+IMG_SIZE_W+IMG_SIZE_H
			top_local_window[item_id*win_str+0] =  window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX];
			top_local_window[item_id*win_str+1] =  img_size[1];
			top_local_window[item_id*win_str+2] =  img_size[2];
			top_local_window[item_id*win_str+3] =  window[Com2DualSourceDataLayer<Dtype>::WINDOW_INDEX];
			top_local_window[item_id*win_str+4] =  window[Com2DualSourceDataLayer<Dtype>::X1];
			top_local_window[item_id*win_str+5] = window[Com2DualSourceDataLayer<Dtype>::Y1];
			top_local_window[item_id*win_str+6] = window[Com2DualSourceDataLayer<Dtype>::X2];
			top_local_window[item_id*win_str+7] = window[Com2DualSourceDataLayer<Dtype>::Y2];
			// Pose estimation de-normailization
			Dtype rec_scale_x = static_cast<Dtype>(cv_crop_size.width)/static_cast<Dtype>(x2-x1+1);
			Dtype rec_scale_y = static_cast<Dtype>(cv_crop_size.height)/static_cast<Dtype>(y2-y1+1);
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::CROP_SIZE] = crop_size;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::PAD_X] = pad_w;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::PAD_Y] = pad_h;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::START_X] = x1;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::START_Y] = y1;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::SCALE_X] = rec_scale_x;
			top_pose_rec[item_id*Com2DualSourceDataLayer<Dtype>::WNUM+Com2DualSourceDataLayer<Dtype>::SCALE_Y] = rec_scale_y;

			// get window label
			Dtype label = window[Com2DualSourceDataLayer<Dtype>::CLOSEST_JOINT];
			//Get regression groundtruth
			Dtype gx_off, gy_off;
			int local_w=sx2-sx1+1, local_h=sy2-sy1+1;
			//default value
			top_local_label[item_id] = 0; // start from 1
			// 2d pos
			top_2d_pos[item_id*2+0] = -1;
			top_2d_pos[item_id*2+1] = -1;
			//
			if(label>0){
				map<int, std::vector<Dtype> > pose_2d = layer->image_joint_2d_pos_[window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX]];
				vector<Dtype> joint_2d = pose_2d[static_cast<int>(label)];
				if(joint_2d.size()>0){
					// get relative coordinates
					x_off = (joint_2d[0]-1 - x1)*rec_scale_x;  // Assume joint_pos starts from 1, should prove it!!! 
					y_off = (joint_2d[1]-1 - y1)*rec_scale_y;
					if(x_off>=0 && y_off>=0 && x_off<cv_crop_size.width && y_off<cv_crop_size.height)
					{
						if(do_mirror){
							x_off = cv_crop_size.width - 1- x_off;
						}
						x_off+=pad_w;
						y_off+=pad_h;
						top_local_label[item_id] = do_mirror ? (layer->mirror_joints_map_[static_cast<int>(label)-1]+1):(label); // start from 1
						// 2d pos
						top_2d_pos[item_id*2+0] = (x_off-(crop_size)/2.)/crop_size;
						top_2d_pos[item_id*2+1] = (y_off-(crop_size)/2.)/crop_size;
					}
				}
			}
			//Just for debug/////////////
			if(1){
				char filename[256];

				cv::Mat patch_local = cv::Mat(crop_size, crop_size, CV_8UC(3));
				cv::Mat patch_global = cv::Mat(crop_size, crop_size, CV_8UC4);
				////////// Local ///////////////////
				sprintf(filename, "input%03d_%03d_%03d_%d_local.png", item_id, (int)window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX], (int)window[Com2DualSourceDataLayer<Dtype>::WINDOW_INDEX], do_mirror);
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < cv_cropped_img.rows; ++h) { //cv_img.rows
          					for (int w = 0; w < cv_cropped_img.cols; ++w) { //cv_img.cols
							patch_local.at<cv::Vec3b>(h+pad_h, w+pad_w)[c] = (top_local_data[((item_id * channels + c) * crop_size + h + pad_h) * crop_size + w + pad_w]+
							mean[(c * mean_height + h + mean_off + pad_h)* mean_width + w + mean_off + pad_w]);
          					}
					}
				}
				printf("(%d, %d) mirror: %d, Valid joint: %f label=%.0f\n", item_id, is_fg, do_mirror, top_local_label[item_id], top_local_label[item_id]);
				
				cv::Point center = cv::Point(top_2d_pos[item_id*2+0]*crop_size+(crop_size/2), top_2d_pos[item_id*2+1]*crop_size+(crop_size/2));
				int radius = crop_size/32;
				if(top_local_label[item_id]>0){ 
					cv::circle(patch_local, center, radius, cv::Scalar(255, 0, 0), -1);
				}
				cv::imwrite(filename , patch_local);

				////////// Global ///////////////////
				sprintf(filename, "input%03d_%03d_%03d_%d_global.png", item_id, (int)global_window[Com2DualSourceDataLayer<Dtype>::IMAGE_INDEX], (int)window[Com2DualSourceDataLayer<Dtype>::WINDOW_INDEX], do_mirror);
				for (int c = 0; c < channels; ++c) {
					for (int h = 0; h < cv_global_cropped_img.rows; ++h) { //cv_img.rows
          					for (int w = 0; w < cv_global_cropped_img.cols; ++w) { //cv_img.cols
#if ALPHA_CHANNEL
							patch_global.at<cv::Vec4b>(h+pad_h_g, w+pad_w_g)[c] = (top_global_data[((item_id * (channels+1) + c) * crop_size + h + pad_h_g) * crop_size + w + pad_w_g]+
#else
							patch_global.at<cv::Vec4b>(h+pad_h_g, w+pad_w_g)[c] = (top_global_data[((item_id * (channels) + c) * crop_size + h + pad_h_g) * crop_size + w + pad_w_g]+
#endif
							mean[(c * mean_height + h + mean_off + pad_h_g)* mean_width + w + mean_off + pad_w_g]);
          					}
					}
				}
#if ALPHA_CHANNEL
				for (int h = 0; h < crop_size; ++h) { //cv_img.rows
          				for (int w = 0; w < crop_size; ++w) { //cv_img.cols
						int tmp= top_global_data[((item_id * (channels+1) + channels) * crop_size + h) * crop_size + w]+128;
						if(tmp<=0){
							patch_global.at<cv::Vec4b>(h, w)[channels]=128; // semi-transparent
						}else if(tmp>255){
							patch_global.at<cv::Vec4b>(h, w)[channels]=255; // 
						}else{
							patch_global.at<cv::Vec4b>(h, w)[channels]=tmp;
						}
          				}
				}
#endif
				cv::rectangle( patch_global, cv::Point( sx1, sy1 ), cv::Point( sx2, sy2), cv::Scalar( 0, 55, 255 ), +1, 4 );
				cv::imwrite(filename , patch_global);
			}
			item_id++;
			if(is_fg){
				layer->fg_cnt_total_++;
			}else{
				layer->bg_cnt_total_++;
			}
		}
	}
	
	layer->batch_cnt_++;
	return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
Com2DualSourceDataLayer<Dtype>::~Com2DualSourceDataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void Com2DualSourceDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {

	CHECK_EQ(bottom.size(), 0) << "Pose window data Layer takes no input blobs.";
  	CHECK_EQ(top->size(), BNUM) << "Pose window data Layer prodcues"<<BNUM<<  "blobs as output.";
	
	const bool test_one_image = this->layer_param_.pose_window_data_param().test_one_image();

	fg_cnt_total_=0;
	bg_cnt_total_=0;
	batch_cnt_=0;

	LOG(INFO) << "Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.pose_window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.pose_window_data_param().bg_threshold() << std::endl
      << "  foreground sampling fraction: "
      << this->layer_param_.pose_window_data_param().fg_fraction();

	string hashtag;
	int image_index, channels, image_cnt=0, batch_cnt_=0, full_body_cnt=0;
	Dtype  torso_area=0;
	num_joints_=NUM_JOINTS; // set a constant
	num_labels_=num_joints_+1; //joints+none
	mirror_joints_map_.push_back(5);
	mirror_joints_map_.push_back(4);
	mirror_joints_map_.push_back(3);
	mirror_joints_map_.push_back(2);
	mirror_joints_map_.push_back(1);
	mirror_joints_map_.push_back(0);
	mirror_joints_map_.push_back(11);
	mirror_joints_map_.push_back(10);
	mirror_joints_map_.push_back(9);
	mirror_joints_map_.push_back(8);
	mirror_joints_map_.push_back(7);
	mirror_joints_map_.push_back(6);
	mirror_joints_map_.push_back(12);
	mirror_joints_map_.push_back(13);
	mirror_joints_map_.push_back(14);

	map<int, int> joint_hist;
	map<int, int> closestjoint_hist;
	map<int, int> jointnum_hist;
	/*
	for(int j=0; i<=num_joint_+1;i++){
		joint_hist.insert(std::make_pair(i,0));
		closestjoint_hist.insert(std::make_pair(i,0));
	}
	*/
	int src_num = this->layer_param_.pose_window_data_param().source_size();
	for(int src_id=0; src_id<src_num; src_id++)
	{
		PoseSource src = this->layer_param_.pose_window_data_param().source(src_id);
		std::ifstream infile(src.address().c_str());
		
		CHECK(infile.good()) << "Failed to open window file "<< src.address().c_str() << std::endl;

		while (infile >> hashtag >> image_index) {
			CHECK_EQ(hashtag, "#");
			// Read image path
	    		string image_path;
	    		infile >> image_path;
			// read image dimensions
			vector<Dtype> image_size(4);
			infile >> image_size[0] >> image_size[1] >> image_size[2] >> image_size[3];
			channels = image_size[0];
			torso_area = image_size[3]*image_size[3];
			
			// Store img info
			image_database_.push_back(std::make_pair(image_path, image_size));

			// Read joint 2d positions
			int current_2d_joint_num, joint_2d_id;
			vector<Dtype> one_joint_2d_pos(2);
			map<int, vector<Dtype> > all_joints_2d_pos;
			infile >> current_2d_joint_num;
			for(int i=0; i<current_2d_joint_num; i++) {
				infile >> joint_2d_id >> one_joint_2d_pos[0] >> one_joint_2d_pos[1];
				all_joints_2d_pos[joint_2d_id] = one_joint_2d_pos;
			}
			image_joint_2d_pos_.push_back(all_joints_2d_pos);
			// Read joint 3d position
			int current_3d_joint_num;
			infile>>current_3d_joint_num;
			assert(current_3d_joint_num==0);

			const float fg_lower_threshold = this->layer_param_.pose_window_data_param().fg_threshold();
			const float bg_upper_threshold = this->layer_param_.pose_window_data_param().bg_threshold();
			const float area_lower_threshold = this->layer_param_.pose_window_data_param().area_threshold();
			const float fg_upper_threshold = this->layer_param_.pose_window_data_param().fg_upper_threshold();
			const float area_upper_threshold = this->layer_param_.pose_window_data_param().area_upper_threshold();

			 // full-body windows
			 vector<vector<Dtype> > all_full_body_windows;
			 vector<Dtype> one_full_body_window(Com2DualSourceDataLayer::NUM);
			 // The first full_body_patch is the whole image
			 one_full_body_window[Com2DualSourceDataLayer::IMAGE_INDEX] = image_cnt;
			 one_full_body_window[Com2DualSourceDataLayer::WINDOW_INDEX] = -1;
			 one_full_body_window[Com2DualSourceDataLayer::OVERLAP] = src.max_jointnum();
			 one_full_body_window[Com2DualSourceDataLayer::X1] = 0;
	      		 one_full_body_window[Com2DualSourceDataLayer::Y1] = 0;
			 one_full_body_window[Com2DualSourceDataLayer::X2] = image_size[2]-1;
	      		 one_full_body_window[Com2DualSourceDataLayer::Y2] = image_size[1]-1;
			 all_full_body_windows.push_back(one_full_body_window);

			
			// Read all boxes
			int num_windows, num_fg_windows=0, num_bg_windows=0;
			infile>>num_windows;
			 for(int i=0; i<num_windows; i++){
				int x1, y1, x2, y2, closest_joint;
				int overlap=0; // joint num in the window
				vector<Dtype> joint_flags(num_joints_);
				infile >> x1 >> y1 >> x2 >> y2>>closest_joint;
				for(int j=0; j<num_joints_; j++){
					infile >> joint_flags[j];
					overlap+=joint_flags[j];
				}

				vector<Dtype> window(Com2DualSourceDataLayer::NUM);
				window[Com2DualSourceDataLayer::IMAGE_INDEX] = image_cnt;
				window[Com2DualSourceDataLayer::WINDOW_INDEX] = i;
				window[Com2DualSourceDataLayer::OVERLAP] = overlap;
				window[Com2DualSourceDataLayer::X1] = x1;
	      			window[Com2DualSourceDataLayer::Y1] = y1;
	      			window[Com2DualSourceDataLayer::X2] = x2;
	      			window[Com2DualSourceDataLayer::Y2] = y2;
				window[Com2DualSourceDataLayer::CLOSEST_JOINT] = closest_joint;

				Dtype img_area =  image_size[1]* image_size[2];
				Dtype win_area = (x2-x1+1)*(y2-y1+1);
				Dtype ratio = win_area/img_area;
				vector<Dtype> win_labels(num_labels_, 0);
				win_labels[closest_joint]=1;

				// Add window to foreground list or background list
				// now closest_joint is always larger than 0
				if ((win_area>area_lower_threshold*torso_area)&&(win_area<=area_upper_threshold*torso_area)&&
					(src.type()==PoseSource_SourceType_FOREGROUND||src.type()==PoseSource_SourceType_BOTHGROUND)&&
					overlap >= fg_lower_threshold && overlap <= fg_upper_threshold) {
					fg_windows_.push_back(window);
					fg_labels_.push_back(win_labels);

					for(int j=0; j<num_joints_; j++) {
						joint_hist[j+1]+=joint_flags[j];
					}
					num_fg_windows++;
					jointnum_hist[overlap]++;
					closestjoint_hist[closest_joint]++;
				}else if ((win_area>area_lower_threshold*torso_area)&&(win_area<=area_upper_threshold*torso_area)&&
					(src.type()==PoseSource_SourceType_BACKGROUND||src.type()==PoseSource_SourceType_BOTHGROUND)&&
					overlap < bg_upper_threshold) {
					// background window
					window[Com2DualSourceDataLayer::OVERLAP] = 0;
					bg_windows_.push_back(window);
					bg_labels_.push_back(win_labels);
					joint_hist[0]++;
					num_bg_windows++;
					jointnum_hist[overlap]++;
					closestjoint_hist[0]++;
				}
				//Full body patch collection
				if(overlap>=min(int(src.max_jointnum()), current_2d_joint_num)){
					all_full_body_windows.push_back(window);
				}
			 }
			 image_windows_.push_back(std::make_pair(num_bg_windows, num_fg_windows));
			 full_body_database_.push_back(all_full_body_windows);

			 if (image_cnt % 100 == 0) {
	      			LOG(INFO)<<"src"<< src_id<<" max_jointnum:"<<src.max_jointnum()<< " num: " << image_index << " "
	          			<< image_path << " "
	          			<< image_size[0] << " "
	          			<< image_size[1] << " "
	          			<< image_size[2] << " "
	          			<< "windows to process: " << num_windows;
				printf("src=%d, image_index=%d\n", src_id, image_index);
	    		}
			image_cnt++;
			full_body_cnt+=all_full_body_windows.size();
		}
	}

	LOG(INFO) << "Number of images: " << image_cnt;
	LOG(INFO) << "Number of bg_windows: " << bg_windows_.size()<<std::endl; 
	LOG(INFO) << "Number of fg_windows: " << fg_windows_.size()<<std::endl;
	LOG(INFO) << "Number of fb_windows: " << full_body_cnt<<std::endl;

	for (map<int, int>::iterator it = joint_hist.begin(); it != joint_hist.end(); ++it) {
    		LOG(INFO) << "Joint " << it->first << " has " << closestjoint_hist[it->first]<< " samples";
  	}

	for (map<int, int>::iterator it = jointnum_hist.begin(); it != jointnum_hist.end(); ++it) {
    		LOG(INFO) << "Jointnum " << it->first << " has " << jointnum_hist[it->first]
              	<< " samples";
  	}

	LOG(INFO) << "Amount of context padding: " << this->layer_param_.pose_window_data_param().context_pad();
  	LOG(INFO) << "Crop mode: "<< this->layer_param_.pose_window_data_param().crop_mode();

	// Image
	int crop_size = this->layer_param_.pose_window_data_param().crop_size();
	CHECK_GT(crop_size, 0);

	int batch_size=this->layer_param_.pose_window_data_param().batch_size();;
	// Local patch
	(*top)[LOCAL_DATA]->Reshape(batch_size, channels, crop_size, crop_size);
	prefetch_local_data_.reset(new Blob<Dtype>(batch_size, channels, crop_size, crop_size));
	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
	<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
	<< (*top)[0]->width();

	// Global patch
#if ALPHA_CHANNEL==1
	(*top)[GLOBAL_DATA]->Reshape(batch_size, channels+1, crop_size, crop_size);
	prefetch_global_data_.reset(new Blob<Dtype>(batch_size, channels+1, crop_size, crop_size));
#else
	(*top)[GLOBAL_DATA]->Reshape(batch_size, channels+0, crop_size, crop_size);
	prefetch_global_data_.reset(new Blob<Dtype>(batch_size, channels+0, crop_size, crop_size));
#endif

	// Joint labels
	(*top)[LOCAL_JOINT_LABEL]->Reshape(batch_size, 1, 1, 1);
	prefetch_local_label_.reset(new Blob<Dtype>(batch_size, 1, 1, 1));

	// Joint 2d positions
	(*top)[POSE_2D]->Reshape(batch_size, 2, 1, 1);
	prefetch_2d_pos_.reset(new Blob<Dtype>(batch_size, 2, 1, 1));

	// Window parameter
	int win_stride = NUM-2+2; // -OVERLAP-CLOSEST_JOINT+IMG_SIZE_W+IMG_SIZE_H
	(*top)[LOCAL_WINDOW]->Reshape(batch_size, win_stride, 1, 1);
	prefetch_local_window_.reset(new Blob<Dtype>(batch_size, win_stride, 1, 1));

	//Parameters for joint position re-normalization
	(*top)[POSE_RESTORE]->Reshape(batch_size, WNUM, 1, 1);
	prefetch_pose_rec_.reset(new Blob<Dtype>(batch_size, WNUM, 1, 1));

	// check if we want to have mean
	 if (this->layer_param_.pose_window_data_param().has_mean_file()) {
		const string& mean_file = this->layer_param_.pose_window_data_param().mean_file();
		LOG(INFO) << "Loading mean file from" << mean_file;
		BlobProto blob_proto;
		ReadProtoFromBinaryFileOrDie(mean_file, &blob_proto);
		data_mean_.FromProto(blob_proto);
    		CHECK_EQ(data_mean_.num(), 1);
    		CHECK_EQ(data_mean_.width(), data_mean_.height());
    		CHECK_EQ(data_mean_.channels(), channels);
	 } else {
		// Simply initialize an all-empty mean.
    		data_mean_.Reshape(1, channels, crop_size, crop_size);
		for(int i=0; i<channels*crop_size*crop_size; i++){
			data_mean_.mutable_cpu_data()[i] = 128;
		}
	 }

	// We do the same thing as caffe
	// Now, start the prefetch thread. Before calling prefetch, we make two
  	// cpu_data calls so that the prefetch thread does not accidentally make
  	// simultaneous cudaMalloc calls when the main thread is running. In some
  	// GPUs this seems to cause failures if we do not so.

	prefetch_local_data_->mutable_cpu_data();
	prefetch_global_data_->mutable_cpu_data();
  	prefetch_local_label_->mutable_cpu_data();
	prefetch_2d_pos_->mutable_cpu_data();
	prefetch_local_window_->mutable_cpu_data();
	prefetch_pose_rec_->mutable_cpu_data();

	data_mean_.cpu_data();
  	DLOG(INFO) << "Initializing prefetch";
  	CreatePrefetchThread();
  	DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void Com2DualSourceDataLayer<Dtype>::CreatePrefetchThread() {
	const bool prefetch_needs_rand =
      		this->layer_param_.pose_window_data_param().mirror() ||
      		this->layer_param_.pose_window_data_param().crop_size();
	if (prefetch_needs_rand) {
    		const unsigned int prefetch_rng_seed = caffe_rng_rand();
    		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  	} else {
    		prefetch_rng_.reset();
  	}
#ifdef LINUX  
	  // Create the thread.
	  CHECK(!pthread_create(&thread_, NULL, Com2DualSourceDataLayerPrefetch<Dtype>,
				static_cast<void*>(this))) << "Pthread execution failed.";
#else
	  thread_ = thread(Com2DualSourceDataLayerPrefetch<Dtype>,reinterpret_cast<void*>(this));
#endif

}

template <typename Dtype>
void Com2DualSourceDataLayer<Dtype>::JoinPrefetchThread() {
#ifdef LINUX
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
#else
  thread_.join();
#endif
}

template <typename Dtype>
unsigned int Com2DualSourceDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype Com2DualSourceDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();

  
  // Copy the data
  caffe_copy(prefetch_local_data_->count(), prefetch_local_data_->cpu_data(),
             (*top)[LOCAL_DATA]->mutable_cpu_data());
  caffe_copy(prefetch_global_data_->count(), prefetch_global_data_->cpu_data(),
             (*top)[GLOBAL_DATA]->mutable_cpu_data());
  caffe_copy(prefetch_local_label_->count(), prefetch_local_label_->cpu_data(),
             (*top)[LOCAL_JOINT_LABEL]->mutable_cpu_data());
  caffe_copy(prefetch_2d_pos_->count(), prefetch_2d_pos_->cpu_data(),
             (*top)[POSE_2D]->mutable_cpu_data());
  caffe_copy(prefetch_local_window_->count(), prefetch_local_window_->cpu_data(),
  		(*top)[LOCAL_WINDOW]->mutable_cpu_data());
  caffe_copy(prefetch_pose_rec_->count(), prefetch_pose_rec_->cpu_data(),
  		(*top)[POSE_RESTORE]->mutable_cpu_data());


  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}


INSTANTIATE_CLASS(Com2DualSourceDataLayer);

}

