

template <typename Dtype>
	void* Com2DualSourceDataLayerPrefetch(void* layer_pointer);
	template <typename Dtype>
	class Com2DualSourceDataLayer : public Layer<Dtype> 
	{
		// The function used to perform prefetching.
  		friend void* Com2DualSourceDataLayerPrefetch<Dtype>(void* layer_pointer);
	 public:
	 	explicit Com2DualSourceDataLayer(const LayerParameter& param)
      			: Layer<Dtype>(param) {}
		virtual ~Com2DualSourceDataLayer();
		virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      			vector<Blob<Dtype>*>* top);
	protected:
		virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      			const bool propagate_down, vector<Blob<Dtype>*>* bottom){ return; };

		virtual void CreatePrefetchThread();
  		virtual void JoinPrefetchThread();
  		virtual unsigned int PrefetchRand();

		virtual void MirrorJointLabels(vector<Dtype>& labels);

		shared_ptr<Caffe::RNG> prefetch_rng_;
#ifdef LINUX
				pthread_t thread_;
#else //win
				std::thread thread_;
#endif
		int num_joints_;
		int num_labels_; // num_joints_+1
		shared_ptr<Blob<Dtype> > prefetch_local_data_;
		shared_ptr<Blob<Dtype> > prefetch_global_data_;
		shared_ptr<Blob<Dtype> > prefetch_local_label_;
		shared_ptr<Blob<Dtype> > prefetch_2d_pos_;
		shared_ptr<Blob<Dtype> > prefetch_local_window_;
		shared_ptr<Blob<Dtype> > prefetch_pose_rec_;

		Blob<Dtype> data_mean_;
  		vector<std::pair<std::string, vector<Dtype> > > image_database_; // <path, image_size>
		vector<vector<vector<Dtype> > > full_body_database_;

		enum Com2DualSourceDataLayerBottom {
			LOCAL_DATA,
			GLOBAL_DATA,
			LOCAL_JOINT_LABEL,
			POSE_2D,
			LOCAL_WINDOW,
			POSE_RESTORE,
			BNUM
		};

		enum Com2DualSourceField { 
			IMAGE_INDEX, WINDOW_INDEX,
			OVERLAP, X1, Y1, X2, Y2, CLOSEST_JOINT,
			//JOINT_LABEL,
			//NUM=JOINT_LABEL+this.num_joints_ 
			NUM
		};

		enum Com2PoseReconstructionField { 
			CROP_SIZE, PAD_X, PAD_Y, START_X, START_Y, SCALE_X, SCALE_Y,
			WNUM
		};

		vector<vector<Dtype> > fg_windows_;
  		vector<vector<Dtype> > bg_windows_;
		vector<vector<Dtype> > fg_labels_;
  		vector<vector<Dtype> > bg_labels_;

		vector<int> mirror_joints_map_;
		vector<std::pair<int, int > > image_windows_;

		 int fg_cnt_total_;
		 int bg_cnt_total_;
		 int batch_cnt_;

		 vector<std::map<int, std::vector<Dtype> > > image_joint_2d_pos_;
	};