DATASET_PATH=/cfs/wangboyuan/dataset/indoor/office

colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images \
   --ImageReader.single_camera 1 \
   --ImageReader.camera_model PINHOLE \
   --SiftExtraction.use_gpu 1


colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db \
   --SiftMatching.use_gpu 1

mkdir $DATASET_PATH/sparse

colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

mkdir $DATASET_PATH/dense

colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/1 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
   #  --max_image_size 2000