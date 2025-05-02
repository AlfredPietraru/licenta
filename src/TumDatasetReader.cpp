#include "../include/TumDatasetReader.h"

std::vector<Position_Entry> read_groundtruth_file(const std::string &file_path) {
    std::vector<Position_Entry> positions;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return positions;
    }

    std::string line;


    while (std::getline(file, line)) {
        // Skip comment lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        Position_Entry entry;

        // Read values from the line
        ss >> entry.timestamp >> entry.tx >> entry.ty >> entry.tz 
           >> entry.qx >> entry.qy >> entry.qz >> entry.qw;
        entry.pose = Sophus::SE3d(Eigen::Quaterniond(entry.qw, entry.qx, entry.qy, entry.qz), Eigen::Vector3d(entry.tx, entry.ty, entry.tz));
        positions.push_back(entry);
    }

    file.close();
    return positions;
}

// void TumDatasetReader::read_mapping_rgb_depth_file() {}
std::string extract_timestamp(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of('/');
    size_t dot = filepath.find_last_of('.');
    return filepath.substr(last_slash + 1, dot - last_slash - 1);
}

double extract_timestamp_double(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of('/');
    size_t dot = filepath.find_last_of('.');
    std::string timestamp_string = filepath.substr(last_slash + 1, dot - last_slash - 1);
    return std::stod(timestamp_string);
}


std::vector<std::pair<std::string, std::string>> get_mapping_between_rgb_depth(std::string file_path) {
    std::vector<std::pair<std::string, std::string>> myMap;
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string first, second;
        if (ss >> first >> second) {
            myMap.push_back({first, second});
        }
    }
    file.close();
    return myMap;
} 

std::unordered_map<std::string, Position_Entry> get_mapping_between_rgb_position(std::string file_path) {
    std::unordered_map<std::string, Position_Entry> out_map;
    std::ifstream file(file_path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        Position_Entry entry; 
        std::string rgb_timestamp, position_entry_timestamp;
        ss >> rgb_timestamp >> position_entry_timestamp;
        ss >> entry.tx >> entry.ty >> entry.tz >> entry.qx >> entry.qy >> entry.qz >> entry.qw;
        entry.timestamp = extract_timestamp_double(position_entry_timestamp);
        entry.pose = Sophus::SE3d(Eigen::Quaterniond(entry.qw, entry.qx, entry.qy, entry.qz), Eigen::Vector3d(entry.tx, entry.ty, entry.tz));
        out_map.insert({rgb_timestamp, entry});
    }
    return out_map;
}

TumDatasetReader::TumDatasetReader(Config cfg) {
    this->cfg = cfg;
    std::string path_to_write = "../rgbd_dataset_freiburg1_xyz/estimated.txt";
    this->outfile = std::ofstream(path_to_write);
    std::string rgb_path = "../rgbd_dataset_freiburg1_xyz/rgb";
    std::string depth_path = "../rgbd_dataset_freiburg1_xyz/depth";
    std::unordered_map<std::string, std::string> map_rgb_file_name_path;
    std::unordered_map<std::string, std::string> map_depth_file_name_path;
    for (std::filesystem::__cxx11::directory_entry entry : fs::directory_iterator(rgb_path)) {
        map_rgb_file_name_path.insert({extract_timestamp(entry.path()), entry.path()});
    }
    for (std::filesystem::__cxx11::directory_entry entry : fs::directory_iterator(depth_path)) {
        map_depth_file_name_path.insert({extract_timestamp(entry.path()), entry.path()});
    }
    std::vector<std::pair<std::string, std::string>> map_rgb_depth = get_mapping_between_rgb_depth("../rgbd_dataset_freiburg1_xyz/maping_rgb_depth.txt");
    std::unordered_map<std::string, Position_Entry> map_rgb_pose = get_mapping_between_rgb_position("../rgbd_dataset_freiburg1_xyz/maping_rgb_groundtruth.txt");
    for (int i = 0; i < (int)map_rgb_depth.size(); i++) {
        std::pair<std::string, std::string> pair_names = map_rgb_depth[i];
        if (map_rgb_pose.find(pair_names.first) == map_rgb_pose.end()) continue;
        this->rgb_path.push_back(map_rgb_file_name_path[pair_names.first]);
        this->depth_path.push_back(map_depth_file_name_path[pair_names.second]);
        this->groundtruth_poses.push_back(map_rgb_pose[pair_names.first].pose);
    }
}   

Sophus::SE3d TumDatasetReader::get_next_groundtruth_pose() {
    Sophus::SE3d pose = this->groundtruth_poses[idx];
    return pose;
}

bool TumDatasetReader::should_end() {
    return this-> idx == 790;
}


cv::Mat TumDatasetReader::get_next_frame() {
    std::cout << idx << " " << this->rgb_path[idx] << " ";
    cv::Mat frame = cv::imread(this->rgb_path[idx], cv::IMREAD_COLOR_RGB);
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    return gray;
}


cv::Mat TumDatasetReader::get_next_depth() {
    std::cout << this->depth_path[idx] << "\n";
    cv::Mat depth = cv::imread(this->depth_path[idx], cv::IMREAD_UNCHANGED);
    return depth;
}

void TumDatasetReader::increase_idx() {
    this->idx++;
}


void TumDatasetReader::store_entry(Sophus::SE3d pose, bool isKeyFrame) {
    FramePoseEntry framePoseEntry;
    framePoseEntry.pose = pose;
    framePoseEntry.isKeyFrame = isKeyFrame;
    this->frame_poses.push_back(framePoseEntry);
}


void TumDatasetReader::write_entry(Sophus::SE3d pose, int index) {
    std::string path = this->rgb_path[index];
    std::string file_name = extract_timestamp(path);
    this->outfile << file_name <<  " " << -pose.translation().y() << " " << -pose.translation().x() << " " << pose.translation().z() << " ";
    this->outfile << pose.unit_quaternion().x() << " " << pose.unit_quaternion().y() << " "  << pose.unit_quaternion().z() << " " << pose.unit_quaternion().w() << std::endl; 
}


void TumDatasetReader::write_all_entries() {
    std::cout << "AICI INCEPE SCRIEREA IN FISIER\n";
    // int last_keyframe_idx = 0;
    for (int i = 0; i <  (int)this->frame_poses.size(); i++) {
        FramePoseEntry framePoseEntry = this->frame_poses[i]; 
        // last_keyframe_idx = i;
        write_entry(framePoseEntry.pose, i);
    }
    std::cout << "AICI SE TERMINA SCRIEREA IN FISIER\n";
}

