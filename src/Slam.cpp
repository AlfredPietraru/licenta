#include "../include/Slam.h"

SLAM::SLAM(std::string voc_file, std::string config_file, Tum_Dataset_Structure dt) {
    this->voc = new ORBVocabulary();
    bool bVocLoad = voc->loadFromTextFile(voc_file);
    if (!bVocLoad) {
        std::cout << "Nu s-a putut incarca corespunzator fisierul ORBvoc.txt\n";
        exit(1);
    } else {
        std::cout << "Fisierul a fost incarcat cu succes\n";
    }

    this->dt = dt;
    Config cfg = loadConfig(config_file);
    this->mapp = new Map();
    this->reader = new TumDatasetReader(mapp, dt.path_to_write, dt.rgb_paths_location, dt.depth_path_location,
        dt.mapping_rgb_depth_filename, dt.mapping_rgb_groundtruth);
    this->tracker = new Tracker(mapp, cfg, voc);
    this->drawer = new MapDrawer(mapp);
    this->local_mapper = new LocalMapping(mapp);
}

void SLAM::display_timing_information() {
    std::cout << "\n\n";  
    std::cout << total_duration << " atat a durat in total algoritmul\n";
    std::cout << total_tracking_duration / 1000 << " atat a durat tracking-ul in total\n";
    std::cout << total_local_mapping_duration / 1000 << " atat a durat doar local mapping\n\n";
    std::cout << tracker->reference_kf->reference_idx << " atatea keyframe-uri avute\n";
    std::cout << tracker->orb_matching_time / 1000 << " orb feature matching time\n";
    std::cout << tracker->motion_only_ba_time / 1000 << " motion only BA time\n";
}

void SLAM::run_slam_systems() {
    auto t1 = high_resolution_clock::now();
    while(!reader->should_end()) {
        cv::Mat frame = reader->get_next_frame(); 
        cv::Mat depth = reader->get_next_depth();
        // cv::Mat depth = reader->get_next_depth_neural_network();
        // cv::imshow("frame", frame);
        // cv::imshow("depth", depth);
        // cv::imshow("neural_network_depth", neural_network_depth);
        // std::cout << depth.row(479) << "\n\n";
        // std::cout << neural_network_depth.row(479) << "\n\n";
        // cv::waitKey(0);   

        auto start = high_resolution_clock::now();
        KeyFrame *kf = tracker->tracking(frame, depth);
        auto end = high_resolution_clock::now();
        total_tracking_duration += duration_cast<milliseconds>(end - start).count();
        if (kf->isKeyFrame) {
            auto start = high_resolution_clock::now();
            local_mapper->local_map(kf);
            auto end = high_resolution_clock::now();
            total_local_mapping_duration += duration_cast<milliseconds>(end - start).count();
        }
        // if (reader->idx == 500) {
        //     cv::imshow("iei", depth);
        //     cv::waitKey(0);
        // }
        reader->store_entry(kf);
        reader->increase_idx();
        drawer->run(kf, frame);
    }

    auto t2 = high_resolution_clock::now();
    this->total_duration = duration_cast<seconds>(t2 - t1).count(); 
    reader->write_all_entries(this->dt.translation_pose);
}