#include "../include/Slam.h"

SLAM::SLAM(std::string voc_file, std::string config_file) {
    this->voc = new ORBVocabulary();
    bool bVocLoad = voc->loadFromTextFile(voc_file);
    if (!bVocLoad) {
        std::cout << "Nu s-a putut incarca corespunzator fisierul ORBvoc.txt\n";
        exit(1);
    } else {
        std::cout << "Fisierul a fost incarcat cu succes\n";
    }

    Config cfg = loadConfig(config_file);

    this->mapp = new Map();
    this->reader = new TumDatasetReader(cfg, mapp);
    this->tracker = new Tracker(mapp, cfg, voc);
    this->drawer = new MapDrawer(mapp);
    this->local_mapper = new LocalMapping(mapp);
}

void SLAM::display_timing_information() {  
    std::cout << tracker->reference_kf->reference_idx << " atatea keyframe-uri avute\n";
    std::cout << this->total_local_mapping_duration / 1000 << " atat a durat doar local mapping\n\n";
    std::cout << this->total_tracking_duration / 1000 << " atat a durat tracking-ul in total\n";
    std::cout << tracker->orb_matching_time / 1000 << " orb feature matching time\n";
    std::cout << tracker->motion_only_ba_time / 1000 << " motion only BA time\n";
}

void SLAM::run_slam_systems() {
    auto t1 = high_resolution_clock::now();
    while(!reader->should_end()) {
        cv::Mat frame = reader->get_next_frame(); 
        cv::Mat depth = reader->get_next_depth();
        // cv::Mat neural_network_depth = reader->get_next_depth_neural_network();
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
        reader->store_entry(kf);
        reader->increase_idx();
        drawer->run(kf, frame);
    }

    reader->write_all_entries();
    auto t2 = high_resolution_clock::now();
    std::cout << duration_cast<seconds>(t2 - t1).count() << "s aici atata a durat\n\n";
    display_timing_information();
}