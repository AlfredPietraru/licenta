
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Tracker.h"
#include "include/LocalMapping.h"
#include "include/TumDatasetReader.h"
#include "include/ORBVocabulary.h"
#include <csignal>
#include <cstdlib>
#include <chrono>
namespace fs = std::filesystem;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::milliseconds;
// evo_traj tum groundtruth.txt estimated.txt -p --plot_mode xyz
// TODO:
// separare si abstractizare pentru a putea testa mai multe implementari ale diversilor algoritmi de matching intre frame - uri, de testat fiecare componenta cu mai multe metode -> FLANN, Brute Force, PnP
// testare pe alt set de date
// abstractizarea constante importante

TumDatasetReader *reader;
Tracker *tracker;
int total_tracking_duration = 0;
int total_local_mapping_duration = 0;

void display_timing_information() {  
    std::cout << tracker->reference_kf->reference_idx << " atatea keyframe-uri avute\n";
    std::cout << total_local_mapping_duration / 1000 << " atat a durat doar local mapping\n\n";
    std::cout << total_tracking_duration / 1000 << " atat a durat tracking-ul in total\n";
    std::cout << tracker->orb_matching_time / 1000 << " orb feature matching time\n";
    std::cout << tracker->motion_only_ba_time / 1000 << " motion only BA time\n";
}

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    reader->write_all_entries();
    display_timing_information();
    
    reader->outfile.close();
    exit(signum);
}

int main()
{
    std::signal(SIGINT, signalHandler);
    ORBVocabulary *voc = new ORBVocabulary();
    bool bVocLoad = voc->loadFromTextFile("../ORBvoc.txt");
    if (!bVocLoad) {
        std::cout << "Nu s-a putut incarca corespunzator fisierul ORBvoc.txt\n";
        exit(1);
    } else {
        std::cout << "Fisierul a fost incarcat cu succes\n";
    }

    Config cfg = loadConfig("../config.yaml");
    
    Map *mapp = new Map();
    MapDrawer *drawer = new MapDrawer(mapp);
    reader = new TumDatasetReader(cfg, mapp);
    std::cout << "reader-ul a fost incarcat cu succes\n"; 
    LocalMapping *local_mapper = new LocalMapping(mapp);
    tracker = new Tracker(mapp, cfg, voc);

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