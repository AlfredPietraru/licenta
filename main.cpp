
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Slam.h"
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

SLAM *slam = nullptr;

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    if (slam != nullptr) {
        slam->reader->write_all_entries();
        slam->display_timing_information();
        slam->reader->outfile.close();
    }
    exit(signum);
}

int main()
{
    std::signal(SIGINT, signalHandler);
    slam = new SLAM("../ORBvoc.txt", "../config.yaml");
    slam->run_slam_systems();
}