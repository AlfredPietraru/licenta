#include "include/Tracker.h"
// #include "DBoW3/DBoW3.h"
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // DBoW3::Vocabulary vocab;
    // vocab.load("../vocab_larger.yml.gz"); 
    char x;
    Tracker *tracker = new Tracker();
    Map mapp = tracker->initialize();
    vector<KeyFrame*> keyframes_buffer;
    while(1) {
        tracker->tracking(mapp, keyframes_buffer);
    }
    return 0;
}