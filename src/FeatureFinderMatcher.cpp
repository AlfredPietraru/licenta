#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(cv::Size frame_size, int window_size) {
    this->nr_cells = frame_size.height / window_size;
    this->window = window_size;
    this->orb = cv::ORB::create(1000, 1.2F, 8, ORB_EDGE_THRESHOLD, 0, 2, cv::ORB::HARRIS_SCORE, ORB_EDGE_THRESHOLD, FAST_THRESHOLD);
    this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

std::vector<cv::DMatch> FeatureMatcherFinder::match_features_last_frame(KeyFrame *current_kf, KeyFrame *past_kf) {
    std::vector<cv::DMatch> matches;
    this->matcher->match(current_kf->orb_descriptors, past_kf->orb_descriptors, matches);
    std::vector<cv::DMatch> good_matches; 
    for (auto match : matches) {
        if(match.distance < LIMIT_MATCHING) {
            good_matches.push_back(match);
        }
    }
    return good_matches;
} 

std::vector<cv::KeyPoint> FeatureMatcherFinder::extract_keypoints(cv::Mat frame) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
    nr_features_extracted.clear();
    for (int i = 0; i < nr_cells; i++) {
        for (int j = 0; j < nr_cells; j++) {
            std::vector<cv::KeyPoint> current_keypoints;
            int max_iter = 0;
            // std::cout << i << " " << j << "\n";
            for (int k = nr_cells * i; k < nr_cells * i + this->window; k++) {
                for (int l = nr_cells * j; l < nr_cells * j + this->window; l++) {
                    mask.at<uint8_t>(k, l) = 1;
                }
            }
            // do stuff
            // ajunge la 0, si dupa NU mai da un kick start ceea ce nu e bine
            while(max_iter < this->ORB_ITERATIONS) {
                if (this->orb->getFastThreshold() <= 0) {
                    this->orb->setFastThreshold(FAST_THRESHOLD);
                } 
                this->orb->detect(frame, current_keypoints, mask);
                // std::cout << current_keypoints.size() << " " << this->orb->getFastThreshold() << "  ";
                if (current_keypoints.size() >= EACH_CELL_THRESHOLD && current_keypoints.size() <= EACH_CELL_THRESHOLD * 2) {
                    // std::cout << "out " << i << " " << j << "\n";
                    break;
                } 
                if (current_keypoints.size() < EACH_CELL_THRESHOLD) {
                    this->orb->setFastThreshold(this->orb->getFastThreshold() - this->FAST_STEP);
                    current_keypoints.clear();
                }
                if (current_keypoints.size() > EACH_CELL_THRESHOLD * 2) {
                    this->orb->setFastThreshold(this->orb->getFastThreshold() + this->FAST_STEP);
                    current_keypoints.clear();
                } 
                max_iter++;
            }
            // std::cout << " " << this->orb->getFastThreshold() << "\n\n";
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());
            nr_features_extracted.push_back(current_keypoints.size());
            // end stuff
            for (int k = nr_cells * i; k < nr_cells * i + this->window; k++) {
                for (int l = nr_cells * j; l < nr_cells * j + this->window; l++) {
                    mask.at<uint8_t>(k, l) = 0;
                }
            }
            // break;           
        }
        // break;
    }
    // for (int q = 0; q < keypoints.size(); q++) {
    //     std::cout << keypoints[q].pt.x << " " << keypoints[q].pt.y << "\n";
    // }
    // std::cout << keypoints.size() << "\n";
    // cv::Mat img2;
    // cv::drawKeypoints(frame, keypoints, img2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    // imshow("Display window", img2);
    // cv::waitKey(0);
    return keypoints;
}

cv::Mat FeatureMatcherFinder::compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint> &kps) {
    // std::cout << kps.size() << "\n\n";
    cv::Mat descriptors;
    this->orb->compute(frame, kps, descriptors);
    return descriptors;
}