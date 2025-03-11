#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(cv::Mat frame){
    this->nr_cells_row = frame.rows / WINDOW;
    this->nr_cells_collumn = frame.cols / WINDOW;
    // std::cout << this->nr_cells_row << " " << this->nr_cells_collumn << " celule pe randuri si coloane\n";
    this->orb = cv::ORB::create(ORB_FEATURES, 1.2F, 8, 10, 0, 2, cv::ORB::HARRIS_SCORE, ORB_EDGE_THRESHOLD, FAST_THRESHOLD);
    this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    this->fast_features_cell = std::vector<int>(this->nr_cells_collumn * this->nr_cells_row, FAST_THRESHOLD);
    this->mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
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
    for (int i = 0; i < nr_cells_row; i++) {
        for (int j = 0; j < nr_cells_collumn; j++) {
            std::vector<cv::KeyPoint> current_keypoints;
            int max_iter = 0;
            // std::cout << i << " " << j << "\n";
            for (int k = WINDOW * i; k < WINDOW * i + WINDOW; k++) {
                for (int l = WINDOW * j; l < WINDOW * j + WINDOW; l++) {
                    mask.at<uint8_t>(k, l) = 1;
                }
            }

            // do stuff
            // ajunge la 0, si dupa NU mai da un kick start ceea ce nu e bine
            int threshold = this->fast_features_cell[i * nr_cells_row + j];
            if (threshold <= 0) threshold = FAST_THRESHOLD / 2;

            for (int i = 0; i < this->ORB_ITERATIONS; i++) {
                this->orb->setFastThreshold(threshold);
                this->orb->detect(frame, current_keypoints, mask);
                // std::cout << current_keypoints.size() << " " << this->orb->getFastThreshold() << "  ";
                if (current_keypoints.size() >= EACH_CELL_THRESHOLD && current_keypoints.size() <= EACH_CELL_THRESHOLD * 2) {
                    // std::cout << "out " << i << " " << j << "\n";
                    break;
                } else if (current_keypoints.size() < EACH_CELL_THRESHOLD) {
                    threshold -= this->FAST_STEP;
                    if (threshold <= 0) break;
                } else if (current_keypoints.size() > 2 * EACH_CELL_THRESHOLD) {
                    threshold += this->FAST_STEP;
                }
                this->orb->setFastThreshold(threshold);
                current_keypoints.clear();
                max_iter++;
            }
            this->fast_features_cell[i * nr_cells_row + j] = threshold;
            // std::cout << " " << this->orb->getFastThreshold() << "\n\n";
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());
            // end stuff
            for (int k = WINDOW * i; k < WINDOW * i + WINDOW; k++) {
                for (int l = WINDOW * j; l < WINDOW * j + WINDOW; l++) {
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
    // for (int i = 0; i < kps.size(); i++) {
    //     std::cout << kps[i].octave << " ";
    // }
    // std::cout << "\n";
    return descriptors;
}