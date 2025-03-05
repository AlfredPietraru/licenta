#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(cv::Size frame_size, int window_size) {
    this->nr_cells = frame_size.height / window_size;
    this->window = window_size;
    for (int i = 0; i < this->nr_cells; i++) {
        std::vector<cv::Ptr<cv::FastFeatureDetector>> current_vec;
        for (int j = 0; j < this->nr_cells; j++) {
            current_vec.push_back(cv::FastFeatureDetector::create(this->INITIAL_FAST_INDEX, true));
        }
        this->fast_vector.push_back(current_vec);
    }
    this->orb = cv::ORB::create(1000, 1.2F, 8, 30, 0, 2, cv::ORB::HARRIS_SCORE, 5, 20);
    this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
}

std::vector<cv::KeyPoint> FeatureMatcherFinder::extract_keypoints(cv::Mat frame) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
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
            while(max_iter < this->FAST_ITERATIONS) {
                if (this->fast_vector[i][j]->getThreshold() <= 0) break;
                this->fast_vector[i][j]->detect(frame, current_keypoints, mask);
                // std::cout << current_keypoints.size() << " " << this->fast_vector[i][j]->getThreshold() << "  ";
                if (current_keypoints.size() >= EACH_CELL_THRESHOLD && current_keypoints.size() <= EACH_CELL_THRESHOLD * 2) {
                    // std::cout << "out " << i << " " << j << "\n";
                    break;
                } 
                if (current_keypoints.size() < EACH_CELL_THRESHOLD) {
                    this->fast_vector[i][j]->setThreshold(this->fast_vector[i][j]->getThreshold() - this->FAST_STEP);
                    current_keypoints.clear();
                }
                if (current_keypoints.size() > EACH_CELL_THRESHOLD * 2) {
                    this->fast_vector[i][j]->setThreshold(this->fast_vector[i][j]->getThreshold() + this->FAST_STEP);
                    current_keypoints.clear();
                } 
                max_iter++;
            }
            // std::cout << " " << this->fast_vector[i][j]->getThreshold() << "\n\n";
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());
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

cv::Mat FeatureMatcherFinder::compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint> &keypoints) {
    cv::Mat descriptors;
    // std::cout << this->orb->getEdgeThreshold() << "\n\n";
    this->orb->compute(frame, keypoints, descriptors);
    // std::cout << keypoints.size() << " " << descriptors.size() << " \n"; 
    return descriptors;
}