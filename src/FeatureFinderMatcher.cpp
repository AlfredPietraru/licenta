#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(cv::Mat frame){
    this->nr_cells_row = frame.rows / WINDOW;
    this->nr_cells_collumn = frame.cols / WINDOW;
    // std::cout << this->nr_cells_row << " " << this->nr_cells_collumn << " celule pe randuri si coloane\n";
    this->orb = cv::ORB::create(ORB_FEATURES, 1.2F, 8, 10, 0, 2, cv::ORB::HARRIS_SCORE, ORB_EDGE_THRESHOLD, FAST_THRESHOLD);
    this->matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    this->fast_features_cell = std::vector<int>(this->nr_cells_collumn * this->nr_cells_row, FAST_THRESHOLD);
    this->nr_keypoints_found = std::vector<int>(this->nr_cells_collumn * this->nr_cells_row, 0);
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
    // for (int i = 0; i < nr_keypoints_found.size(); i++) {
    //     std::cout << this->nr_keypoints_found[i] << " ";
    // }
    // std::cout << "\n\n";
    // for (int i = 0; i < fast_features_cell.size(); i++) {
    //     std::cout << this->fast_features_cell[i] << " ";
    // }
    // std::cout << "\n\n\n";

    std::vector<cv::KeyPoint> keypoints;  
    for (int i = 0; i < nr_cells_row; i++) {
        for (int j = 0; j < nr_cells_collumn; j++) {
            std::vector<cv::KeyPoint> current_keypoints;
            // std::cout << i << " " << j << "\n";
            for (int k = WINDOW * i; k < WINDOW * i + WINDOW; k++) {
                for (int l = WINDOW * j; l < WINDOW * j + WINDOW; l++) {
                    mask.at<uint8_t>(k, l) = 1;
                }
            }

            // do stuff
            // ajunge la 0, si dupa NU mai da un kick start ceea ce nu e bine
            int threshold = this->fast_features_cell[i * nr_cells_collumn + j];

            for (int iter = 0; iter < this->ORB_ITERATIONS - 1; iter++) {
                this->orb->setFastThreshold(threshold);
                this->orb->detect(frame, current_keypoints, mask);
                // std::cout << current_keypoints.size() << " " << this->orb->getFastThreshold() << "  ";
                if (current_keypoints.size() >= EACH_CELL_THRESHOLD && current_keypoints.size() <= EACH_CELL_THRESHOLD * 2) {
                    // std::cout << "out " << i << " " << j << "\n";
                    break;
                } else if (current_keypoints.size() < EACH_CELL_THRESHOLD) {
                    if (threshold == 5) break;
                    threshold -= this->FAST_STEP;
                     
                } else if (current_keypoints.size() > 2 * EACH_CELL_THRESHOLD) {
                    threshold += this->FAST_STEP;
                }
                this->orb->setFastThreshold(threshold);
                current_keypoints.clear();
            }
            this->orb->setFastThreshold(threshold);
            this->orb->detect(frame, current_keypoints, mask);
            this->fast_features_cell[i * nr_cells_collumn + j] = threshold;
            this->nr_keypoints_found[i * nr_cells_collumn + j] = current_keypoints.size();
            // std::cout << " " << this->fast_features_cell[i * nr_cells_collumn + j] << " " << this->orb->getFastThreshold() << "\n\n";
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

    // for (int i = 0; i < nr_keypoints_found.size(); i++) {
    //     std::cout << this->nr_keypoints_found[i] << " ";
    // }
    // std::cout << "\n\n";
    // for (int i = 0; i < fast_features_cell.size(); i++) {
    //     std::cout << this->fast_features_cell[i] << " ";
    // }
    // std::cout << "\n\n\n";
    return keypoints;
}

cv::Mat FeatureMatcherFinder::compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint> &kps) {
    // cv::Mat descriptors = cv::Mat::zeros(cv::Size(32, kps.size()), CV_64F);
    // std::cout << descriptors.size() << "\n";
    // std::cout << descriptors.size().height << " " << descriptors.size().width << "\n";
    // // this->orb->compute(frame, kps, descriptors);
    // int kps_idx = 0;
    // int descriptor_idx = 0;
    // for (int i = 0; i < nr_cells_row; i++) {
    //     for (int j = 0; j < nr_cells_collumn; j++) {
    //         std::vector<cv::KeyPoint> current_kps;
    //         cv::Mat current_descriptors;
    //         int kps_found_in_cell = this->nr_keypoints_found[i * nr_cells_row + j];
    //         copy(kps.begin() + kps_idx, kps.begin() + kps_idx + kps_found_in_cell, back_inserter(current_kps));
    //         this->orb->setFastThreshold(this->fast_features_cell[i * nr_cells_row + j]);
    //         kps_idx += kps_found_in_cell;
    //         this->orb->compute(frame, current_kps, current_descriptors);
    //         if (current_descriptors.size().height == 0) continue;
    //         for (int q = 0; q < current_descriptors.size().height; q++) {
    //             descriptors.row(descriptor_idx + q) = current_descriptors.row(q);
    //         }
    //         descriptor_idx += current_descriptors.size().height;
    //     }
    // }
    
    // if (descriptor_idx < descriptors.size().height) {
    //     std::cout << descriptor_idx << " waiiiii trebuie de modificat\n";
    // }
    // // std::cout << descriptors << "\n";
    // cv::Mat dst_roi = descriptors(cv::Rect(0, 0, 32, descriptor_idx));
    // // std::cout << dst_roi.row(100) << "\n\n";
    // // std::cout << descriptors.row(100) << "\n";

    // std::cout << this->orb->getFastThreshold() << "\n";
    cv::Mat descriptors;
    this->orb->compute(frame, kps, descriptors);
    return descriptors;
}