#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(int rows, int cols, Config cfg) {
    this->nr_cells_row = rows / cfg.feature_window;
    this->nr_cells_collumn = cols / cfg.feature_window;
    this->window = cfg.feature_window;
    this->orb = cv::ORB::create(cfg.num_features, 1.2F, 8, cfg.edge_threshold, 0, 2, cv::ORB::HARRIS_SCORE, cfg.patch_size, cfg.fast_threshold);
    this->orb_edge_threshold = cfg.edge_threshold;
    this->fast_step = cfg.fast_step;
    this->orb_iterations = cfg.orb_iterations;
    this->minim_keypoints = cfg.min_keypoints_cell;
    this->fast_lower_limit = cfg.fast_lower_limit;
    this->fast_higher_limit = cfg.fast_higher_limit;
    this->fast_threshold = cfg.fast_threshold;
    this->interlaping = cfg.interlaping;
    this->matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

std::vector<cv::DMatch> FeatureMatcherFinder::match_features_last_frame(KeyFrame *current_kf, KeyFrame *past_kf) {
    std::vector< std::vector<cv::DMatch>> knn_matches;
    cv::Mat desc1_32f, desc2_32f;
    current_kf->orb_descriptors.convertTo(desc1_32f, CV_32F);
    past_kf->orb_descriptors.convertTo(desc2_32f, CV_32F);
    matcher->knnMatch(desc2_32f, desc1_32f, knn_matches, 2 );
    const float ratio_thresh = 0.85f;
    std::vector<cv::DMatch> good_matches;

    for (const auto& knn_match : knn_matches) {
        if (knn_match.size() < 2) continue;

        const auto& best_match = knn_match[0];
        const auto& second_best_match = knn_match[1];

        if (best_match.distance < ratio_thresh * second_best_match.distance &&
            best_match.queryIdx >= 0 && best_match.queryIdx < static_cast<int>(current_kf->features.size()) &&
            best_match.trainIdx >= 0 && best_match.trainIdx < static_cast<int>(past_kf->features.size())) {
            good_matches.push_back(best_match);
        }
    }
    // std::cout << " acoloooooo\n";

    // cv::Mat img_matches;
    // std::cout << past_kf->keypoints.size() << " " << current_kf->keypoints.size() << "\n";
    // std::cout << good_matches.size() << "\n";
    // cv::drawMatches(past_kf->frame, past_kf->keypoints, current_kf->frame, current_kf->keypoints, good_matches, img_matches);
    // // Show the matches
    // cv::imshow("Feature Matches", img_matches);
    // cv::waitKey(0);
    return good_matches;
} 

std::vector<cv::KeyPoint> FeatureMatcherFinder::extract_keypoints(cv::Mat& frame) {
    std::vector<cv::KeyPoint> keypoints;
    // std::cout << nr_cells_row << " " << nr_cells_collumn << "\n";
    for (int i = 0; i < nr_cells_row * interlaping - interlaping + 1; i++) {
        for (int j = 0; j < nr_cells_collumn * interlaping - interlaping + 1; j++) {
            std::vector<cv::KeyPoint> current_keypoints;
            // std::cout << i << " " << j << "\n";
            cv::Rect roi(j * this->window / interlaping, i * this->window / interlaping, this->window, this->window);
            // std::cout << roi << "\n";
            cv::Mat cell_img = frame(roi).clone();

            // do stuff
            // ajunge la 0, si dupa NU mai da un kick start ceea ce nu e bine
            int threshold = this->fast_threshold;
            for (int iter = 0; iter < this->orb_iterations - 1; iter++) {
                this->orb->setFastThreshold(threshold);
                this->orb->detect(cell_img, current_keypoints);
                // std::cout << current_keypoints.size() << " " << this->orb->getFastThreshold() << "  ";
                if (current_keypoints.size() >= minim_keypoints && current_keypoints.size() <= minim_keypoints * 2) {
                    // std::cout << "out " << i << " " << j << "\n";
                    break;
                } else if (current_keypoints.size() < minim_keypoints) {
                    if (threshold == this->fast_lower_limit) break;
                    threshold -= this->fast_step;
                     
                } else if (current_keypoints.size() > 2 * minim_keypoints) {
                    if (threshold == this->fast_higher_limit) break;
                    threshold += this->fast_step;
                }
                this->orb->setFastThreshold(threshold);
                current_keypoints.clear();
            }
            this->orb->setFastThreshold(threshold);
            this->orb->detect(cell_img, current_keypoints);

            // for (cv::KeyPoint kp : current_keypoints) {
            //     std::cout << kp.pt << " ";
            // }
            // cv::Mat current;
            // cv::drawKeypoints(cell_img, current_keypoints, current, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
            // imshow("Display window", current);
            // cv::waitKey(0);
            // std::cout << "\n";
            for (auto &kp : current_keypoints) {
                kp.pt.x += j * window / interlaping;
                kp.pt.y += i * window / interlaping;
            }
            // for (cv::KeyPoint kp : current_keypoints) {
            //     std::cout << kp.pt << " ";
            // }
            // std::cout << "\n";
            // std::cout << " " << this->fast_features_cell[i * nr_cells_collumn + j] << " " << this->orb->getFastThreshold() << "\n\n";
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());

            // break;
        }
        // break;
    }
    // std::cout << keypoints.size() << " keypoints obtinute inainte de filtrare\n";
    cv::KeyPointsFilter::removeDuplicated(keypoints);
    cv::KeyPointsFilter::runByImageBorder(keypoints, frame.size(), this->orb_edge_threshold);
    // std::cout << keypoints.size() << " keypoints obtinute dupa filtrare\n\n";
    // std::cout << keypoints.size() << "\n";

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
    //  for (int q = 0; q < kps.size(); q++) {
    //     std::cout << kps[q].pt.x << " " << kps[q].pt.y << "\n";
    // }
    return descriptors;
}