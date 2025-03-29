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
    this->fast_features_cell = std::vector<int>(this->nr_cells_collumn * this->nr_cells_row * this->interlaping, this->fast_threshold);
    this->matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
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
            // int threshold = this->fast_threshold;
            int threshold = this->fast_features_cell[i * nr_cells_collumn * interlaping + j];
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
            this->fast_features_cell[i * nr_cells_collumn * interlaping + j] = threshold;
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
    cv::Mat descriptors;
    this->orb->compute(frame, kps, descriptors);
    return descriptors;
}