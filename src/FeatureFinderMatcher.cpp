#include "../include/FeatureFinderMatcher.h"


FeatureMatcherFinder::FeatureMatcherFinder(int rows, int cols, Config cfg) {
    this->nr_cells_row = rows / 40;
    this->nr_cells_collumn = cols / 40;
    this->window = 40;
    this->K = cfg.K;
    this->mDistCoef = cfg.distortion;
    this->orb = cv::ORB::create(cfg.num_features, 1.2F, 8, cfg.edge_threshold, 0, 2, cv::ORB::HARRIS_SCORE, cfg.patch_size, cfg.fast_threshold);
    this->orb_edge_threshold = cfg.edge_threshold;
    this->fast_step = cfg.fast_step;
    this->orb_iterations = cfg.orb_iterations;
    this->minim_keypoints = cfg.min_keypoints_cell;
    this->fast_lower_limit = cfg.fast_lower_limit;
    this->fast_higher_limit = cfg.fast_higher_limit;
    this->fast_threshold = cfg.fast_threshold;
    this->fast_features_cell = std::vector<int>(this->nr_cells_collumn * this->nr_cells_row, this->fast_threshold);
    this->matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}


std::vector<cv::KeyPoint> FeatureMatcherFinder::UndistortKeyPoints(std::vector<cv::KeyPoint> kps)
{
    if(mDistCoef[0]==0.0) return kps;
    cv::Mat mat(kps.size(), 2, CV_32F);
    for(int i=0; i<kps.size(); i++)
    {
        mat.at<float>(i,0)=kps[i].pt.x;
        mat.at<float>(i,1)=kps[i].pt.y;
    }

    mat=mat.reshape(2);
    cv::undistortPoints(mat, mat, this->K, mDistCoef,cv::Mat(),this->K);
    mat=mat.reshape(1);

    std::vector<cv::KeyPoint> out;
    
    for(int i=0; i<kps.size(); i++)
    {
        kps[i].pt.x=mat.at<float>(i,0);
        kps[i].pt.y=mat.at<float>(i,1);
        out.push_back(kps[i]);
    }
    return out;
}

std::pair<std::pair<std::vector<cv::KeyPoint>, cv::Mat>, std::vector<cv::KeyPoint>> FeatureMatcherFinder::compute_keypoints_descriptors(cv::Mat frame) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    (*this->extractor)(frame, cv::Mat(), keypoints, descriptors);
    std::vector<cv::KeyPoint> undistorted_kps = this->UndistortKeyPoints(keypoints);

    return {{keypoints, descriptors}, undistorted_kps};
}



std::vector<cv::KeyPoint> FeatureMatcherFinder::extract_keypoints(cv::Mat& frame) {
    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < nr_cells_row - 1; i++) {
        for (int j = 0; j < nr_cells_collumn - 1; j++) {
            std::vector<cv::KeyPoint> current_keypoints;
            cv::Rect roi(j * this->window, i * this->window, this->window, this->window);
            cv::Mat cell_img = frame(roi).clone();
            int threshold = this->fast_features_cell[i * nr_cells_collumn + j];
            for (int iter = 0; iter < this->orb_iterations - 1; iter++) {
                this->orb->setFastThreshold(threshold);
                this->orb->detect(cell_img, current_keypoints);
                if (current_keypoints.size() >= minim_keypoints && current_keypoints.size() <= minim_keypoints * 3) {
                    break;
                } else if (current_keypoints.size() < minim_keypoints) {
                    if (threshold == this->fast_lower_limit) break;
                    threshold -= this->fast_step;
                     
                } else if (current_keypoints.size() > 3 * minim_keypoints) {
                    if (threshold == this->fast_higher_limit) break;
                    threshold += this->fast_step;
                }
                this->orb->setFastThreshold(threshold);
                current_keypoints.clear();
            }
            this->orb->setFastThreshold(threshold);
            this->orb->detect(cell_img, current_keypoints);
            for (auto &kp : current_keypoints) {
                kp.pt.x += j * window;
                kp.pt.y += i * window;
            }
            this->fast_features_cell[i * nr_cells_collumn + j] = threshold;
            if (current_keypoints.size() > minim_keypoints * 2) {
                std::sort(current_keypoints.begin(), current_keypoints.end(), 
                    [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                        return a.response > b.response;
                    });
                current_keypoints.resize(minim_keypoints);
            }
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());
        }
    }
    cv::KeyPointsFilter::removeDuplicated(keypoints);
    cv::KeyPointsFilter::runByImageBorder(keypoints, frame.size(), 20);
    return keypoints;
}

