#include "../include/FeatureFinderMatcher.h"

FeatureMatcherFinder::FeatureMatcherFinder(int rows, int cols, Config cfg)
{
    this->splits = cfg.split_size;
    this->nr_cells_row = rows / cfg.split_size;
    this->nr_cells_collumn = cols / cfg.split_size;
    this->fast = cv::FastFeatureDetector::create(cfg.fast_threshold, true, cv::FastFeatureDetector::TYPE_9_16);
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

std::vector<cv::KeyPoint> FeatureMatcherFinder::get_keypoints_current_sub_image(cv::Mat &sub_img, int i, int j)
{
    std::vector<cv::KeyPoint> kps;
    int threshold = this->fast_features_cell[i * this->nr_cells_collumn + j];
    for (int iter = 0; iter < this->orb_iterations - 1; iter++)
    {
        this->orb->setFastThreshold(threshold);
        this->orb->detect(sub_img, kps);
        if (kps.size() >= 2 * minim_keypoints && kps.size() <= minim_keypoints * 4)
        {
            // std::cout << "out " << i << " " << j << "\n";
            break;
        }
        else if (kps.size() < 2 * minim_keypoints)
        {
            if (threshold == this->fast_lower_limit)
                break;
            threshold -= this->fast_step;
        }
        else if (kps.size() > 4 * minim_keypoints)
        {
            if (threshold == this->fast_higher_limit)
                break;
            threshold += this->fast_step;
        }
        this->orb->setFastThreshold(threshold);
        kps.clear();
    }
    this->orb->setFastThreshold(threshold);
    this->fast->detect(sub_img, kps);
    for (auto &kp : kps)
    {
        kp.pt.x += j * this->nr_cells_collumn;
        kp.pt.y += i * this->nr_cells_row;
    }
    this->fast_features_cell[i * nr_cells_collumn + j] = threshold; 

    if (kps.size() <= 2 * minim_keypoints) return kps;    
    for (int i = 0; i < kps.size() - 1; i++)
    {
            for (int j = i + 1; j < kps.size(); j++)
            {
                if (kps[i].response < kps[j].response)
                {
                    cv::KeyPoint aux = kps[i];
                    kps[i] = kps[j];
                    kps[j] = aux;
                }
            }
    }
    return std::vector<cv::KeyPoint>(kps.begin(), kps.begin() + 2 * minim_keypoints);
}

std::vector<cv::KeyPoint> FeatureMatcherFinder::extract_keypoints(cv::Mat &frame)
{
    cv::Mat grayMat;
    cv::cvtColor(frame, grayMat, cv::COLOR_RGB2GRAY);
    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < this->splits - 1; i++)
    {
        for (int j = 0; j < this->splits - 1; j++)
        {
            cv::Rect roi(j * nr_cells_collumn, i * nr_cells_row, nr_cells_collumn, nr_cells_row);
            cv::Mat sub_img = grayMat(roi).clone();
            std::vector<cv::KeyPoint> current_keypoints = get_keypoints_current_sub_image(sub_img, i, j);
            if (current_keypoints.size() == 0) continue;
            // std::cout << current_keypoints.size() << "\n";
            keypoints.insert(keypoints.end(), current_keypoints.begin(), current_keypoints.end());
        }
    }
    std::cout << keypoints.size() << " keypoints obtinute inainte de filtrare\n";
    cv::KeyPointsFilter::runByImageBorder(keypoints, frame.size(), 20);
    std::cout << keypoints.size() << " keypoints obtinute dupa filtrare\n\n";
    return keypoints;
}

cv::Mat FeatureMatcherFinder::compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint> &kps)
{
    cv::Mat descriptors;
    // int points_above_threshold = 0;
    // for (int i = 0; i < kps.size(); i++)
    // {
    //     if (kps[i].response > 30)
    //         points_above_threshold++;
    // }
    // std::cout << "puncte sunt valide " << points_above_threshold << " \n";
    this->orb->compute(frame, kps, descriptors);
    return descriptors;
}