
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "ParticleSlam.h"

using json = nlohmann::json;

// ============================================================================
// HSL Color Interpolation Functions
// ============================================================================

// Helper function to interpolate hue (circular interpolation)
float interpolate_hue(float h1, float h2, float frac) {
    float d = h2 - h1;
    if (h1 > h2) {
        // Swap
        float temp = h2;
        h2 = h1;
        h1 = temp;
        d = -d;
        frac = 1.0f - frac;
    }
    
    if (d > 180.0f) {
        // Go the other way
        h1 = h1 + 360.0f;
        float h = (h1 + frac * (h2 - h1));
        if (h > 360.0f) h -= 360.0f;
        return h;
    }
    
    return h1 + frac * d;
}

// Convert RGB to HLS
void rgb_to_hls(float r, float g, float b, float& h, float& l, float& s) {
    float maxc = std::max(std::max(r, g), b);
    float minc = std::min(std::min(r, g), b);
    
    l = (maxc + minc) / 2.0f;
    
    if (maxc == minc) {
        h = 0.0f;
        s = 0.0f;
        return;
    }
    
    if (l <= 0.5f) {
        s = (maxc - minc) / (maxc + minc);
    } else {
        s = (maxc - minc) / (2.0f - maxc - minc);
    }
    
    float rc = (maxc - r) / (maxc - minc);
    float gc = (maxc - g) / (maxc - minc);
    float bc = (maxc - b) / (maxc - minc);
    
    if (r == maxc) {
        h = bc - gc;
    } else if (g == maxc) {
        h = 2.0f + rc - bc;
    } else {
        h = 4.0f + gc - rc;
    }
    
    h = fmod(h / 6.0f, 1.0f);
    if (h < 0.0f) h += 1.0f;
}

// Helper for HLS to RGB conversion
float hls_helper(float n1, float n2, float h) {
    if (h > 1.0f) h -= 1.0f;
    if (h < 0.0f) h += 1.0f;
    
    if (h < 1.0f / 6.0f) {
        return n1 + (n2 - n1) * 6.0f * h;
    }
    if (h < 0.5f) {
        return n2;
    }
    if (h < 2.0f / 3.0f) {
        return n1 + (n2 - n1) * (2.0f / 3.0f - h) * 6.0f;
    }
    return n1;
}

// Convert HLS to RGB
void hls_to_rgb(float h, float l, float s, float& r, float& g, float& b) {
    if (s == 0.0f) {
        r = g = b = l;
        return;
    }
    
    float m2;
    if (l <= 0.5f) {
        m2 = l * (1.0f + s);
    } else {
        m2 = l + s - (l * s);
    }
    float m1 = 2.0f * l - m2;
    
    r = hls_helper(m1, m2, h + 1.0f / 3.0f);
    g = hls_helper(m1, m2, h);
    b = hls_helper(m1, m2, h - 1.0f / 3.0f);
}

// Interpolate between two colors in HSL space
cv::Scalar interpolate_hsl(cv::Scalar c1, cv::Scalar c2, float frac) {
    // Convert BGR to RGB (OpenCV uses BGR)
    float c1_r = c1[2] / 255.0f;
    float c1_g = c1[1] / 255.0f;
    float c1_b = c1[0] / 255.0f;
    
    float c2_r = c2[2] / 255.0f;
    float c2_g = c2[1] / 255.0f;
    float c2_b = c2[0] / 255.0f;
    
    float c1_h, c1_l, c1_s;
    float c2_h, c2_l, c2_s;
    
    rgb_to_hls(c1_r, c1_g, c1_b, c1_h, c1_l, c1_s);
    rgb_to_hls(c2_r, c2_g, c2_b, c2_h, c2_l, c2_s);
    
    float h = interpolate_hue(c1_h * 360.0f, c2_h * 360.0f, frac) / 360.0f;
    float l = c1_l * (1.0f - frac) + c2_l * frac;
    float s = c1_s * (1.0f - frac) + c2_s * frac;
    
    float res_r, res_g, res_b;
    hls_to_rgb(h, l, s, res_r, res_g, res_b);
    
    // Return as BGR for OpenCV
    return cv::Scalar(
        (int)(res_b * 255.0f),
        (int)(res_g * 255.0f),
        (int)(res_r * 255.0f)
    );
}

int main()
{
    // Create ParticleSlam instance with parameters
    const int NUM_PARTICLES = 10000;
    const int MAX_TRAJECTORY_LENGTH = 300;  // 10 seconds at 30 Hz
    const int MAX_CHUNK_LENGTH = 3600;       // 1 hr at 1 Hz
    
    ParticleSlam slam(NUM_PARTICLES, MAX_TRAJECTORY_LENGTH, MAX_CHUNK_LENGTH);
    
    // Initialize the particle filter
    slam.init();
    
    // Visualization setup
    cv::namedWindow("Particle SLAM", cv::WINDOW_AUTOSIZE);
    const int img_size = 1000;
    const float view_range = 30.0f; // meters
    const float pixels_per_meter = img_size / view_range;
    
    cv::Mat img(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
    int center_x = img_size / 2;
    int center_y = img_size / 2;
    
    // Draw coordinate axes
    cv::line(img, cv::Point(center_x, 0), cv::Point(center_x, img_size), cv::Scalar(200, 200, 200), 1);
    cv::line(img, cv::Point(0, center_y), cv::Point(img_size, center_y), cv::Scalar(200, 200, 200), 1);
    
    // Allocate host memory for visualization
    float* h_scores = new float[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) h_scores[i] = 0.5f;
    
    Vec3* h_chunk_states = new Vec3[MAX_CHUNK_LENGTH * NUM_PARTICLES];

    // Load JSON data
    std::cout << "Loading bigboi_munged.json..." << std::endl;
    std::ifstream json_file("bigboi_munged.json");
    if (!json_file.is_open()) {
        fprintf(stderr, "Failed to open bigboi_munged.json!\n");
        delete[] h_scores;
        delete[] h_chunk_states;
        return 1;
    }
    
    json data;
    json_file >> data;
    json_file.close();
    
    std::cout << "Loaded " << data.size() << " entries from JSON" << std::endl;

    // Track state for dx_step calculation
    Vec3 prev_state = {0.0f, 0.0f, 0.0f};
    bool first_dr_step = true;

    // Process all entries from JSON
    for (const auto& entry : data) {
        std::string type = entry["type"];
        double ts = entry["ts"];

        if (type == "dr_step") {
            Vec3 cur_state;
            cur_state.x = entry["value"]["x"][0];
            cur_state.y = entry["value"]["x"][1];
            cur_state.z = entry["value"]["x"][2];

            if (!first_dr_step) {
                float theta = -prev_state.z;
                Mat2 R = get_R_from_theta(theta);
                Vec2 state_delta = {cur_state.x - prev_state.x, cur_state.y - prev_state.y};
                Vec2 mean_delta = R.transpose() * state_delta;
                float theta_delta = cur_state.z - prev_state.z;

                Vec3 dx_step = {mean_delta.x, mean_delta.y, theta_delta};
                slam.apply_step(dx_step, ts);
            } else {
                first_dr_step = false;
            }

            prev_state = cur_state;

        } else if (type == "map_measurement") {
            // Package map measurement into chunk structure
            Chunk h_chunk;
            h_chunk.timestamp = ts;
            
            for (int i = 0; i < 60; i++) {
                for (int j = 0; j < 60; j++) {
                    h_chunk.cells[i][j].num_pos = 0;
                    h_chunk.cells[i][j].num_neg = 0;
                }
            }

            const auto& cells = entry["value"]["cells"];
            for (int i = 0; i < 60; i++) {
                for (int j = 0; j < 60; j++) {
                    if (!cells[i][j].is_null()) {
                        h_chunk.cells[i][j].num_pos = cells[i][j]["num_pos"];
                        h_chunk.cells[i][j].num_neg = cells[i][j]["num_neg"];
                    }
                }
            }

            // Ingest the chunk (with resampling)
            slam.ingest_visual_measurement(h_chunk, ts, true);

            continue;
            
            // Download data for visualization
            int num_chunks = slam.get_current_chunk_count();
            slam.download_chunk_states(h_chunk_states, MAX_CHUNK_LENGTH);
            slam.download_scores(h_scores);

            // Visualize
            img = cv::Mat(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
            
            cv::line(img, cv::Point(center_x, 0), cv::Point(center_x, img_size), cv::Scalar(200, 200, 200), 1);
            cv::line(img, cv::Point(0, center_y), cv::Point(img_size, center_y), cv::Scalar(200, 200, 200), 1);

            for (int p = 0; p < NUM_PARTICLES; p++) {
                float score = h_scores[p];
                
                int b = 0;
                int g = (int)(255 * score);
                int r = (int)(255 * (1.0f - score));
                cv::Scalar color(b, g, r);
                
                for (int chunk_i = 1; chunk_i < num_chunks; chunk_i++) {
                    Vec3 prev = h_chunk_states[(chunk_i - 1) * NUM_PARTICLES + p];
                    Vec3 curr = h_chunk_states[chunk_i * NUM_PARTICLES + p];
                    
                    int px1 = center_x + (int)(prev.x * pixels_per_meter);
                    int py1 = center_y - (int)(prev.y * pixels_per_meter);
                    int px2 = center_x + (int)(curr.x * pixels_per_meter);
                    int py2 = center_y - (int)(curr.y * pixels_per_meter);
                    
                    cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2), color, 1, cv::LINE_AA);
                }
            }

            cv::rectangle(img, cv::Point(5, 5), cv::Point(250, 90), cv::Scalar(255, 255, 255), -1);
            cv::putText(img, "Timestep: " + std::to_string(slam.get_current_timestep()), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
            cv::putText(img, "Particles: " + std::to_string(NUM_PARTICLES), cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

            cv::imshow("Particle SLAM", img);
            
            if (cv::waitKey(1) == 27) {
                break;
            }
        }
    }

    // Bake final map from best particle
    Map* final_map = slam.bake_map();
    
    // Create image for map visualization
    int pixels_per_cell = 5;
    int map_img_width = final_map->width * pixels_per_cell;
    int map_img_height = final_map->height * pixels_per_cell;
    cv::Mat map_img(map_img_height, map_img_width, CV_8UC3, cv::Scalar(255, 255, 255));  // White background
    
    // Draw each cell
    for (int y = 0; y < final_map->height; y++) {
        for (int x = 0; x < final_map->width; x++) {
            int idx = y * final_map->width + x;
            ChunkCell cell = final_map->cells[idx];
            
            // Only color cells with sufficient observations
            int total_obs = cell.num_pos + cell.num_neg;
            if (total_obs > 1) {  // At least 2 observations
                // Calculate occupancy probability using Beta-Bernoulli model
                float alpha = 1.0f + 0.7f * cell.num_pos;
                float beta = 1.5f + 0.4f * cell.num_neg;
                float prob = alpha / (alpha + beta);
                
                // Color using HSL interpolation from black to orange
                // start = (0, 0, 0) = black
                // stop = (0, 165, 255) = orange (BGR format)
                cv::Scalar start(0, 0, 0);
                cv::Scalar stop(0, 165, 255);
                cv::Scalar color = interpolate_hsl(start, stop, prob);
                
                cv::rectangle(map_img, 
                            cv::Point(x * pixels_per_cell, (final_map->height - 1 - y) * pixels_per_cell), 
                            cv::Point((x + 1) * pixels_per_cell, (final_map->height - y) * pixels_per_cell),
                            color, -1);
            }
        }
    }
    
    // Draw the final chunk trajectory on top
    int num_chunks = slam.get_current_chunk_count();
    
    // Get best particle index
    float* h_scores_for_traj = new float[NUM_PARTICLES];
    slam.download_scores(h_scores_for_traj);
    int best_particle = 0;
    for (int i = 1; i < NUM_PARTICLES; i++) {
        if (h_scores_for_traj[i] > h_scores_for_traj[best_particle]) {
            best_particle = i;
        }
    }
    delete[] h_scores_for_traj;
    
    Vec3* h_chunk_traj = new Vec3[num_chunks];
    slam.download_chunk_states_for_particle(h_chunk_traj, best_particle, num_chunks);
    
    // Draw trajectory as blue line
    for (int i = 1; i < num_chunks; i++) {
        // Transform chunk positions to map coordinates
        Vec3 prev = h_chunk_traj[i - 1];
        Vec3 curr = h_chunk_traj[i];
        
        // Convert to pixel coordinates
        int px1 = (int)(((prev.x - final_map->min_x) / final_map->cell_size) * pixels_per_cell);
        int py1 = (int)(((final_map->max_y - prev.y) / final_map->cell_size) * pixels_per_cell);
        int px2 = (int)(((curr.x - final_map->min_x) / final_map->cell_size) * pixels_per_cell);
        int py2 = (int)(((final_map->max_y - curr.y) / final_map->cell_size) * pixels_per_cell);
        
        cv::line(map_img, cv::Point(px1, py1), cv::Point(px2, py2), 
                cv::Scalar(255, 0, 0), 2, cv::LINE_AA);  // Blue line
    }
    
    // Draw start and end points
    if (num_chunks > 0) {
        Vec3 start = h_chunk_traj[0];
        Vec3 end = h_chunk_traj[num_chunks - 1];
        
        int start_x = (int)(((start.x - final_map->min_x) / final_map->cell_size) * pixels_per_cell);
        int start_y = (int)(((final_map->max_y - start.y) / final_map->cell_size) * pixels_per_cell);
        int end_x = (int)(((end.x - final_map->min_x) / final_map->cell_size) * pixels_per_cell);
        int end_y = (int)(((final_map->max_y - end.y) / final_map->cell_size) * pixels_per_cell);
        
        cv::circle(map_img, cv::Point(start_x, start_y), 5, cv::Scalar(0, 255, 0), -1);  // Green start
        cv::circle(map_img, cv::Point(end_x, end_y), 5, cv::Scalar(0, 0, 255), -1);      // Red end
    }
    
    delete[] h_chunk_traj;
    
    cv::namedWindow("Final Baked Map", cv::WINDOW_AUTOSIZE);
    cv::imshow("Final Baked Map", map_img);
    cv::waitKey(0);
    
    delete final_map;

    cv::destroyAllWindows();

    return 0;
}
