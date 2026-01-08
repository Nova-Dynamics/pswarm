
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "ParticleSlam.h"

using json = nlohmann::json;
using namespace pswarm;

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

int localize()
{
    // Create CUDA events for benchmarking
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Timing accumulators
    float total_dr_step_time = 0.0f;
    float total_prune_time = 0.0f;
    float total_ingest_time = 0.0f;
    float total_accumulate_time = 0.0f;
    float total_resample_time = 0.0f;
    float total_memcpy_time = 0.0f;
    float total_measurement_time = 0.0f;
    int dr_step_count = 0;
    int measurement_count = 0;
    
    // Load JSON data
    std::cout << "Loading ./examples/data/partial_run.json..." << std::endl;
    std::ifstream json_file("./examples/data/partial_run.json");
    if (!json_file.is_open()) {
        fprintf(stderr, "Failed to open ./examples/data/partial_run.json!\n");
        return 1;
    }
    
    json data;
    json_file >> data;
    json_file.close();
    
    std::cout << "Loaded " << data.size() << " entries from JSON" << std::endl;

    // Map* map = load_map_from_file("big_boi.bin");
    Map* map = load_map_from_file("./examples/data/baked_map.bin");
    if (map == nullptr) {
        std::cerr << "Failed to load map from baked_map.bin" << std::endl;
        return 1;
    }

    // Create ParticleSlam instance with parameters
    const int NUM_PARTICLES = 5000;
    const int MAX_TRAJECTORY_LENGTH = 300;  // 10 seconds at 30 Hz
    const int MAX_CHUNK_LENGTH = 60;
    
    ParticleSlam mcl_slam(NUM_PARTICLES, MAX_TRAJECTORY_LENGTH, MAX_CHUNK_LENGTH);
    
    std::cout << "Initializing MCL SLAM with " << NUM_PARTICLES << " particles." << std::endl;
    // Initialize the particle filter
    mcl_slam.init();

    std::cout << "Setting global map for MCL SLAM." << std::endl;  
    
    mcl_slam.set_global_map(*map);
    
    std::cout << "Uniformly initializing particles." << std::endl;
    mcl_slam.uniform_initialize_particles();

    // Visualization setup
    cv::namedWindow("MCL Localization", cv::WINDOW_AUTOSIZE);
    const int img_size = 1000;
    const float view_range = 30.0f; // meters
    const float pixels_per_meter = img_size / view_range;
    int center_x = img_size / 2;
    int center_y = img_size / 2;
    
    // Pre-render the map background once for better performance
    std::cout << "Pre-rendering map background..." << std::endl;
    cv::Mat background(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Draw global map as background
    for (int my = 0; my < map->height; my++) {
        for (int mx = 0; mx < map->width; mx++) {
            int idx = my * map->width + mx;
            ChunkCell cell = map->cells[idx];
            
            int total_obs = cell.num_pos + cell.num_neg;
            if (total_obs > 1) {
                // Calculate occupancy probability
                float alpha = 1.0f + 0.7f * cell.num_pos;
                float beta = 1.5f + 0.4f * cell.num_neg;
                float prob = alpha / (alpha + beta);
                
                // Color from white (unoccupied) to black (occupied)
                int gray = (int)(255.0f * (1.0f - prob));
                cv::Scalar color(gray, gray, gray);
                
                // Convert map coordinates to pixel coordinates
                float cell_world_x = map->min_x + mx * map->cell_size;
                float cell_world_y = map->min_y + my * map->cell_size;
                
                int px = center_x + (int)(cell_world_x * pixels_per_meter);
                int py = center_y - (int)(cell_world_y * pixels_per_meter);
                
                int cell_pixels = std::max(1, (int)(map->cell_size * pixels_per_meter));
                
                if (px >= 0 && px < img_size && py >= 0 && py < img_size) {
                    cv::rectangle(background, 
                                cv::Point(px, py), 
                                cv::Point(px + cell_pixels, py + cell_pixels),
                                color, -1);
                }
            }
        }
    }
    
    // Draw coordinate axes on background
    cv::line(background, cv::Point(center_x, 0), cv::Point(center_x, img_size), cv::Scalar(200, 200, 200), 1);
    cv::line(background, cv::Point(0, center_y), cv::Point(img_size, center_y), cv::Scalar(200, 200, 200), 1);
    
    std::cout << "Map background ready. Starting visualization..." << std::endl;
    
    // Track state for dx_step calculation
    Vec3 prev_state = {0.0f, 0.0f, 0.0f};
    bool first_dr_step = true;
    
    // Allocate host memory for current particle states only
    Particle* h_particles = new Particle[NUM_PARTICLES];
    
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
                
                // Benchmark apply_step
                cudaEventRecord(start);
                mcl_slam.apply_step(dx_step, ts, 0.1f, 0.003f);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float dr_step_ms = 0;
                cudaEventElapsedTime(&dr_step_ms, start, stop);
                total_dr_step_time += dr_step_ms;
                dr_step_count++;

                // Benchmark prune_particles_outside_map
                cudaEventRecord(start);
                mcl_slam.prune_particles_outside_map();
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float prune_ms = 0;
                cudaEventElapsedTime(&prune_ms, start, stop);
                total_prune_time += prune_ms;
                
                // Benchmark calculate_measurement
                cudaEventRecord(start);
                Measurement measurement = mcl_slam.calculate_measurement();
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float measurement_ms = 0;
                cudaEventElapsedTime(&measurement_ms, start, stop);
                total_measurement_time += measurement_ms;
                
                // Benchmark memcpy
                cudaEventRecord(start);
                mcl_slam.download_current_particle_states(h_particles);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float memcpy_ms = 0;
                cudaEventElapsedTime(&memcpy_ms, start, stop);
                total_memcpy_time += memcpy_ms;
                
                int current_timestep = mcl_slam.get_current_timestep();
                
                // Copy background instead of recreating it
                cv::Mat img = background.clone();
                
                // Draw particles as arrows
                for (int p = 0; p < NUM_PARTICLES; p++) {
                    Particle particle = h_particles[p];
                    
                    float x = particle.state.x;
                    float y = particle.state.y;
                    float theta = particle.state.z;
                    
                    int px = center_x + (int)(x * pixels_per_meter);
                    int py = center_y - (int)(y * pixels_per_meter);
                    
                    // Arrow length
                    float arrow_length = 0.3f; // 30cm arrow
                    int arrow_px = (int)(arrow_length * -sinf(-theta) * pixels_per_meter);
                    int arrow_py = -(int)(arrow_length * cosf(-theta) * pixels_per_meter);
                    
                    // Draw arrow (blue for particles)
                    cv::arrowedLine(img, cv::Point(px, py), cv::Point(px + arrow_px, py + arrow_py),
                                  cv::Scalar(255, 0, 0), 1, cv::LINE_AA, 0, 0.3);
                    
                    // Draw small circle at particle position
                    cv::circle(img, cv::Point(px, py), 2, cv::Scalar(0, 0, 255), -1);
                }
                
                // Draw mean pose as a larger arrow (blue if Gaussian, orange if not)
                {
                    float mean_x = measurement.mean.x;
                    float mean_y = measurement.mean.y;
                    float mean_theta = measurement.mean.z;
                    
                    int mean_px = center_x + (int)(mean_x * pixels_per_meter);
                    int mean_py = center_y - (int)(mean_y * pixels_per_meter);
                    
                    // Color: blue if Gaussian, orange if not
                    cv::Scalar arrow_color = measurement.is_gaussian ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
                    cv::Scalar circle_color = measurement.is_gaussian ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 140, 255);
                    
                    // Draw covariance ellipse (1-sigma) for x,y
                    float cov_xx = measurement.covariance(0, 0);
                    float cov_xy = measurement.covariance(0, 1);
                    float cov_yy = measurement.covariance(1, 1);
                    
                    // Compute eigenvalues for ellipse axes
                    float trace = cov_xx + cov_yy;
                    float det = cov_xx * cov_yy - cov_xy * cov_xy;
                    float discriminant = sqrtf(trace * trace / 4.0f - det);
                    float lambda1 = trace / 2.0f + discriminant;
                    float lambda2 = trace / 2.0f - discriminant;
                    
                    // Ellipse axes lengths (3-sigma)
                    float axis1 = sqrtf(lambda1) * 3 * pixels_per_meter;
                    float axis2 = sqrtf(lambda2) * 3 * pixels_per_meter;
                    
                    // Rotation angle
                    float angle_rad = 0.5f * atan2f(2.0f * cov_xy, cov_xx - cov_yy);
                    float angle_deg = angle_rad * 180.0f / 3.14159265359f;
                    
                    // Draw ellipse
                    cv::ellipse(img, cv::Point(mean_px, mean_py), 
                              cv::Size((int)axis1, (int)axis2), 
                              -angle_deg, 0, 360, arrow_color, 2, cv::LINE_AA);
                    
                    // Larger arrow for mean
                    float arrow_length = 0.6f; // 60cm arrow
                    int arrow_px = (int)(arrow_length * -sinf(-mean_theta) * pixels_per_meter);
                    int arrow_py = -(int)(arrow_length * cosf(-mean_theta) * pixels_per_meter);
                    
                    // Draw arrow with thicker line
                    cv::arrowedLine(img, cv::Point(mean_px, mean_py), 
                                  cv::Point(mean_px + arrow_px, mean_py + arrow_py),
                                  arrow_color, 3, cv::LINE_AA, 0, 0.3);
                    
                    // Draw larger circle at mean position
                    cv::circle(img, cv::Point(mean_px, mean_py), 5, circle_color, -1);
                }
                
                // Draw info
                cv::rectangle(img, cv::Point(5, 5), cv::Point(280, 90), cv::Scalar(255, 255, 255), -1);
                cv::putText(img, "Timestep: " + std::to_string(current_timestep), cv::Point(10, 30), 
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                cv::putText(img, "Particles: " + std::to_string(NUM_PARTICLES), cv::Point(10, 60), 
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                
                cv::imshow("MCL Localization", img);
                
                if (cv::waitKey(1) == 27) {
                    break;
                }
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

            // Ingest the chunk
            cudaEventRecord(start);
            int c_idx = mcl_slam.ingest_visual_measurement(h_chunk);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ingest_ms = 0;
            cudaEventElapsedTime(&ingest_ms, start, stop);
            total_ingest_time += ingest_ms;

            if (c_idx == -1) {
                std::cerr << "Failed to ingest chunk at timestamp " << ts << std::endl;
                continue;
            }

            // Benchmark accumulate_map_from_map
            cudaEventRecord(start);
            mcl_slam.accumulate_map_from_map(c_idx);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float accumulate_ms = 0;
            cudaEventElapsedTime(&accumulate_ms, start, stop);
            total_accumulate_time += accumulate_ms;

            // Benchmark evaluate_and_resample
            cudaEventRecord(start);
            mcl_slam.evaluate_and_resample(c_idx);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float resample_ms = 0;
            cudaEventElapsedTime(&resample_ms, start, stop);
            total_resample_time += resample_ms;
            measurement_count++;
        }
    }
    
    delete[] h_particles;
    
    // Print benchmark results
    std::cout << "\n========== BENCHMARK RESULTS ==========" << std::endl;
    std::cout << "DR Steps: " << dr_step_count << std::endl;
    std::cout << "  Average: " << (total_dr_step_time / dr_step_count) << " ms/step" << std::endl;
    std::cout << "\nPrune Particles: " << dr_step_count << std::endl;
    std::cout << "  Average: " << (total_prune_time / dr_step_count) << " ms/prune" << std::endl;
    std::cout << "\nCalculate Measurement: " << dr_step_count << std::endl;
    std::cout << "  Average: " << (total_measurement_time / dr_step_count) << " ms/calculation" << std::endl;
    std::cout << "\nMemcpy (D2H): " << dr_step_count << std::endl;
    std::cout << "  Average: " << (total_memcpy_time / dr_step_count) << " ms/copy" << std::endl;
    std::cout << "\nMap Measurements: " << measurement_count << std::endl;
    std::cout << "  Ingest avg: " << (total_ingest_time / measurement_count) << " ms/measurement" << std::endl;
    std::cout << "  Accumulate avg: " << (total_accumulate_time / measurement_count) << " ms/accumulate" << std::endl;
    std::cout << "  Resample avg: " << (total_resample_time / measurement_count) << " ms/resample" << std::endl;
    std::cout << "\nTotal Pipeline Time: " << (total_dr_step_time + total_prune_time + total_ingest_time + total_accumulate_time + total_resample_time) << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Cleanup CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Bake and visualize updated global map
    std::cout << "Baking updated global map from best particle..." << std::endl;
    Map* updated_map = mcl_slam.bake_global_map_best_particle();

    save_map_to_file(updated_map, "updated_global_map.bin");
    
    if (updated_map != nullptr) {
        // Create image for map visualization
        int pixels_per_cell = 5;
        int map_img_width = updated_map->width * pixels_per_cell;
        int map_img_height = updated_map->height * pixels_per_cell;
        cv::Mat map_img(map_img_height, map_img_width, CV_8UC3, cv::Scalar(255, 255, 255));
        
        // Draw each cell with color
        for (int y = 0; y < updated_map->height; y++) {
            for (int x = 0; x < updated_map->width; x++) {
                int idx = y * updated_map->width + x;
                ChunkCell cell = updated_map->cells[idx];
                
                int total_obs = cell.num_pos + cell.num_neg;
                if (total_obs > 1) {
                    // Calculate occupancy probability
                    float alpha = 1.0f + 0.7f * cell.num_pos;
                    float beta = 1.5f + 0.4f * cell.num_neg;
                    float prob = alpha / (alpha + beta);
                    
                    // Color using HSL interpolation from black to orange
                    cv::Scalar start(0, 0, 0);       // Black
                    cv::Scalar stop(0, 165, 255);    // Orange (BGR)
                    cv::Scalar color = interpolate_hsl(start, stop, prob);
                    
                    cv::rectangle(map_img, 
                                cv::Point(x * pixels_per_cell, (updated_map->height - 1 - y) * pixels_per_cell), 
                                cv::Point((x + 1) * pixels_per_cell, (updated_map->height - y) * pixels_per_cell),
                                color, -1);
                }
            }
        }
        
        cv::namedWindow("Updated Global Map", cv::WINDOW_AUTOSIZE);
        cv::imshow("Updated Global Map", map_img);
        std::cout << "Updated map displayed. Press any key to close..." << std::endl;
        cv::waitKey(0);
        
        delete updated_map;
    }
    
    delete map;
    
    cv::destroyAllWindows();

    return 0;
}

int map()
{
     // Create ParticleSlam instance with parameters
    const int NUM_PARTICLES = 1000;
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
    std::cout << "Loading ./examples/data/full_run.json..." << std::endl;
    std::ifstream json_file("./examples/data/full_run.json");
    if (!json_file.is_open()) {
        fprintf(stderr, "Failed to open ./examples/data/full_run.json!\n");
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
                // SLAM noise parameters: small noise for mapping (uses defaults)
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
            int c_idx = slam.ingest_visual_measurement(h_chunk);

            if (c_idx == -1) {
                std::cerr << "Failed to ingest chunk at timestamp " << ts << std::endl;
                continue;
            }

            slam.accumulate_map_from_trajectories(c_idx);

            slam.evaluate_and_resample(c_idx);
            
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
    Map* final_map = slam.bake_best_particle_map();
    
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
    
    bool success = save_map_to_file(final_map, "./examples/data/baked_map.bin");
    if (!success) {
        std::cerr << "Failed to save final baked map to file!" << std::endl;
    } else {
        std::cout << "Final baked map saved to ./examples/data/baked_map.bin" << std::endl;
    }

    delete[] h_chunk_traj;
    
    cv::namedWindow("Final Baked Map", cv::WINDOW_AUTOSIZE);
    cv::imshow("Final Baked Map", map_img);
    cv::waitKey(0);
    
    delete final_map;

    cv::destroyAllWindows();

    return 0;
}

int visualize() {
    // Load map from file
    std::cout << "Loading ./map-data-m1.bin..." << std::endl;
    Map* map = load_map_from_file("./map-data-m2.bin");
    if (map == nullptr) {
        std::cerr << "Failed to load map from ./map-data-m1.bin" << std::endl;
        return 1;
    }
    
    std::cout << "Map loaded successfully!" << std::endl;
    std::cout << "  Width: " << map->width << " cells" << std::endl;
    std::cout << "  Height: " << map->height << " cells" << std::endl;
    std::cout << "  Cell size: " << map->cell_size << " meters" << std::endl;
    std::cout << "  Bounds: X[" << map->min_x << ", " << map->max_x << "], Y[" << map->min_y << ", " << map->max_y << "]" << std::endl;
    
    // Create image for map visualization
    int pixels_per_cell = 5;
    int map_img_width = map->width * pixels_per_cell;
    int map_img_height = map->height * pixels_per_cell;
    cv::Mat map_img(map_img_height, map_img_width, CV_8UC3, cv::Scalar(255, 255, 255));  // White background
    
    std::cout << "Rendering map..." << std::endl;
    
    // Draw each cell
    for (int y = 0; y < map->height; y++) {
        for (int x = 0; x < map->width; x++) {
            int idx = y * map->width + x;
            ChunkCell cell = map->cells[idx];
            
            // Only color cells with sufficient observations
            int total_obs = cell.num_pos + cell.num_neg;
            if (total_obs > 1) {  // At least 2 observations
                // Calculate occupancy probability using Beta-Bernoulli model
                float alpha = 1.0f + 0.7f * cell.num_pos;
                float beta = 1.5f + 0.4f * cell.num_neg;
                float prob = alpha / (alpha + beta);

                //std::cout << "Cell (" << x << ", " << y << "): num_pos=" << unsigned(cell.num_pos)
                          //<< ", num_neg=" << unsigned(cell.num_neg) << ", prob=" << prob << std::endl;
                
                // Color using HSL interpolation from black to orange
                cv::Scalar start(0, 0, 0);       // Black
                cv::Scalar stop(0, 165, 255);    // Orange (BGR)
                cv::Scalar color = interpolate_hsl(start, stop, prob);
                
                cv::rectangle(map_img, 
                            cv::Point(x * pixels_per_cell, (map->height - 1 - y) * pixels_per_cell), 
                            cv::Point((x + 1) * pixels_per_cell, (map->height - y) * pixels_per_cell),
                            color, -1);
            }
        }
    }
    
    std::cout << "Map rendered. Displaying..." << std::endl;
    
    // Display the map
    cv::namedWindow("Map Visualization", cv::WINDOW_AUTOSIZE);
    cv::imshow("Map Visualization", map_img);
    
    std::cout << "Press any key to close the window..." << std::endl;
    cv::waitKey(0);
    
    // Cleanup
    delete map;
    cv::destroyAllWindows();
    
    std::cout << "Visualization complete." << std::endl;

    return 0;
}

int main()
{
    return localize();
    //return map();
    //return visualize();
}
