
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "rt_jr.hpp"

using json = nlohmann::json;

#define NUM_PARTICLES 2000

// 10 seconds at 30 Hz
#define MAX_TRAJECTORY_LENGTH 300

// 1 hr at 1 Hz
#define MAX_CHUNK_LENGTH 3600

#define CELL_SIZE_M 0.1f

#define POS_WEIGHT 0.7f
#define NEG_WEIGHT 0.4f

#define ALPHA_PRIOR 1.0f
#define BETA_PRIOR 1.5f

using namespace rt_jr;


struct Particle {
    Vec3 state;
    double timestamp;
};

struct ChunkCell {
    int16_t num_pos;
    int16_t num_neg;
};

struct Chunk {

    // 60x60 cells for 6m x 6m area with 0.1m cell size
    ChunkCell cells[60][60];
    double timestamp;

};

__global__ void initRandom(curandState* states, int num_particles, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void applyStep(Particle* particles, curandState* states, int num_particles, Vec3 dx_step, double step_timestamp, int current_timestep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Calculate positions in the ring buffer (trajectory-major: each row contains complete trajectory for one particle)
    int prev_pos = idx * MAX_TRAJECTORY_LENGTH + ((current_timestep - 1 + MAX_TRAJECTORY_LENGTH) % MAX_TRAJECTORY_LENGTH);
    int curr_pos = idx * MAX_TRAJECTORY_LENGTH + (current_timestep % MAX_TRAJECTORY_LENGTH);

    // Read previous state
    Particle prev_particle = particles[prev_pos];

    Mat2 P_R { cosf(-prev_particle.state.z), -sinf(-prev_particle.state.z),
               sinf(-prev_particle.state.z),  cosf(-prev_particle.state.z) };

    Vec2 mean_delta = { dx_step.x, dx_step.y };
    Vec2 particle_delta = P_R * mean_delta;

    float theta_noise = 0;
    
    // Add gaussian noise to particle_delta
    if (particle_delta.length() > 1e-8f) {
        float noise_x = curand_normal(&states[idx]) * 1.6e-3f + 1.0f;  // mean=1.0, std=1.6e-3
        float noise_y = curand_normal(&states[idx]) * 1.6e-3f + 1.0f;  // mean=1.0, std=1.6e-3
        
        particle_delta.x *= noise_x;
        particle_delta.y *= noise_y;
        theta_noise = curand_normal(&states[idx]) * 1e-3f;  // mean=0, std=1e-3
    }


    // Add gaussian noise to theta


    // Write new state to current timestep position
    particles[curr_pos].state.x = prev_particle.state.x + particle_delta.x;
    particles[curr_pos].state.y = prev_particle.state.y + particle_delta.y;
    particles[curr_pos].state.z = prev_particle.state.z + dx_step.z + theta_noise;
    particles[curr_pos].timestamp = step_timestamp;
}

__global__ void extractParticleStatesAtTimestep(Particle* particles, Vec3* chunk_states, int num_particles, int timestep, int chunk_index)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Read particle state at the given timestep
    int pos = idx * MAX_TRAJECTORY_LENGTH + (timestep % MAX_TRAJECTORY_LENGTH);
    Vec3 state = particles[pos].state;

    // Write to chunk_states array
    chunk_states[chunk_index * num_particles + idx] = state;
}


__global__ void accumulateMapForParticles(Vec3* chunk_states, Chunk* chunks, ChunkCell* cur_maps, int num_particles, int last_chunk_index)
{
    // Each block handles one particle, threads loop over cells
    int particle_idx = blockIdx.x;
    if (particle_idx >= num_particles) return;
    
    // Get the particle state for this chunk (shared across all threads in block)
    Vec3 state = chunk_states[last_chunk_index * num_particles + particle_idx];
    
    Pair<Mat2, Vec2> R_t = get_affine_tx_from_state(state);
    Mat2 R = R_t.first;
    Vec2 t = R_t.second;
    
    // Access the prediction chunk
    Chunk prediction = chunks[last_chunk_index];
    
    // Grid-stride loop over cells (each thread processes multiple cells)
    for (int cell_idx = threadIdx.x; cell_idx < 3600; cell_idx += blockDim.x) {
        int xi = cell_idx / 60;
        int yi = cell_idx % 60;
        ChunkCell pred_cell = prediction.cells[xi][yi];
        
        // Skip empty prediction cells
        if (pred_cell.num_pos == 0 && pred_cell.num_neg == 0) {
            int map_idx = particle_idx * 3600 + cell_idx;
            cur_maps[map_idx].num_pos = 0;
            cur_maps[map_idx].num_neg = 0;
            continue;  // Skip to next cell, don't exit the entire thread!
        }
        
        // Accumulate contributions from previous chunks
        int accumulated_pos = 0;
        int accumulated_neg = 0;
        
        // Unquantize cell position
        Vec2 cell_pos = {(float)xi * CELL_SIZE_M - 3.0f, (float)yi * CELL_SIZE_M - 3.0f};
        
        // Loop through previous chunk states
        for (int chunk_i = 0; chunk_i < last_chunk_index; chunk_i++) {
            Vec3 other_state = chunk_states[chunk_i * num_particles + particle_idx];
            float dx = other_state.x - state.x;
            float dy = other_state.y - state.y;
            float dist_sq = dx * dx + dy * dy;
            
            if (dist_sq <= 36.0f) { // 6m radius
                Pair<Mat2, Vec2> R_t_other = get_affine_tx_from_state(other_state);
                Mat2 Rc = R_t_other.first;
                Vec2 tc = R_t_other.second;
                
                Vec2 m = tc - t;
                Vec2 q_body = Rc.transpose() * cell_pos;
                Vec2 pt_body = R * (m + q_body);
                
                // Quantize to find corresponding cell in other chunk
                int px = __float2int_rn((pt_body.x + 3.0f) / CELL_SIZE_M);
                int py = __float2int_rn((pt_body.y + 3.0f) / CELL_SIZE_M);
                
                if (px >= 0 && px < 60 && py >= 0 && py < 60) {
                    // Accumulate from the previous chunk's cell (not the prediction chunk!)
                    accumulated_pos += chunks[chunk_i].cells[px][py].num_pos;
                    accumulated_neg += chunks[chunk_i].cells[px][py].num_neg;
                }
            }
        }
    
        // Store accumulated map
        int map_idx = particle_idx * 3600 + cell_idx;
        cur_maps[map_idx].num_pos = accumulated_pos;
        cur_maps[map_idx].num_neg = accumulated_neg;
    }
}

// Compute log-likelihoods - parallelized over (particle, xi, yi)
__global__ void computeLogLikelihoods(ChunkCell* cur_maps, Chunk* chunks, float* log_likelihoods, int num_particles, int last_chunk_index)
{
    // Each block handles one particle, threads loop over cells
    int particle_idx = blockIdx.x;
    if (particle_idx >= num_particles) return;
    
    // Access prediction chunk
    Chunk prediction = chunks[last_chunk_index];
    
    // Grid-stride loop over cells (each thread processes multiple cells)
    for (int cell_idx = threadIdx.x; cell_idx < 3600; cell_idx += blockDim.x) {
        int xi = cell_idx / 60;
        int yi = cell_idx % 60;

        int map_idx = particle_idx * 3600 + cell_idx;
        ChunkCell cur_cell = cur_maps[map_idx];
        ChunkCell pred_cell = prediction.cells[xi][yi];

        // Tom help? Is this correct?
        if (pred_cell.num_pos == 0 && pred_cell.num_neg == 0) {
            continue; // Skip empty prediction cells
        }
        
        // Calculate log-likelihood contribution for this cell
        float p_alpha = ALPHA_PRIOR + POS_WEIGHT * pred_cell.num_pos;
        float p_beta = BETA_PRIOR + NEG_WEIGHT * pred_cell.num_neg;
        float p_theta = roundf((p_alpha) / (p_alpha + p_beta));
        // float p_theta = (p_alpha) / (p_alpha + p_beta);
        
        float alpha = ALPHA_PRIOR + POS_WEIGHT * cur_cell.num_pos;
        float beta = BETA_PRIOR + NEG_WEIGHT * cur_cell.num_neg;
        float theta = (alpha) / (alpha + beta);
    
        float cell_ll = logf(p_theta * theta + (1.0f - p_theta) * (1.0f - theta));
        
        // Atomically add to particle's total log-likelihood
        atomicAdd(&log_likelihoods[particle_idx], cell_ll);
    }
}

__global__ void calculateScores(float* log_likelihoods, float* scores_raw, float* scores, int num_particles, float min_ll, float max_ll)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (max_ll - min_ll < 1e-3f) {
        // Uniform
        scores_raw[idx] = 1.0f / num_particles;
    } else {
        // Normalize by range
        scores_raw[idx] = (log_likelihoods[idx] - min_ll) / (max_ll - min_ll);
    }
}

__global__ void normalizeScoresBySum(float* scores_raw, float* scores, int num_particles, float sum_scores_raw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    scores[idx] = scores_raw[idx] / sum_scores_raw;
}

__global__ void calculateCumulativeSum(float* scores, float* cumsum, int num_particles)
{
    // Simple sequential cumulative sum (could be optimized with parallel scan)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cumsum[0] = scores[0];
        for (int i = 1; i < num_particles; i++) {
            cumsum[i] = cumsum[i - 1] + scores[i];
        }
    }
}

__global__ void susResample(Particle* particles, Particle* particles_swap, Vec3* chunk_states, Vec3* chunk_states_swap, float* cumsum, int num_particles, int num_chunks, float initial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    // Calculate sample value for this particle
    float val = initial + ((float)i / (float)num_particles);

    // Find the source particle index using binary search through cumsum
    int idx = 0;
    for (int j = 0; j < num_particles; j++) {
        if (val <= cumsum[j]) {
            idx = j;
            break;
        }
    }

    // Copy entire trajectory from source particle to destination
    for (int t = 0; t < MAX_TRAJECTORY_LENGTH; t++) {
        int src_pos = idx * MAX_TRAJECTORY_LENGTH + t;
        int dst_pos = i * MAX_TRAJECTORY_LENGTH + t;
        particles_swap[dst_pos] = particles[src_pos];
    }
    
    // Copy all chunk states from source particle to destination
    for (int chunk_i = 0; chunk_i < num_chunks; chunk_i++) {
        int src_pos = chunk_i * num_particles + idx;
        int dst_pos = chunk_i * num_particles + i;
        chunk_states_swap[dst_pos] = chunk_states[src_pos];
    }
}

int main()
{
    Particle* d_particles;
    Particle* d_particles_swap;
    Vec3* d_chunk_states;
    Vec3* d_chunk_states_swap;
    Chunk* d_chunks;
    curandState* d_randStates;
    float* d_log_likelihoods;
    float* d_scores_raw;
    float* d_scores;
    float* d_cumsum;
    cudaError_t cudaStatus;

    // Allocate device memory for particles
    cudaStatus = cudaMalloc((void **)&d_particles, NUM_PARTICLES * MAX_TRAJECTORY_LENGTH * sizeof(Particle));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate swap buffer for resampling
    cudaStatus = cudaMalloc((void **)&d_particles_swap, NUM_PARTICLES * MAX_TRAJECTORY_LENGTH * sizeof(Particle));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for particles_swap failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void **)&d_particles_swap, NUM_PARTICLES * MAX_TRAJECTORY_LENGTH * sizeof(Particle));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Allocate chunk memory
    cudaStatus = cudaMalloc((void **)&d_chunks, MAX_CHUNK_LENGTH * sizeof(Chunk));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for chunks failed!");
        goto Error;
    }

    std::cout << "sizeof(Chunk) = " << sizeof(Chunk) / 1024.0 << " KB" << std::endl;
    std::cout << "Allocated " << MAX_CHUNK_LENGTH * sizeof(Chunk) / (1024.0 * 1024.0) << " MB for chunks." << std::endl;

    // Allocate device memory for chunk states
    cudaStatus = cudaMalloc((void **)&d_chunk_states, MAX_CHUNK_LENGTH * NUM_PARTICLES * sizeof(Vec3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for chunk states failed!");
        goto Error;
    }

    std::cout << "Allocated " << MAX_CHUNK_LENGTH * NUM_PARTICLES * sizeof(Vec3) / (1024.0 * 1024.0) << " MB for chunk states." << std::endl;

    // Allocate swap buffer for chunk states resampling
    cudaStatus = cudaMalloc((void **)&d_chunk_states_swap, MAX_CHUNK_LENGTH * NUM_PARTICLES * sizeof(Vec3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for chunk_states_swap failed!");
        goto Error;
    }

    // Allocate random states
    cudaStatus = cudaMalloc((void **)&d_randStates, NUM_PARTICLES * sizeof(curandState));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for random states failed!");
        goto Error;
    }

    // Allocate scoring arrays
    cudaStatus = cudaMalloc((void **)&d_log_likelihoods, NUM_PARTICLES * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for log_likelihoods failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&d_scores_raw, NUM_PARTICLES * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for scores_raw failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&d_scores, NUM_PARTICLES * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for scores failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&d_cumsum, NUM_PARTICLES * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for cumsum failed!");
        goto Error;
    }

    // Allocate memory for accumulated maps (per-particle)
    ChunkCell* d_cur_maps;
    cudaStatus = cudaMalloc((void **)&d_cur_maps, NUM_PARTICLES * 60 * 60 * sizeof(ChunkCell));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc for cur_maps failed!");
        goto Error;
    }
    std::cout << "Allocated " << NUM_PARTICLES * 60 * 60 * sizeof(ChunkCell) / (1024.0 * 1024.0) << " MB for accumulated maps." << std::endl;

    // Memset to 0
    cudaStatus = cudaMemset((void *)d_particles, 0, NUM_PARTICLES * MAX_TRAJECTORY_LENGTH * sizeof(Particle));

    // Initialize random states
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initRandom<<<numBlocks, blockSize>>>(d_randStates, NUM_PARTICLES, 1234ULL);
    cudaDeviceSynchronize();

    std::cout << "Allocated " << 2 * NUM_PARTICLES * MAX_TRAJECTORY_LENGTH * sizeof(Vec3) / (1024.0 * 1024.0) << " MB for particle trajectories." << std::endl;
    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Visualization and simulation in a separate block to avoid goto issues
    {
        // Create visualization window and image
        cv::namedWindow("Particle SLAM", cv::WINDOW_AUTOSIZE);
        const int img_size = 1000;
        const float view_range = 30.0f; // meters
        const float pixels_per_meter = img_size / view_range;
        
        // Create image once - will accumulate trajectories
        cv::Mat img(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
        int center_x = img_size / 2;
        int center_y = img_size / 2;
        
        // Draw coordinate axes
        cv::line(img, cv::Point(center_x, 0), cv::Point(center_x, img_size), cv::Scalar(200, 200, 200), 1);
        cv::line(img, cv::Point(0, center_y), cv::Point(img_size, center_y), cv::Scalar(200, 200, 200), 1);
        
        // Allocate host memory for visualization
        float* h_scores = new float[NUM_PARTICLES];
        for (int i = 0; i < NUM_PARTICLES; i++) h_scores[i] = 0.5f; // Initialize to mid-range
        
        // Allocate host memory for chunk states
        Vec3* h_chunk_states = new Vec3[MAX_CHUNK_LENGTH * NUM_PARTICLES];

        // Load JSON data
        std::cout << "Loading bigboi_munged.json..." << std::endl;
        std::ifstream json_file("bigboi_munged.json");
        if (!json_file.is_open()) {
            fprintf(stderr, "Failed to open bigboi_munged.json!");
            delete[] h_scores;
            delete[] h_chunk_states;
            goto Error;
        }
        
        json data;
        json_file >> data;
        json_file.close();
        
        std::cout << "Loaded " << data.size() << " entries from JSON" << std::endl;

        // Track state for dx_step calculation
        int timestep = 0;
        int cur_chunk_index = 0;
        Vec3 prev_state = {0.0f, 0.0f, 0.0f};
        bool first_dr_step = true;
        double last_chunk_timestamp = -1.0;

        // Process all entries from JSON
        for (const auto& entry : data) {
            std::string type = entry["type"];
            double ts = entry["ts"];

            if (type == "dr_step") {
                // Extract current state from the dr_step
                Vec3 cur_state;
                cur_state.x = entry["value"]["x"][0];
                cur_state.y = entry["value"]["x"][1];
                cur_state.z = entry["value"]["x"][2];

                if (!first_dr_step) {
                    // Calculate dx_step from prev_state to cur_state
                    float theta = -prev_state.z;
                    Mat2 R = get_R_from_theta(theta);
                    Vec2 state_delta = {cur_state.x - prev_state.x, cur_state.y - prev_state.y};
                    Vec2 mean_delta = R.transpose() * state_delta;
                    float theta_delta = cur_state.z - prev_state.z;

                    Vec3 dx_step = {mean_delta.x, mean_delta.y, theta_delta};

                    timestep += 1;
                    cudaEventRecord(start);
                    applyStep<<<numBlocks, blockSize>>>(d_particles, d_randStates, NUM_PARTICLES, dx_step, ts, timestep);
                    cudaEventRecord(stop);
                    cudaEventSynchronize(stop);
                    float milliseconds = 0;
                    cudaEventElapsedTime(&milliseconds, start, stop);
                    if (timestep % 100 == 0) {
                        std::cout << "  applyStep: " << milliseconds << " ms" << std::endl;
                    }
                } else {
                    first_dr_step = false;
                }

                prev_state = cur_state;

            } else if (type == "map_measurement") { 

                 //Ensure the new chunk timestamp is greater than the last
                if (ts <= last_chunk_timestamp) {
                    fprintf(stderr, "Warning: Chunk timestamp %f is not greater than last chunk timestamp %f\n", ts, last_chunk_timestamp);
                    continue;
                }

                // Pull a single particle's trajectory to find the timestep that matches this chunk timestamp
                Particle single_particle_traj[MAX_TRAJECTORY_LENGTH];
                cudaStatus = cudaMemcpy(single_particle_traj, d_particles, MAX_TRAJECTORY_LENGTH * sizeof(Particle), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for single particle trajectory failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Find the timestep index where the timestamp matches or is just after the chunk timestamp
                int chunk_timestep = -1;
                for (int t = 0; t < MAX_TRAJECTORY_LENGTH; t++) {
                    if (single_particle_traj[t].timestamp >= ts) {
                        chunk_timestep = t;
                        break;
                    }
                }

                if (chunk_timestep == -1) {
                    fprintf(stderr, "Could not find matching timestep for chunk timestamp %f\n", ts);
                    continue;
                }

                // Package map measurement into chunk structure
                Chunk h_chunk;
                h_chunk.timestamp = ts;
                
                // Initialize all cells to zero
                for (int i = 0; i < 60; i++) {
                    for (int j = 0; j < 60; j++) {
                        h_chunk.cells[i][j].num_pos = 0;
                        h_chunk.cells[i][j].num_neg = 0;
                    }
                }

                // Fill in cells from JSON
                const auto& cells = entry["value"]["cells"];
                for (int i = 0; i < 60; i++) {
                    for (int j = 0; j < 60; j++) {
                        if (!cells[i][j].is_null()) {
                            h_chunk.cells[i][j].num_pos = cells[i][j]["num_pos"];
                            h_chunk.cells[i][j].num_neg = cells[i][j]["num_neg"];
                        }
                    }
                }

                // Upload chunk to device
                cudaStatus = cudaMemcpy(&d_chunks[cur_chunk_index], &h_chunk, sizeof(Chunk), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for chunk upload failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Extract particle states at chunk timestep and upload to device
                cudaEventRecord(start);
                extractParticleStatesAtTimestep<<<numBlocks, blockSize>>>(d_particles, d_chunk_states, NUM_PARTICLES, chunk_timestep, cur_chunk_index);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_extract = 0;
                cudaEventElapsedTime(&ms_extract, start, stop);
                std::cout << "  extractParticleStatesAtTimestep: " << ms_extract << " ms" << std::endl;

                std::cout << "Processed chunk " << cur_chunk_index << " at timestep " << chunk_timestep << " (ts=" << ts << ")" << std::endl;

                // Initialize log_likelihoods to zero
                cudaMemset(d_log_likelihoods, 0, NUM_PARTICLES * sizeof(float));

                // Step 1: Accumulate maps for all particles (one block per particle)
                // 500 blocks, 256 threads per block, each thread handles multiple cells
                dim3 blockDim(256);
                dim3 gridDim(NUM_PARTICLES);  // 500 blocks
                
                cudaEventRecord(start);
                accumulateMapForParticles<<<gridDim, blockDim>>>(d_chunk_states, d_chunks, d_cur_maps, NUM_PARTICLES, cur_chunk_index);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_accumulate = 0;
                cudaEventElapsedTime(&ms_accumulate, start, stop);
                std::cout << "  accumulateMapForParticles: " << ms_accumulate << " ms" << std::endl;

                // Step 2: Compute log-likelihoods (parallelize over particles × cells)
                cudaEventRecord(start);
                computeLogLikelihoods<<<gridDim, blockDim>>>(d_cur_maps, d_chunks, d_log_likelihoods, NUM_PARTICLES, cur_chunk_index);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_likelihood = 0;
                cudaEventElapsedTime(&ms_likelihood, start, stop);
                std::cout << "  computeLogLikelihoods: " << ms_likelihood << " ms" << std::endl;

                // Update tracking variables (AFTER evaluation)
                last_chunk_timestamp = ts;
                cur_chunk_index += 1;

                // Download log-likelihoods
                float h_log_likelihoods[NUM_PARTICLES];
                cudaStatus = cudaMemcpy(h_log_likelihoods, d_log_likelihoods, NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for log_likelihoods download failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Find min/max on CPU (could be optimized with reduction kernel)
                float min_ll = FLT_MAX;
                float max_ll = -FLT_MAX;
                for (int i = 0; i < NUM_PARTICLES; i++) {
                    if (h_log_likelihoods[i] < min_ll) min_ll = h_log_likelihoods[i];
                    if (h_log_likelihoods[i] > max_ll) max_ll = h_log_likelihoods[i];
                }

                // Upload log_likelihoods back to device for score calculation
                cudaStatus = cudaMemcpy(d_log_likelihoods, h_log_likelihoods, NUM_PARTICLES * sizeof(float), cudaMemcpyHostToDevice);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for log_likelihoods upload failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Calculate scores_raw
                cudaEventRecord(start);
                calculateScores<<<numBlocks, blockSize>>>(d_log_likelihoods, d_scores_raw, d_scores, NUM_PARTICLES, min_ll, max_ll);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_calcscores = 0;
                cudaEventElapsedTime(&ms_calcscores, start, stop);
                std::cout << "  calculateScores: " << ms_calcscores << " ms" << std::endl;

                // Download scores_raw to compute sum
                float h_scores_raw[NUM_PARTICLES];
                cudaStatus = cudaMemcpy(h_scores_raw, d_scores_raw, NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for scores_raw download failed!");
                    cudaFree(d_scores_raw);
                    cudaFree(d_scores);
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Calculate sum of scores_raw
                float sum_scores_raw = 0.0f;
                for (int i = 0; i < NUM_PARTICLES; i++) {
                    sum_scores_raw += h_scores_raw[i];
                }

                // Normalize by sum to get final scores
                cudaEventRecord(start);
                normalizeScoresBySum<<<numBlocks, blockSize>>>(d_scores_raw, d_scores, NUM_PARTICLES, sum_scores_raw);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_normalize = 0;
                cudaEventElapsedTime(&ms_normalize, start, stop);
                std::cout << "  normalizeScoresBySum: " << ms_normalize << " ms" << std::endl;

                // Download normalized scores for visualization
                cudaStatus = cudaMemcpy(h_scores, d_scores_raw, NUM_PARTICLES * sizeof(float), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for scores download failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Calculate cumulative sum for resampling
                cudaEventRecord(start);
                calculateCumulativeSum<<<1, 1>>>(d_scores, d_cumsum, NUM_PARTICLES);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_cumsum = 0;
                cudaEventElapsedTime(&ms_cumsum, start, stop);
                std::cout << "  calculateCumulativeSum: " << ms_cumsum << " ms" << std::endl;

                // for (int i = 0; i < 3; i++) {
     
                
                // Generate initial random offset for SUS
                float initial = ((float)rand() / (float)RAND_MAX) * (1.0f / (float)NUM_PARTICLES);

                // Perform SUS resampling (copies both particles and chunk_states)
                cudaEventRecord(start);
                susResample<<<numBlocks, blockSize>>>(d_particles, d_particles_swap, d_chunk_states, d_chunk_states_swap, d_cumsum, NUM_PARTICLES, cur_chunk_index, initial);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float ms_resample = 0;
                cudaEventElapsedTime(&ms_resample, start, stop);
                std::cout << "  susResample: " << ms_resample << " ms" << std::endl;

                // Swap pointers for both particles and chunk states
                Particle* temp = d_particles;
                d_particles = d_particles_swap;
                d_particles_swap = temp;
                
                Vec3* temp_chunks = d_chunk_states;
                d_chunk_states = d_chunk_states_swap;
                d_chunk_states_swap = temp_chunks;

                std::cout << "Resampled particles with initial=" << initial << std::endl;
                // }

                // Print statistics
                float avg_ll = 0.0f;
                for (int i = 0; i < NUM_PARTICLES; i++) {
                    avg_ll += h_log_likelihoods[i];
                }
                avg_ll /= NUM_PARTICLES;

                if (max_ll - min_ll < 1e-3f) {
                    std::cout << "Chunk " << (cur_chunk_index - 1) << ": Uniform scores (range too small)" << std::endl;
                } else {
                    std::cout << "Chunk " << (cur_chunk_index - 1) << ": LL range [" << min_ll << ", " << max_ll
                              << "], avg=" << avg_ll << ", sum_scores_raw=" << sum_scores_raw << std::endl;
                }

                // Visualization update after each chunk
                // Download chunk states
                cudaStatus = cudaMemcpy(h_chunk_states, d_chunk_states, cur_chunk_index * NUM_PARTICLES * sizeof(Vec3), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy for chunk_states failed!");
                    delete[] h_scores;
                    delete[] h_chunk_states;
                    goto Error;
                }

                // Clear image for this frame
                img = cv::Mat(img_size, img_size, CV_8UC3, cv::Scalar(255, 255, 255));
                
                // Draw coordinate axes
                cv::line(img, cv::Point(center_x, 0), cv::Point(center_x, img_size), cv::Scalar(200, 200, 200), 1);
                cv::line(img, cv::Point(0, center_y), cv::Point(img_size, center_y), cv::Scalar(200, 200, 200), 1);

                // Draw chunk_states as lines for each particle
                for (int p = 0; p < NUM_PARTICLES; p++) {
                    float score = h_scores[p];
                    
                    // Particle color: Red (BGR: 0,0,255) at score=0, Green (BGR: 0,255,0) at score=1
                    int b = 0;
                    int g = (int)(255 * score);
                    int r = (int)(255 * (1.0f - score));
                    cv::Scalar color(b, g, r);
                    
                    // Draw lines connecting chunk states
                    for (int chunk_i = 1; chunk_i < cur_chunk_index; chunk_i++) {
                        Vec3 prev_state = h_chunk_states[(chunk_i - 1) * NUM_PARTICLES + p];
                        Vec3 curr_state = h_chunk_states[chunk_i * NUM_PARTICLES + p];
                        
                        // Convert to pixel coordinates
                        int px1 = center_x + (int)(prev_state.x * pixels_per_meter);
                        int py1 = center_y - (int)(prev_state.y * pixels_per_meter);
                        int px2 = center_x + (int)(curr_state.x * pixels_per_meter);
                        int py2 = center_y - (int)(curr_state.y * pixels_per_meter);
                        
                        cv::line(img, cv::Point(px1, py1), cv::Point(px2, py2), color, 1, cv::LINE_AA);
                    }
                }

                // Add text info
                cv::rectangle(img, cv::Point(5, 5), cv::Point(250, 90), cv::Scalar(255, 255, 255), -1);
                cv::putText(img, "Timestep: " + std::to_string(timestep), cv::Point(10, 30), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
                cv::putText(img, "Particles: " + std::to_string(NUM_PARTICLES), cv::Point(10, 60), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

                cv::imshow("Particle SLAM", img);
                
                if (cv::waitKey(1) == 27) { // ESC key to exit
                    break;
                }
            }
        }

        std::cout << "Applied " << timestep << " steps to particles." << std::endl;

        delete[] h_scores;
        delete[] h_chunk_states;
        cv::destroyAllWindows();
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

Error:
    cudaFree(d_particles);
    cudaFree(d_particles_swap);
    cudaFree(d_randStates);
    cudaFree(d_chunks);
    cudaFree(d_chunk_states);
    cudaFree(d_chunk_states_swap);
    cudaFree(d_log_likelihoods);
    cudaFree(d_scores_raw);
    cudaFree(d_scores);
    cudaFree(d_cumsum);
    cudaFree(d_cur_maps);

    return 0;
}
