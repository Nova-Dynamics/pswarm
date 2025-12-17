#include "ParticleSlam.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>
#include <cfloat>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void initRandom(curandState* states, int num_particles, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void applyStepKernel(Particle* particles, curandState* states, KernelParams params, 
                                Vec3 dx_step, double step_timestamp, int current_timestep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_particles) return;

    int prev_pos = idx * params.max_trajectory_length + 
                   ((current_timestep - 1 + params.max_trajectory_length) % params.max_trajectory_length);
    int curr_pos = idx * params.max_trajectory_length + (current_timestep % params.max_trajectory_length);

    Particle prev_particle = particles[prev_pos];

    Mat2 P_R { cosf(-prev_particle.state.z), -sinf(-prev_particle.state.z),
               sinf(-prev_particle.state.z),  cosf(-prev_particle.state.z) };

    Vec2 mean_delta = { dx_step.x, dx_step.y };
    Vec2 particle_delta = P_R * mean_delta;

    float theta_noise = 0;
    
    if (particle_delta.length() > 1e-8f) {
        float noise_x = curand_normal(&states[idx]) * 1.6e-3f + 1.0f;
        float noise_y = curand_normal(&states[idx]) * 1.6e-3f + 1.0f;
        
        particle_delta.x *= noise_x;
        particle_delta.y *= noise_y;
        theta_noise = curand_normal(&states[idx]) * 1e-3f;
    }

    particles[curr_pos].state.x = prev_particle.state.x + particle_delta.x;
    particles[curr_pos].state.y = prev_particle.state.y + particle_delta.y;
    particles[curr_pos].state.z = prev_particle.state.z + dx_step.z + theta_noise;
    particles[curr_pos].timestamp = step_timestamp;
}

__global__ void extractParticleStatesAtTimestep(Particle* particles, Vec3* chunk_states, 
                                                KernelParams params, int timestep, int chunk_index)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_particles) return;

    int pos = idx * params.max_trajectory_length + (timestep % params.max_trajectory_length);
    Vec3 state = particles[pos].state;

    chunk_states[chunk_index * params.num_particles + idx] = state;
}

__global__ void accumulateMapForParticles(Vec3* chunk_states, Chunk* chunks, ChunkCell* cur_maps, 
                                         KernelParams params, int last_chunk_index)
{
    int particle_idx = blockIdx.x;
    if (particle_idx >= params.num_particles) return;
    
    Vec3 state = chunk_states[last_chunk_index * params.num_particles + particle_idx];
    
    Pair<Mat2, Vec2> R_t = get_affine_tx_from_state(state);
    Mat2 R = R_t.first;
    Vec2 t = R_t.second;
    
    Chunk prediction = chunks[last_chunk_index];
    
    for (int cell_idx = threadIdx.x; cell_idx < 3600; cell_idx += blockDim.x) {
        int xi = cell_idx / 60;
        int yi = cell_idx % 60;
        ChunkCell pred_cell = prediction.cells[xi][yi];
        
        if (pred_cell.num_pos == 0 && pred_cell.num_neg == 0) {
            int map_idx = particle_idx * 3600 + cell_idx;
            cur_maps[map_idx].num_pos = 0;
            cur_maps[map_idx].num_neg = 0;
            continue;
        }
        
        int accumulated_pos = 0;
        int accumulated_neg = 0;
        
        Vec2 cell_pos = {(float)xi * params.cell_size_m - 3.0f, 
                        (float)yi * params.cell_size_m - 3.0f};
        
        for (int chunk_i = 0; chunk_i < last_chunk_index; chunk_i++) {
            Vec3 other_state = chunk_states[chunk_i * params.num_particles + particle_idx];
            float dx = other_state.x - state.x;
            float dy = other_state.y - state.y;
            float dist_sq = dx * dx + dy * dy;
            
            if (dist_sq <= 36.0f) {
                Pair<Mat2, Vec2> R_t_other = get_affine_tx_from_state(other_state);
                Mat2 Rc = R_t_other.first;
                Vec2 tc = R_t_other.second;
                
                Vec2 m = tc - t;
                Vec2 q_body = Rc.transpose() * cell_pos;
                Vec2 pt_body = R * (m + q_body);
                
                int px = __float2int_rn((pt_body.x + 3.0f) / params.cell_size_m);
                int py = __float2int_rn((pt_body.y + 3.0f) / params.cell_size_m);
                
                if (px >= 0 && px < 60 && py >= 0 && py < 60) {
                    accumulated_pos += chunks[chunk_i].cells[px][py].num_pos;
                    accumulated_neg += chunks[chunk_i].cells[px][py].num_neg;
                }
            }
        }
    
        int map_idx = particle_idx * 3600 + cell_idx;
        cur_maps[map_idx].num_pos = accumulated_pos;
        cur_maps[map_idx].num_neg = accumulated_neg;
    }
}

__global__ void computeLogLikelihoods(ChunkCell* cur_maps, Chunk* chunks, float* log_likelihoods, 
                                     KernelParams params, int last_chunk_index)
{
    int particle_idx = blockIdx.x;
    if (particle_idx >= params.num_particles) return;
    
    Chunk prediction = chunks[last_chunk_index];
    
    for (int cell_idx = threadIdx.x; cell_idx < 3600; cell_idx += blockDim.x) {
        int xi = cell_idx / 60;
        int yi = cell_idx % 60;

        int map_idx = particle_idx * 3600 + cell_idx;
        ChunkCell cur_cell = cur_maps[map_idx];
        ChunkCell pred_cell = prediction.cells[xi][yi];

        if (pred_cell.num_pos == 0 && pred_cell.num_neg == 0) {
            continue;
        }
        
        float p_alpha = params.alpha_prior + params.pos_weight * pred_cell.num_pos;
        float p_beta = params.beta_prior + params.neg_weight * pred_cell.num_neg;
        float p_theta = roundf((p_alpha) / (p_alpha + p_beta));
        //float p_theta = (p_alpha) / (p_alpha + p_beta);
        
        float alpha = params.alpha_prior + params.pos_weight * cur_cell.num_pos;
        float beta = params.beta_prior + params.neg_weight * cur_cell.num_neg;
        float theta = (alpha) / (alpha + beta);
    
        float cell_ll = logf(p_theta * theta + (1.0f - p_theta) * (1.0f - theta));
        
        atomicAdd(&log_likelihoods[particle_idx], cell_ll);
    }
}

__global__ void calculateScores(float* log_likelihoods, float* scores_raw, 
                               int num_particles, float min_ll, float max_ll)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    if (max_ll - min_ll < 1e-3f) {
        scores_raw[idx] = 1.0f / num_particles;
    } else {
        scores_raw[idx] = (log_likelihoods[idx] - min_ll) / (max_ll - min_ll);
    }
}

__global__ void normalizeScoresBySum(float* scores_raw, float* scores, 
                                    int num_particles, float sum_scores_raw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    scores[idx] = scores_raw[idx] / sum_scores_raw;
}

__global__ void calculateCumulativeSum(float* scores, float* cumsum, int num_particles)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cumsum[0] = scores[0];
        for (int i = 1; i < num_particles; i++) {
            cumsum[i] = cumsum[i - 1] + scores[i];
        }
    }
}

__global__ void susResample(Particle* particles, Particle* particles_swap, 
                           Vec3* chunk_states, Vec3* chunk_states_swap, 
                           float* cumsum, KernelParams params, int num_chunks, float initial)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.num_particles) return;

    float val = initial + ((float)i / (float)params.num_particles);

    int idx = 0;
    for (int j = 0; j < params.num_particles; j++) {
        if (val <= cumsum[j]) {
            idx = j;
            break;
        }
    }

    for (int t = 0; t < params.max_trajectory_length; t++) {
        int src_pos = idx * params.max_trajectory_length + t;
        int dst_pos = i * params.max_trajectory_length + t;
        particles_swap[dst_pos] = particles[src_pos];
    }
    
    for (int chunk_i = 0; chunk_i < num_chunks; chunk_i++) {
        int src_pos = chunk_i * params.num_particles + idx;
        int dst_pos = chunk_i * params.num_particles + i;
        chunk_states_swap[dst_pos] = chunk_states[src_pos];
    }
}

// ============================================================================
// ParticleSlam Class Implementation
// ============================================================================

ParticleSlam::ParticleSlam(int num_particles, int max_trajectory_length, int max_chunk_length,
                           float cell_size_m, float pos_weight, float neg_weight, 
                           float alpha_prior, float beta_prior)
    : max_chunk_length_(max_chunk_length)
    , current_timestep_(0)
    , current_chunk_index_(0)
    , last_chunk_timestamp_(-1.0)
    , first_step_(true)
    , initialized_(false)
    , d_particles_(nullptr)
    , d_particles_swap_(nullptr)
    , d_chunk_states_(nullptr)
    , d_chunk_states_swap_(nullptr)
    , d_chunks_(nullptr)
    , d_randStates_(nullptr)
    , d_log_likelihoods_(nullptr)
    , d_scores_raw_(nullptr)
    , d_scores_(nullptr)
    , d_cumsum_(nullptr)
    , d_cur_maps_(nullptr)
{
    params_.num_particles = num_particles;
    params_.max_trajectory_length = max_trajectory_length;
    params_.cell_size_m = cell_size_m;
    params_.pos_weight = pos_weight;
    params_.neg_weight = neg_weight;
    params_.alpha_prior = alpha_prior;
    params_.beta_prior = beta_prior;
}

ParticleSlam::~ParticleSlam()
{
    if (initialized_) {
        cudaFree(d_particles_);
        cudaFree(d_particles_swap_);
        cudaFree(d_chunk_states_);
        cudaFree(d_chunk_states_swap_);
        cudaFree(d_chunks_);
        cudaFree(d_randStates_);
        cudaFree(d_log_likelihoods_);
        cudaFree(d_scores_raw_);
        cudaFree(d_scores_);
        cudaFree(d_cumsum_);
        cudaFree(d_cur_maps_);
    }
}

void ParticleSlam::init(unsigned long long random_seed)
{
    if (initialized_) {
        fprintf(stderr, "ParticleSlam already initialized!\n");
        return;
    }
    
    cudaError_t status;
    
    // Allocate device memory
    status = cudaMalloc(&d_particles_, params_.num_particles * params_.max_trajectory_length * sizeof(Particle));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_particles failed");
    
    status = cudaMalloc(&d_particles_swap_, params_.num_particles * params_.max_trajectory_length * sizeof(Particle));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_particles_swap failed");
    
    status = cudaMalloc(&d_chunk_states_, max_chunk_length_ * params_.num_particles * sizeof(Vec3));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_chunk_states failed");
    
    status = cudaMalloc(&d_chunk_states_swap_, max_chunk_length_ * params_.num_particles * sizeof(Vec3));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_chunk_states_swap failed");
    
    status = cudaMalloc(&d_chunks_, max_chunk_length_ * sizeof(Chunk));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_chunks failed");
    
    status = cudaMalloc(&d_randStates_, params_.num_particles * sizeof(curandState));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_randStates failed");
    
    status = cudaMalloc(&d_log_likelihoods_, params_.num_particles * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_log_likelihoods failed");
    
    status = cudaMalloc(&d_scores_raw_, params_.num_particles * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_scores_raw failed");
    
    status = cudaMalloc(&d_scores_, params_.num_particles * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_scores failed");
    
    status = cudaMalloc(&d_cumsum_, params_.num_particles * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_cumsum failed");
    
    status = cudaMalloc(&d_cur_maps_, params_.num_particles * 60 * 60 * sizeof(ChunkCell));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_cur_maps failed");
    
    // Initialize to zero
    cudaMemset(d_particles_, 0, params_.num_particles * params_.max_trajectory_length * sizeof(Particle));
    
    // Initialize random states
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    initRandom<<<numBlocks, blockSize>>>(d_randStates_, params_.num_particles, random_seed);
    cudaDeviceSynchronize();
    
    initialized_ = true;
}

void ParticleSlam::apply_step(Vec3 dx_step, double timestamp)
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    if (first_step_) {
        first_step_ = false;
        return;
    }
    
    current_timestep_++;
    
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    applyStepKernel<<<numBlocks, blockSize>>>(d_particles_, d_randStates_, params_, 
                                              dx_step, timestamp, current_timestep_);
    cudaDeviceSynchronize();
}

void ParticleSlam::extract_chunk_states(int timestep, int chunk_index)
{
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    extractParticleStatesAtTimestep<<<numBlocks, blockSize>>>(
        d_particles_, d_chunk_states_, params_, timestep, chunk_index);
    cudaDeviceSynchronize();
}

void ParticleSlam::evaluate_and_resample(int chunk_index)
{
    cudaMemset(d_log_likelihoods_, 0, params_.num_particles * sizeof(float));
    
    dim3 blockDim(256);
    dim3 gridDim(params_.num_particles);
    
    // Accumulate maps
    accumulateMapForParticles<<<gridDim, blockDim>>>(
        d_chunk_states_, d_chunks_, d_cur_maps_, params_, chunk_index);
    cudaDeviceSynchronize();
    
    // Compute log-likelihoods
    computeLogLikelihoods<<<gridDim, blockDim>>>(
        d_cur_maps_, d_chunks_, d_log_likelihoods_, params_, chunk_index);
    cudaDeviceSynchronize();
    
    // Download log-likelihoods to find min/max
    float* h_log_likelihoods = new float[params_.num_particles];
    cudaMemcpy(h_log_likelihoods, d_log_likelihoods_, 
               params_.num_particles * sizeof(float), cudaMemcpyDeviceToHost);
    
    float min_ll = FLT_MAX;
    float max_ll = -FLT_MAX;
    for (int i = 0; i < params_.num_particles; i++) {
        if (h_log_likelihoods[i] < min_ll) min_ll = h_log_likelihoods[i];
        if (h_log_likelihoods[i] > max_ll) max_ll = h_log_likelihoods[i];
    }
    delete[] h_log_likelihoods;
    
    // Calculate scores
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    calculateScores<<<numBlocks, blockSize>>>(
        d_log_likelihoods_, d_scores_raw_, params_.num_particles, min_ll, max_ll);
    cudaDeviceSynchronize();
    
    // Download scores to compute sum
    float* h_scores_raw = new float[params_.num_particles];
    cudaMemcpy(h_scores_raw, d_scores_raw_, 
               params_.num_particles * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum_scores_raw = 0.0f;
    for (int i = 0; i < params_.num_particles; i++) {
        sum_scores_raw += h_scores_raw[i];
    }
    delete[] h_scores_raw;
    
    // Normalize scores
    normalizeScoresBySum<<<numBlocks, blockSize>>>(
        d_scores_raw_, d_scores_, params_.num_particles, sum_scores_raw);
    cudaDeviceSynchronize();
    
    // Calculate cumulative sum
    calculateCumulativeSum<<<1, 1>>>(d_scores_, d_cumsum_, params_.num_particles);
    cudaDeviceSynchronize();
    
    // Resample
    float initial = ((float)rand() / (float)RAND_MAX) * (1.0f / (float)params_.num_particles);
    
    susResample<<<numBlocks, blockSize>>>(
        d_particles_, d_particles_swap_, d_chunk_states_, d_chunk_states_swap_,
        d_cumsum_, params_, chunk_index + 1, initial);
    cudaDeviceSynchronize();
    
    // Swap pointers
    Particle* temp = d_particles_;
    d_particles_ = d_particles_swap_;
    d_particles_swap_ = temp;
    
    Vec3* temp_chunks = d_chunk_states_;
    d_chunk_states_ = d_chunk_states_swap_;
    d_chunk_states_swap_ = temp_chunks;
}

void ParticleSlam::ingest_visual_measurement(const Chunk& chunk, double timestamp, bool resample)
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    if (timestamp <= last_chunk_timestamp_) {
        fprintf(stderr, "Warning: Chunk timestamp %f is not greater than last chunk timestamp %f\n", 
                timestamp, last_chunk_timestamp_);
        return;
    }
    
    // Find matching timestep by downloading first particle's trajectory
    Particle* single_particle_traj = new Particle[params_.max_trajectory_length];
    cudaMemcpy(single_particle_traj, d_particles_, 
               params_.max_trajectory_length * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    int chunk_timestep = -1;
    for (int t = 0; t < params_.max_trajectory_length; t++) {
        if (single_particle_traj[t].timestamp >= timestamp) {
            chunk_timestep = t;
            break;
        }
    }
    delete[] single_particle_traj;
    
    if (chunk_timestep == -1) {
        fprintf(stderr, "Could not find matching timestep for chunk timestamp %f\n", timestamp);
        return;
    }
    
    // Upload chunk to device
    cudaMemcpy(&d_chunks_[current_chunk_index_], &chunk, sizeof(Chunk), cudaMemcpyHostToDevice);
    
    // Extract particle states at this timestep
    extract_chunk_states(chunk_timestep, current_chunk_index_);
    
    // Evaluate and resample if requested
    if (resample) {
        evaluate_and_resample(current_chunk_index_);
    }
    
    last_chunk_timestamp_ = timestamp;
    current_chunk_index_++;
}

void ParticleSlam::download_chunk_states(Vec3* h_chunk_states, int max_chunks) const
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    int num_chunks = (max_chunks < current_chunk_index_) ? max_chunks : current_chunk_index_;
    
    cudaError_t status = cudaMemcpy(h_chunk_states, d_chunk_states_, 
                                     num_chunks * params_.num_particles * sizeof(Vec3), 
                                     cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to download chunk states for visualization");
    }
}

void ParticleSlam::download_chunk_states_for_particle(Vec3* h_chunk_states, int particle_idx, int max_chunks) const
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    if (particle_idx < 0 || particle_idx >= params_.num_particles) 
        throw std::runtime_error("Invalid particle index");
    
    int num_chunks = (max_chunks < current_chunk_index_) ? max_chunks : current_chunk_index_;
    
    // Download all chunk states then extract the particle we want
    Vec3* h_all_chunk_states = new Vec3[num_chunks * params_.num_particles];
    cudaError_t status = cudaMemcpy(h_all_chunk_states, d_chunk_states_, 
                                     num_chunks * params_.num_particles * sizeof(Vec3), 
                                     cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        delete[] h_all_chunk_states;
        throw std::runtime_error("Failed to download chunk states");
    }
    
    // Extract just the requested particle's chunk states
    for (int i = 0; i < num_chunks; i++) {
        h_chunk_states[i] = h_all_chunk_states[i * params_.num_particles + particle_idx];
    }
    
    delete[] h_all_chunk_states;
}

void ParticleSlam::download_scores(float* h_scores) const
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    cudaMemcpy(h_scores, d_scores_raw_, 
               params_.num_particles * sizeof(float), cudaMemcpyDeviceToHost);
}

Map* ParticleSlam::bake_map()
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    if (current_chunk_index_ == 0) throw std::runtime_error("No chunks ingested yet");
    
    // Download scores to find best particle
    float* h_scores = new float[params_.num_particles];
    cudaMemcpy(h_scores, d_scores_raw_, 
               params_.num_particles * sizeof(float), cudaMemcpyDeviceToHost);
    
    int best_particle_idx = 0;
    float best_score = h_scores[0];
    for (int i = 1; i < params_.num_particles; i++) {
        if (h_scores[i] > best_score) {
            best_score = h_scores[i];
            best_particle_idx = i;
        }
    }
    delete[] h_scores;
    
    // Download all chunk states
    Vec3* h_all_chunk_states = new Vec3[current_chunk_index_ * params_.num_particles];
    cudaError_t status = cudaMemcpy(h_all_chunk_states, d_chunk_states_, 
                                     current_chunk_index_ * params_.num_particles * sizeof(Vec3), 
                                     cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        delete[] h_all_chunk_states;
        throw std::runtime_error("Failed to download chunk states");
    }
    
    // Extract chunk states for best particle
    Vec3* h_chunk_states = new Vec3[current_chunk_index_];
    for (int chunk_i = 0; chunk_i < current_chunk_index_; chunk_i++) {
        h_chunk_states[chunk_i] = h_all_chunk_states[chunk_i * params_.num_particles + best_particle_idx];
    }
    delete[] h_all_chunk_states;
    
    // Download all chunks
    Chunk* h_chunks = new Chunk[current_chunk_index_];
    cudaMemcpy(h_chunks, d_chunks_, current_chunk_index_ * sizeof(Chunk), cudaMemcpyDeviceToHost);
    
    // First pass: find bounds by transforming all chunk cells to reference frame
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;
    
    for (int chunk_i = 0; chunk_i < current_chunk_index_; chunk_i++) {
        Vec3 chunk_state = h_chunk_states[chunk_i];
        
        // Transform from chunk frame to reference frame
        for (int xi = 0; xi < 60; xi++) {
            for (int yi = 0; yi < 60; yi++) {
                ChunkCell cell = h_chunks[chunk_i].cells[xi][yi];
                if (cell.num_pos == 0 && cell.num_neg == 0) continue;
                
                // Cell position in chunk frame (unquantized)
                Vec2 cell_pos_chunk = {(float)xi * params_.cell_size_m - 3.0f, 
                                      (float)yi * params_.cell_size_m - 3.0f};
                
                // Transform to reference frame using body2map
                Vec2 cell_pos_ref = body2map(chunk_state, cell_pos_chunk);
                
                if (cell_pos_ref.x < min_x) min_x = cell_pos_ref.x;
                if (cell_pos_ref.x > max_x) max_x = cell_pos_ref.x;
                if (cell_pos_ref.y < min_y) min_y = cell_pos_ref.y;
                if (cell_pos_ref.y > max_y) max_y = cell_pos_ref.y;
            }
        }
    }
    
    // Add padding
    min_x -= params_.cell_size_m;
    min_y -= params_.cell_size_m;
    max_x += params_.cell_size_m;
    max_y += params_.cell_size_m;
    
    // Calculate grid dimensions
    int width = (int)ceilf((max_x - min_x) / params_.cell_size_m);
    int height = (int)ceilf((max_y - min_y) / params_.cell_size_m);
    
    // Allocate map
    Map* map = new Map();
    map->cells = new ChunkCell[width * height];
    map->width = width;
    map->height = height;
    map->min_x = min_x;
    map->min_y = min_y;
    map->max_x = max_x;
    map->max_y = max_y;
    map->cell_size = params_.cell_size_m;
    
    // Initialize all cells to zero
    for (int i = 0; i < width * height; i++) {
        map->cells[i].num_pos = 0;
        map->cells[i].num_neg = 0;
    }
    
    // Second pass: accumulate cells into map
    for (int chunk_i = 0; chunk_i < current_chunk_index_; chunk_i++) {
        Vec3 chunk_state = h_chunk_states[chunk_i];
        
        for (int xi = 0; xi < 60; xi++) {
            for (int yi = 0; yi < 60; yi++) {
                ChunkCell cell = h_chunks[chunk_i].cells[xi][yi];
                if (cell.num_pos == 0 && cell.num_neg == 0) continue;
                
                // Cell position in chunk frame (unquantized)
                Vec2 cell_pos_chunk = {(float)xi * params_.cell_size_m - 3.0f, 
                                      (float)yi * params_.cell_size_m - 3.0f};
                
                // Transform to reference frame using body2map
                Vec2 cell_pos_ref = body2map(chunk_state, cell_pos_chunk);
                
                // Quantize to map grid
                int map_x = (int)roundf((cell_pos_ref.x - min_x) / params_.cell_size_m);
                int map_y = (int)roundf((cell_pos_ref.y - min_y) / params_.cell_size_m);
                
                if (map_x >= 0 && map_x < width && map_y >= 0 && map_y < height) {
                    int map_idx = map_y * width + map_x;
                    map->cells[map_idx].num_pos += cell.num_pos;
                    map->cells[map_idx].num_neg += cell.num_neg;
                }
            }
        }
    }
    
    delete[] h_chunk_states;
    delete[] h_chunks;
    
    return map;
}
