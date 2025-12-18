#include "ParticleSlam.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cfloat>

namespace pswarm {

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void init_random(curandState* states, int num_particles, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void apply_step_kernel(Particle* particles, curandState* states, KernelParams params, 
                                Vec3 dx_step, double step_timestamp, int current_timestep,
                                float pos_std, float yaw_std)
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
        float noise_x = curand_normal(&states[idx]) * pos_std + 1.0f;
        float noise_y = curand_normal(&states[idx]) * pos_std + 1.0f;
        
        particle_delta.x *= noise_x;
        particle_delta.y *= noise_y;
        theta_noise = curand_normal(&states[idx]) * yaw_std;
    }

    particles[curr_pos].state.x = prev_particle.state.x + particle_delta.x;
    particles[curr_pos].state.y = prev_particle.state.y + particle_delta.y;
    particles[curr_pos].state.z = prev_particle.state.z + dx_step.z + theta_noise;
    particles[curr_pos].timestamp = step_timestamp;
}

__global__ void append_particle_states_for_timestep(Particle* particles, Vec3* chunk_states, 
                                                KernelParams params, int timestep, int chunk_index)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_particles) return;

    int pos = idx * params.max_trajectory_length + (timestep % params.max_trajectory_length);
    Vec3 state = particles[pos].state;

    chunk_states[chunk_index * params.num_particles + idx] = state;
}

__global__ void accumulate_map_for_particles(Vec3* chunk_states, Chunk* chunks, ChunkCell* cur_maps, 
                                         KernelParams params, int last_chunk_index, int num_valid_chunks)
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
        
        Vec2 quantized_cell = {(float)xi, (float)yi};
        Vec2 cell_pos = dequantize_pt(quantized_cell, params.cell_size_m);
        cell_pos.x -= 3.0f;
        cell_pos.y -= 3.0f;
        
        // Loop through all valid chunks except the current one
        for (int i = 0; i < num_valid_chunks - 1; i++) {
            Vec3 other_state = chunk_states[i * params.num_particles + particle_idx];
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
                
                Vec2 pt_body_offset = {pt_body.x + 3.0f, pt_body.y + 3.0f};
                Vec2 quantized = quantize_pt(pt_body_offset, params.cell_size_m);
                int px = (int)quantized.x;
                int py = (int)quantized.y;
                
                if (px >= 0 && px < 60 && py >= 0 && py < 60) {
                    accumulated_pos += chunks[i].cells[px][py].num_pos;
                    accumulated_neg += chunks[i].cells[px][py].num_neg;
                }
            }
        }
    
        int map_idx = particle_idx * 3600 + cell_idx;
        cur_maps[map_idx].num_pos = accumulated_pos;
        cur_maps[map_idx].num_neg = accumulated_neg;
    }
}

__global__ void compute_log_likelihoods(ChunkCell* cur_maps, Chunk* chunks, float* log_likelihoods, 
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

__global__ void calculate_scores(float* log_likelihoods, float* scores_raw, 
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

__global__ void normalize_scores_by_sum(float* scores_raw, float* scores, 
                                    int num_particles, float sum_scores_raw)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    scores[idx] = scores_raw[idx] / sum_scores_raw;
}

__global__ void calculate_cumulative_sum(float* scores, float* cumsum, int num_particles)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cumsum[0] = scores[0];
        for (int i = 1; i < num_particles; i++) {
            cumsum[i] = cumsum[i - 1] + scores[i];
        }
    }
}

__global__ void sus_resample(Particle* particles, Particle* particles_swap, 
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

__global__ void accumulate_map_from_global( Vec3* chunk_states, Map* global_map, ChunkCell* cur_maps, 
                                      KernelParams params, int last_chunk_index)
{
    int particle_idx = blockIdx.x;
    if (particle_idx >= params.num_particles) return;
    
    // Get particle state at the chunk we're accumulating for
    Vec3 state = chunk_states[last_chunk_index * params.num_particles + particle_idx];
    
    // For each cell in the 60x60 chunk grid
    for (int cell_idx = threadIdx.x; cell_idx < 3600; cell_idx += blockDim.x) {
        int xi = cell_idx / 60;
        int yi = cell_idx % 60;
        
        // Cell position in chunk frame (unquantized)
        Vec2 quantized_cell = {(float)xi, (float)yi};
        Vec2 cell_pos_chunk = dequantize_pt(quantized_cell, params.cell_size_m);
        cell_pos_chunk.x -= 3.0f;
        cell_pos_chunk.y -= 3.0f;
        
        // Transform to global/reference frame using body2map
        Vec2 cell_pos_global = body2map(state, cell_pos_chunk);
        
        // Quantize to global map grid
        Vec2 relative_pos = {cell_pos_global.x - global_map->min_x, cell_pos_global.y - global_map->min_y};
        Vec2 quantized = quantize_pt(relative_pos, global_map->cell_size);
        int map_x = (int)quantized.x;
        int map_y = (int)quantized.y;
        
        // Check if position is within global map bounds
        if (map_x >= 0 && map_x < global_map->width && map_y >= 0 && map_y < global_map->height) {
            int map_idx = map_y * global_map->width + map_x;
            ChunkCell global_cell = global_map->cells[map_idx];
            
            // Copy to accumulated map
            int output_idx = particle_idx * 3600 + cell_idx;
            cur_maps[output_idx].num_pos = global_cell.num_pos;
            cur_maps[output_idx].num_neg = global_cell.num_neg;
        } else {
            // Cell is outside global map bounds - zero it out
            int output_idx = particle_idx * 3600 + cell_idx;
            cur_maps[output_idx].num_pos = 0;
            cur_maps[output_idx].num_neg = 0;
        }
    }
}

__global__ void prune_particles_kernel(Particle* particles, Map* global_map, 
                                        curandState* states, KernelParams params,
                                        int current_timestep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_particles) return;
    
    // Get the latest particle state
    int latest_pos = idx * params.max_trajectory_length + (current_timestep % params.max_trajectory_length);
    Particle latest_particle = particles[latest_pos];
    
    // Transform to map coordinates
    float x = latest_particle.state.x;
    float y = latest_particle.state.y;
    
    Vec2 particle_pos = {x, y};
    Vec2 relative_pos = {x - global_map->min_x, y - global_map->min_y};
    Vec2 quantized = quantize_pt(relative_pos, global_map->cell_size);
    int map_x = (int)quantized.x;
    int map_y = (int)quantized.y;
    
    bool needs_reinit = false;
    
    // Check if particle is outside map bounds or in invalid cell
    if (map_x < 0 || map_x >= global_map->width || map_y < 0 || map_y >= global_map->height) {
        needs_reinit = true;
    } else {
        int map_idx = map_y * global_map->width + map_x;
        ChunkCell cell = global_map->cells[map_idx];
        
        // If cell has no observations, it's invalid
        if (cell.num_pos == 0 && cell.num_neg == 0) {
            needs_reinit = true;
        }
    }
    
    // If particle needs reinitialization, find a valid unoccupied position
    if (needs_reinit) {
        bool valid = false;
        float proposed_x, proposed_y, proposed_theta;
        double proposed_timestamp = latest_particle.timestamp;
        
        int max_attempts = 10000;
        int attempt = 0;
        
        while (!valid && attempt < max_attempts) {
            attempt++;
            
            // Sample uniform random position within map bounds
            proposed_x = curand_uniform(&states[idx]) * (global_map->max_x - global_map->min_x) + global_map->min_x;
            proposed_y = curand_uniform(&states[idx]) * (global_map->max_y - global_map->min_y) + global_map->min_y;
            proposed_theta = curand_uniform(&states[idx]) * 2.0f * 3.14159265359f;
            
            // Quantize position to map grid
            Vec2 proposed_relative = {proposed_x - global_map->min_x, proposed_y - global_map->min_y};
            Vec2 proposed_quantized = quantize_pt(proposed_relative, global_map->cell_size);
            int new_map_x = (int)proposed_quantized.x;
            int new_map_y = (int)proposed_quantized.y;
            
            // Check if position is within map bounds
            if (new_map_x >= 0 && new_map_x < global_map->width && new_map_y >= 0 && new_map_y < global_map->height) {
                int new_map_idx = new_map_y * global_map->width + new_map_x;
                ChunkCell cell = global_map->cells[new_map_idx];
                
                // Calculate occupancy probability using Beta-Bernoulli model
                float alpha = params.alpha_prior + params.pos_weight * cell.num_pos;
                float beta = params.beta_prior + params.neg_weight * cell.num_neg;
                float occ_prob = alpha / (alpha + beta);
                
                // Accept if likely unoccupied (occ_prob < 0.35)
                if (occ_prob < 0.35f) {
                    valid = true;
                }
            }
        }
        
        // Set entire trajectory to the new state
        for (int t = 0; t < params.max_trajectory_length; t++) {
            int traj_pos = idx * params.max_trajectory_length + t;
            particles[traj_pos].state.x = proposed_x;
            particles[traj_pos].state.y = proposed_y;
            particles[traj_pos].state.z = proposed_theta;
            particles[traj_pos].timestamp = proposed_timestamp;
        }
    }
}

__global__ void uniform_initialize_particles_kernel(Particle* particles, Map* global_map, 
                                                     curandState* states, KernelParams params,
                                                     double initial_timestamp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_particles) return;
    
    // Each particle tries to find a valid unoccupied position
    bool valid = false;
    float proposed_x, proposed_y, proposed_theta;
    
    // Limit iterations to prevent infinite loops
    int max_attempts = 10000;
    int attempt = 0;
    
    while (!valid && attempt < max_attempts) {
        attempt++;
        
        // Sample uniform random position within map bounds
        proposed_x = curand_uniform(&states[idx]) * (global_map->max_x - global_map->min_x) + global_map->min_x;
        proposed_y = curand_uniform(&states[idx]) * (global_map->max_y - global_map->min_y) + global_map->min_y;
        proposed_theta = curand_uniform(&states[idx]) * 2.0f * 3.14159265359f;
        
        // Quantize position to map grid
        Vec2 proposed_relative = {proposed_x - global_map->min_x, proposed_y - global_map->min_y};
        Vec2 proposed_quantized = quantize_pt(proposed_relative, global_map->cell_size);
        int map_x = (int)proposed_quantized.x;
        int map_y = (int)proposed_quantized.y;
        
        // Check if position is within map bounds
        if (map_x >= 0 && map_x < global_map->width && map_y >= 0 && map_y < global_map->height) {
            int map_idx = map_y * global_map->width + map_x;
            ChunkCell cell = global_map->cells[map_idx];
            
            // Calculate occupancy probability using Beta-Bernoulli model
            float alpha = params.alpha_prior + params.pos_weight * cell.num_pos;
            float beta = params.beta_prior + params.neg_weight * cell.num_neg;
            float occ_prob = alpha / (alpha + beta);
            
            // Accept if likely unoccupied (occ_prob < 0.35)
            if (occ_prob < 0.35f) {
                valid = true;
            }
        }
    }
    
    // Set particle state at timestep 0
    // If no valid position found after max attempts, use the last proposal anyway
    int particle_pos = idx * params.max_trajectory_length + 0;
    particles[particle_pos].state.x = proposed_x;
    particles[particle_pos].state.y = proposed_y;
    particles[particle_pos].state.z = proposed_theta;
    particles[particle_pos].timestamp = initial_timestamp;
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
    , has_global_map_(false)
    , chunks_wrapped_(false)
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
    , d_accumulated_maps(nullptr)
    , d_global_map_(nullptr)
    , d_global_map_cells_(nullptr)
    , h_global_map_(nullptr)
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
        cudaFree(d_accumulated_maps);
        
        if (has_global_map_) {
            if (d_global_map_cells_) cudaFree(d_global_map_cells_);
            if (d_global_map_) cudaFree(d_global_map_);
            if (h_global_map_) {
                delete h_global_map_;
            }
        }
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
    
    status = cudaMalloc(&d_accumulated_maps, params_.num_particles * 60 * 60 * sizeof(ChunkCell));
    if (status != cudaSuccess) throw std::runtime_error("cudaMalloc d_cur_maps failed");
    
    // Initialize to zero
    cudaMemset(d_particles_, 0, params_.num_particles * params_.max_trajectory_length * sizeof(Particle));
    
    // Initialize random states
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    init_random<<<numBlocks, blockSize>>>(d_randStates_, params_.num_particles, random_seed);
    cudaDeviceSynchronize();
    
    initialized_ = true;
}

void ParticleSlam::set_global_map(const Map& map)
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    // Clean up existing global map if any
    if (has_global_map_) {
        if (d_global_map_cells_) {
            cudaFree(d_global_map_cells_);
            d_global_map_cells_ = nullptr;
        }
        if (d_global_map_) {
            cudaFree(d_global_map_);
            d_global_map_ = nullptr;
        }
        if (h_global_map_) {
            delete h_global_map_;
            h_global_map_ = nullptr;
        }
    }
    
    // Allocate and copy host map
    h_global_map_ = new Map();
    h_global_map_->width = map.width;
    h_global_map_->height = map.height;
    h_global_map_->min_x = map.min_x;
    h_global_map_->min_y = map.min_y;
    h_global_map_->max_x = map.max_x;
    h_global_map_->max_y = map.max_y;
    h_global_map_->cell_size = map.cell_size;
    
    size_t num_cells = static_cast<size_t>(map.width) * static_cast<size_t>(map.height);
    h_global_map_->cells = new ChunkCell[num_cells];
    memcpy(h_global_map_->cells, map.cells, num_cells * sizeof(ChunkCell));
    
    // Allocate device map structure
    cudaError_t status = cudaMalloc(&d_global_map_, sizeof(Map));
    if (status != cudaSuccess) {
        delete h_global_map_;
        h_global_map_ = nullptr;
        throw std::runtime_error("cudaMalloc d_global_map failed");
    }
    
    // Allocate device cells
    status = cudaMalloc(&d_global_map_cells_, num_cells * sizeof(ChunkCell));
    if (status != cudaSuccess) {
        cudaFree(d_global_map_);
        d_global_map_ = nullptr;
        delete h_global_map_;
        h_global_map_ = nullptr;
        throw std::runtime_error("cudaMalloc d_global_map cells failed");
    }
    
    // Copy cells to device
    status = cudaMemcpy(d_global_map_cells_, map.cells, num_cells * sizeof(ChunkCell), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(d_global_map_cells_);
        d_global_map_cells_ = nullptr;
        cudaFree(d_global_map_);
        d_global_map_ = nullptr;
        delete h_global_map_;
        h_global_map_ = nullptr;
        throw std::runtime_error("cudaMemcpy d_global_map cells failed");
    }
    
    // Create a temporary Map struct with device pointer
    Map temp_map;
    temp_map.cells = d_global_map_cells_;
    temp_map.width = map.width;
    temp_map.height = map.height;
    temp_map.min_x = map.min_x;
    temp_map.min_y = map.min_y;
    temp_map.max_x = map.max_x;
    temp_map.max_y = map.max_y;
    temp_map.cell_size = map.cell_size;
    
    // Copy map structure to device
    status = cudaMemcpy(d_global_map_, &temp_map, sizeof(Map), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(d_global_map_cells_);
        d_global_map_cells_ = nullptr;
        cudaFree(d_global_map_);
        d_global_map_ = nullptr;
        delete h_global_map_;
        h_global_map_ = nullptr;
        throw std::runtime_error("cudaMemcpy d_global_map structure failed");
    }
    
    // IMPORTANT: Prevent temp_map destructor from freeing device memory
    // temp_map.cells points to device memory, but Map destructor uses delete[]
    temp_map.cells = nullptr;
    
    has_global_map_ = true;

    std::cout << "Global map set with dimensions: " 
              << map.width << " x " << map.height << std::endl;
}

void ParticleSlam::uniform_initialize_particles()
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    if (!has_global_map_) throw std::runtime_error("Global map not set for localization");
    
    // Reset timestep and state
    current_timestep_ = 0;
    current_chunk_index_ = 0;
    last_chunk_timestamp_ = -1.0;
    first_step_ = true;
    chunks_wrapped_ = false;
    
    // Launch kernel to initialize particles uniformly across unoccupied areas
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    double initial_timestamp = 0.0;
    uniform_initialize_particles_kernel<<<numBlocks, blockSize>>>(
        d_particles_, d_global_map_, d_randStates_, params_, initial_timestamp);
    
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error("uniform_initialize_particles kernel failed");
    }
}

void ParticleSlam::apply_step(Vec3 dx_step, double timestamp, float pos_std, float yaw_std)
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    if (first_step_) {
        first_step_ = false;
        return;
    }
    
    current_timestep_++;
    
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    apply_step_kernel<<<numBlocks, blockSize>>>(d_particles_, d_randStates_, params_, 
                                              dx_step, timestamp, current_timestep_,
                                              pos_std, yaw_std);
    cudaDeviceSynchronize();
}

void ParticleSlam::accumulate_map_from_trajectories(int chunk_index)
{
    dim3 blockDim(256);
    dim3 gridDim(params_.num_particles);
    
    // Determine number of valid chunks
    int num_valid_chunks = chunks_wrapped_ ? max_chunk_length_ : (chunk_index + 1);
    
    // Accumulate maps
    accumulate_map_for_particles<<<gridDim, blockDim>>>(
        d_chunk_states_, d_chunks_, d_accumulated_maps, params_, chunk_index, num_valid_chunks);

    cudaDeviceSynchronize();
}

void ParticleSlam::accumulate_map_from_map(int chunk_index)
{
    
    if (!has_global_map_) throw std::runtime_error("Global map not set for localization");
    
    dim3 blockDim(256);
    dim3 gridDim(params_.num_particles);
    
    // Accumulate maps from global map
    accumulate_map_from_global<<<gridDim, blockDim>>>(
        d_chunk_states_, d_global_map_, d_accumulated_maps, params_, chunk_index);

    cudaDeviceSynchronize();
}

void ParticleSlam::prune_particles_outside_map()
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    if (!has_global_map_) throw std::runtime_error("Global map not set for localization");

    // Launch kernel to prune and reinitialize particles outside valid map regions
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    prune_particles_kernel<<<numBlocks, blockSize>>>(
        d_particles_, d_global_map_, d_randStates_, params_, current_timestep_);
    
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        throw std::runtime_error("prune_particles_kernel failed");
    }
}


void ParticleSlam::evaluate_and_resample(int chunk_index)
{
    cudaMemset(d_log_likelihoods_, 0, params_.num_particles * sizeof(float));
 
    dim3 blockDim(256);
    dim3 gridDim(params_.num_particles);

    // Compute log-likelihoods
    compute_log_likelihoods<<<gridDim, blockDim>>>(
        d_accumulated_maps, d_chunks_, d_log_likelihoods_, params_, chunk_index);
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
    
    calculate_scores<<<numBlocks, blockSize>>>(
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
    normalize_scores_by_sum<<<numBlocks, blockSize>>>(
        d_scores_raw_, d_scores_, params_.num_particles, sum_scores_raw);
    cudaDeviceSynchronize();
    
    // Calculate cumulative sum
    calculate_cumulative_sum<<<1, 1>>>(d_scores_, d_cumsum_, params_.num_particles);
    cudaDeviceSynchronize();
    
    // Resample
    float initial = ((float)rand() / (float)RAND_MAX) * (1.0f / (float)params_.num_particles);
    
    // Number of valid chunks for resampling
    int num_valid_chunks = chunks_wrapped_ ? max_chunk_length_ : (chunk_index + 1);
    
    sus_resample<<<numBlocks, blockSize>>>(
        d_particles_, d_particles_swap_, d_chunk_states_, d_chunk_states_swap_, 
        d_cumsum_, params_, num_valid_chunks, initial);
    cudaDeviceSynchronize();
    
    // Swap pointers
    Particle* temp = d_particles_;
    d_particles_ = d_particles_swap_;
    d_particles_swap_ = temp;
    
    Vec3* temp_chunks = d_chunk_states_;
    d_chunk_states_ = d_chunk_states_swap_;
    d_chunk_states_swap_ = temp_chunks;
}

int ParticleSlam::ingest_visual_measurement(const Chunk& chunk)
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    if (chunk.timestamp <= last_chunk_timestamp_) {
        fprintf(stderr, "Warning: Chunk timestamp %f is not greater than last chunk timestamp %f\n", 
                chunk.timestamp, last_chunk_timestamp_);
        return -1;
    }
    
    // Find matching timestep by downloading first particle's trajectory
    Particle* single_particle_traj = new Particle[params_.max_trajectory_length];
    cudaMemcpy(single_particle_traj, d_particles_, 
               params_.max_trajectory_length * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    int chunk_timestep = -1;
    for (int t = 0; t < params_.max_trajectory_length; t++) {
        if (single_particle_traj[t].timestamp >= chunk.timestamp) {
            chunk_timestep = t;
            break;
        }
    }
    delete[] single_particle_traj;
    
    if (chunk_timestep == -1) {
        fprintf(stderr, "Could not find matching timestep for chunk timestamp %f\n", chunk.timestamp);
        return -1;
    }
    
    // Upload chunk to device
    cudaMemcpy(&d_chunks_[current_chunk_index_], &chunk, sizeof(Chunk), cudaMemcpyHostToDevice);
    
    // Extract particle states at this timestep
    int blockSize = 256;
    int numBlocks = (params_.num_particles + blockSize - 1) / blockSize;
    
    append_particle_states_for_timestep<<<numBlocks, blockSize>>>(
        d_particles_, d_chunk_states_, params_, chunk_timestep, current_chunk_index_);

    cudaDeviceSynchronize();
    
    last_chunk_timestamp_ = chunk.timestamp;
    
    // Store the chunk index to return before incrementing
    int returned_index = current_chunk_index_;
    
    // Increment and wrap chunk index
    current_chunk_index_++;
    if (current_chunk_index_ >= max_chunk_length_) {
        current_chunk_index_ = 0;
        chunks_wrapped_ = true;
    }

    return returned_index;
}

void ParticleSlam::download_chunk_states(Vec3* h_chunk_states, int max_chunks) const
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    // Determine actual number of valid chunks
    int num_valid_chunks = chunks_wrapped_ ? max_chunk_length_ : current_chunk_index_;
    int num_chunks = (max_chunks < num_valid_chunks) ? max_chunks : num_valid_chunks;
    
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
    
    // Determine actual number of valid chunks
    int num_valid_chunks = chunks_wrapped_ ? max_chunk_length_ : current_chunk_index_;
    int num_chunks = (max_chunks < num_valid_chunks) ? max_chunks : num_valid_chunks;
    
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

Map* ParticleSlam::bake_global_map_best_particle()
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    if (!has_global_map_) throw std::runtime_error("Global map not set for localization");
    
    // Bake best particle map
    Map* best_map = bake_best_particle_map();

    // Copy host global map
    Map* global_map_copy = new Map(*h_global_map_);

    // Merge best particle map into global_map_copy (both on host)
    // For each cell in best_map, add its observations to the corresponding global map cell
    for (int best_y = 0; best_y < best_map->height; best_y++) {
        for (int best_x = 0; best_x < best_map->width; best_x++) {
            int best_idx = best_y * best_map->width + best_x;
            ChunkCell best_cell = best_map->cells[best_idx];
            
            // Skip cells with no observations
            if (best_cell.num_pos == 0 && best_cell.num_neg == 0) continue;
            
            // Calculate world position of this cell
            Vec2 quantized_pos = {(float)best_x, (float)best_y};
            Vec2 relative_pos = dequantize_pt(quantized_pos, best_map->cell_size);
            float world_x = best_map->min_x + relative_pos.x;
            float world_y = best_map->min_y + relative_pos.y;
            
            // Find corresponding cell in global map
            Vec2 world_pos_relative = {world_x - global_map_copy->min_x, world_y - global_map_copy->min_y};
            Vec2 global_quantized = quantize_pt(world_pos_relative, global_map_copy->cell_size);
            int global_x = (int)global_quantized.x;
            int global_y = (int)global_quantized.y;
            
            // Check if within global map bounds
            if (global_x >= 0 && global_x < global_map_copy->width && 
                global_y >= 0 && global_y < global_map_copy->height) {
                int global_idx = global_y * global_map_copy->width + global_x;
                
                // Add observations (with saturation to prevent overflow)
                int new_pos = (int)global_map_copy->cells[global_idx].num_pos + (int)best_cell.num_pos;
                int new_neg = (int)global_map_copy->cells[global_idx].num_neg + (int)best_cell.num_neg;
                
                global_map_copy->cells[global_idx].num_pos = (int16_t)(new_pos > 32767 ? 32767 : new_pos);
                global_map_copy->cells[global_idx].num_neg = (int16_t)(new_neg > 32767 ? 32767 : new_neg);
            }
        }
    }
    
    // Clean up best_map
    delete best_map;

    return global_map_copy;
}

Map* ParticleSlam::bake_best_particle_map()
{
    if (!initialized_) throw std::runtime_error("ParticleSlam not initialized");
    
    // Determine actual number of valid chunks
    int num_valid_chunks = chunks_wrapped_ ? max_chunk_length_ : current_chunk_index_;
    if (num_valid_chunks == 0) throw std::runtime_error("No chunks ingested yet");
    
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
    Vec3* h_all_chunk_states = new Vec3[num_valid_chunks * params_.num_particles];
    cudaError_t status = cudaMemcpy(h_all_chunk_states, d_chunk_states_, 
                                     num_valid_chunks * params_.num_particles * sizeof(Vec3), 
                                     cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        delete[] h_all_chunk_states;
        throw std::runtime_error("Failed to download chunk states");
    }
    
    // Extract chunk states for best particle
    Vec3* h_chunk_states = new Vec3[num_valid_chunks];
    for (int chunk_i = 0; chunk_i < num_valid_chunks; chunk_i++) {
        h_chunk_states[chunk_i] = h_all_chunk_states[chunk_i * params_.num_particles + best_particle_idx];
    }
    delete[] h_all_chunk_states;
    
    // Download all chunks
    Chunk* h_chunks = new Chunk[num_valid_chunks];
    cudaMemcpy(h_chunks, d_chunks_, num_valid_chunks * sizeof(Chunk), cudaMemcpyDeviceToHost);
    
    // First pass: find bounds by transforming all chunk cells to reference frame
    float min_x = FLT_MAX, min_y = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX;
    
    for (int chunk_i = 0; chunk_i < num_valid_chunks; chunk_i++) {
        Vec3 chunk_state = h_chunk_states[chunk_i];
        
        // Transform from chunk frame to reference frame
        for (int xi = 0; xi < 60; xi++) {
            for (int yi = 0; yi < 60; yi++) {
                ChunkCell cell = h_chunks[chunk_i].cells[xi][yi];
                if (cell.num_pos == 0 && cell.num_neg == 0) continue;
                
                // Cell position in chunk frame (unquantized)
                Vec2 quantized_cell = {(float)xi, (float)yi};
                Vec2 cell_pos_chunk = dequantize_pt(quantized_cell, params_.cell_size_m);
                cell_pos_chunk.x -= 3.0f;
                cell_pos_chunk.y -= 3.0f;
                
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
    for (int chunk_i = 0; chunk_i < num_valid_chunks; chunk_i++) {
        Vec3 chunk_state = h_chunk_states[chunk_i];
        
        for (int xi = 0; xi < 60; xi++) {
            for (int yi = 0; yi < 60; yi++) {
                ChunkCell cell = h_chunks[chunk_i].cells[xi][yi];
                if (cell.num_pos == 0 && cell.num_neg == 0) continue;
                
                // Cell position in chunk frame (unquantized)
                Vec2 quantized_cell = {(float)xi, (float)yi};
                Vec2 cell_pos_chunk = dequantize_pt(quantized_cell, params_.cell_size_m);
                cell_pos_chunk.x -= 3.0f;
                cell_pos_chunk.y -= 3.0f;
                
                // Transform to reference frame using body2map
                Vec2 cell_pos_ref = body2map(chunk_state, cell_pos_chunk);
                
                // Quantize to map grid
                Vec2 ref_relative = {cell_pos_ref.x - min_x, cell_pos_ref.y - min_y};
                Vec2 ref_quantized = quantize_pt(ref_relative, params_.cell_size_m);
                int map_x = (int)ref_quantized.x;
                int map_y = (int)ref_quantized.y;
                
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


// ============================================================================
// Map I/O Implementation
// ============================================================================

// File format:
// [Header]
//   uint32_t magic_number (0x4D415053 = "MAPS")
//   uint32_t version (currently 1)
//   uint32_t header_size (size of header in bytes, for future extensibility)
// [Version 1 Data]
//   int32_t width
//   int32_t height
//   float cell_size
//   float min_x, min_y, max_x, max_y
//   ChunkCell[width * height] cells (each cell is 4 bytes: 2x int16_t)

#define MAP_FILE_MAGIC 0x4D415053  // "MAPS" in ASCII
#define MAP_FILE_VERSION 1

bool save_map_to_file(const Map* map, const char* filename) {
    if (!map || !map->cells || map->width <= 0 || map->height <= 0) {
        fprintf(stderr, "save_map_to_file: Invalid map\n");
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "save_map_to_file: Failed to open file '%s' for writing\n", filename);
        return false;
    }
    
    try {
        // Write header
        uint32_t magic = MAP_FILE_MAGIC;
        uint32_t version = MAP_FILE_VERSION;
        uint32_t header_size = sizeof(uint32_t) * 3;  // magic + version + header_size
        
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
        
        // Write map metadata
        int32_t width = map->width;
        int32_t height = map->height;
        
        file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        file.write(reinterpret_cast<const char*>(&map->cell_size), sizeof(map->cell_size));
        file.write(reinterpret_cast<const char*>(&map->min_x), sizeof(map->min_x));
        file.write(reinterpret_cast<const char*>(&map->min_y), sizeof(map->min_y));
        file.write(reinterpret_cast<const char*>(&map->max_x), sizeof(map->max_x));
        file.write(reinterpret_cast<const char*>(&map->max_y), sizeof(map->max_y));
        
        // Write cell data
        size_t num_cells = static_cast<size_t>(width) * static_cast<size_t>(height);
        file.write(reinterpret_cast<const char*>(map->cells), num_cells * sizeof(ChunkCell));
        
        if (!file.good()) {
            fprintf(stderr, "save_map_to_file: Error writing to file\n");
            file.close();
            return false;
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "save_map_to_file: Exception - %s\n", e.what());
        file.close();
        return false;
    }
}

Map* load_map_from_file(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "load_map_from_file: Failed to open file '%s' for reading\n", filename);
        return nullptr;
    }
    
    try {
        // Read and verify header
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t header_size = 0;
        
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
        
        if (magic != MAP_FILE_MAGIC) {
            fprintf(stderr, "load_map_from_file: Invalid file format (bad magic number)\n");
            file.close();
            return nullptr;
        }
        
        if (version > MAP_FILE_VERSION) {
            fprintf(stderr, "load_map_from_file: File version %u is newer than supported version %u\n", 
                    version, MAP_FILE_VERSION);
            file.close();
            return nullptr;
        }
        
        // Skip any extra header data (for forward compatibility)
        if (header_size > sizeof(uint32_t) * 3) {
            file.seekg(header_size - sizeof(uint32_t) * 3, std::ios::cur);
        }
        
        // Read map metadata
        int32_t width = 0;
        int32_t height = 0;
        float cell_size = 0.0f;
        float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
        
        file.read(reinterpret_cast<char*>(&width), sizeof(width));
        file.read(reinterpret_cast<char*>(&height), sizeof(height));
        file.read(reinterpret_cast<char*>(&cell_size), sizeof(cell_size));
        file.read(reinterpret_cast<char*>(&min_x), sizeof(min_x));
        file.read(reinterpret_cast<char*>(&min_y), sizeof(min_y));
        file.read(reinterpret_cast<char*>(&max_x), sizeof(max_x));
        file.read(reinterpret_cast<char*>(&max_y), sizeof(max_y));
        
        if (!file.good() || width <= 0 || height <= 0) {
            fprintf(stderr, "load_map_from_file: Invalid map dimensions (%d x %d)\n", width, height);
            file.close();
            return nullptr;
        }
        
        // Allocate map
        Map* map = new Map();
        map->width = width;
        map->height = height;
        map->cell_size = cell_size;
        map->min_x = min_x;
        map->min_y = min_y;
        map->max_x = max_x;
        map->max_y = max_y;
        
        size_t num_cells = static_cast<size_t>(width) * static_cast<size_t>(height);
        map->cells = new ChunkCell[num_cells];
        
        // Read cell data
        file.read(reinterpret_cast<char*>(map->cells), num_cells * sizeof(ChunkCell));
        
        if (!file.good()) {
            fprintf(stderr, "load_map_from_file: Error reading cell data\n");
            delete map;
            file.close();
            return nullptr;
        }
        
        file.close();
        
        std::cout << "Loaded map: " << width << " x " << height 
                  << " (" << num_cells << " cells, " 
                  << (num_cells * sizeof(ChunkCell)) / 1024.0 << " KB)" << std::endl;
        
        return map;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "load_map_from_file: Exception - %s\n", e.what());
        file.close();
        return nullptr;
    }
}

} // namespace pswarm