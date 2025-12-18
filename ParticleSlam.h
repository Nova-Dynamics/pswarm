#pragma once

#include "rt_jr.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

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

struct Map {
    ChunkCell* cells;  // Dynamically allocated array
    int width;         // Number of cells in x direction
    int height;        // Number of cells in y direction
    float min_x;       // Minimum x coordinate in meters
    float min_y;       // Minimum y coordinate in meters
    float max_x;       // Maximum x coordinate in meters
    float max_y;       // Maximum y coordinate in meters
    float cell_size;   // Size of each cell in meters
    
    Map() : cells(nullptr), width(0), height(0), 
            min_x(0), min_y(0), max_x(0), max_y(0), cell_size(0.1f) {}

    // Copy constructor - deep copy
    Map(const Map& other)
        : width(other.width), height(other.height),
        min_x(other.min_x), min_y(other.min_y),
        max_x(other.max_x), max_y(other.max_y),
        cell_size(other.cell_size)
    {
        if (other.cells && width > 0 && height > 0) {
            size_t num_cells = static_cast<size_t>(width) * static_cast<size_t>(height);
            cells = new ChunkCell[num_cells];
            memcpy(cells, other.cells, num_cells * sizeof(ChunkCell));
        } else {
            cells = nullptr;
        }
    }
    
    ~Map() {
        if (cells) delete[] cells;
    }
};

struct KernelParams {
    int num_particles;
    int max_trajectory_length;
    float cell_size_m;
    float pos_weight;
    float neg_weight;
    float alpha_prior;
    float beta_prior;
};

class ParticleSlam {
public:
    ParticleSlam(int num_particles, 
                 int max_trajectory_length, 
                 int max_chunk_length,
                 float cell_size_m = 0.1f,
                 float pos_weight = 0.7f,
                 float neg_weight = 0.4f,
                 float alpha_prior = 1.0f,
                 float beta_prior = 1.5f);
    
    ~ParticleSlam();
    
    // Initialize and allocate memory
    void init(unsigned long long random_seed = 1234ULL);
    
    // Apply a dead reckoning step with configurable process noise
    // pos_std: standard deviation for position noise (multiplicative)
    // yaw_std: standard deviation for yaw noise (additive)
    void apply_step(Vec3 dx_step, double timestamp, float pos_std = 1.6e-3f, float yaw_std = 1e-3f);
    
    // Ingest a visual measurement chunk
    int ingest_visual_measurement(const Chunk& chunk);
    
    void evaluate_and_resample(int chunk_index);


    // Download functions for visualization
    void download_chunk_states(Vec3* h_chunk_states, int max_chunks) const;
    void download_chunk_states_for_particle(Vec3* h_chunk_states, int particle_idx, int max_chunks) const;
    void download_scores(float* h_scores) const;
    
    // State accessors
    int get_current_chunk_count() const { return current_chunk_index_; }
    int get_current_timestep() const { return current_timestep_; }
    int get_num_particles() const { return params_.num_particles; }
    Particle* get_d_particles() const { return d_particles_; }

    // Create accumulated map from best particle
    Map* bake_best_particle_map();

    // Mapping specific functions ==========================
    void accumulate_map_from_trajectories(int chunk_index);


    // Localization specific functions ======================
    void accumulate_map_from_map(int chunk_index);
    void set_global_map(const Map& map);
    void uniform_initialize_particles();
    void prune_particles_outside_map();
    Map* bake_global_map_best_particle();
    
private:
    // Parameters
    KernelParams params_;
    int max_chunk_length_;
    
    // Device pointers
    Particle* d_particles_;
    Particle* d_particles_swap_;
    Vec3* d_chunk_states_;
    Vec3* d_chunk_states_swap_;
    Chunk* d_chunks_;
    struct curandStateXORWOW* d_randStates_;
    float* d_log_likelihoods_;
    float* d_scores_raw_;
    float* d_scores_;
    float* d_cumsum_;
    ChunkCell* d_accumulated_maps;
    Map* d_global_map_;
    ChunkCell* d_global_map_cells_;
    Map* h_global_map_;

    // State tracking
    int current_timestep_;
    int current_chunk_index_;
    double last_chunk_timestamp_;
    bool first_step_;
    bool initialized_;
    bool has_global_map_;
    bool chunks_wrapped_;  // True if current_chunk_index_ has wrapped around

};

Map* load_map_from_file(const char* filename);
bool save_map_to_file(const Map* map, const char* filename);