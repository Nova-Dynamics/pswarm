#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace pswarm {

// ============================================================================
// Math Utilities
// ============================================================================

template<typename T1, typename T2>
struct Pair {
    T1 first;
    T2 second;
};

// functions and ops defined in structs are implicitly inlined
struct Vec2 {
    float x, y;

    // const here just means that this method doesn't modify the object
    // that it's called on.
    __host__ __device__ Vec2 operator+(Vec2 v) const { return { x + v.x, y + v.y }; }
    __host__ __device__ Vec2 operator-(Vec2 v) const { return { x - v.x, y - v.y }; }
    __host__ __device__ Vec2 operator*(float s) const { return { x * s, y * s }; }
    __host__ __device__ float dot(Vec2 v) const { return x * v.x + y * v.y; }
    __host__ __device__ float cross(Vec2 v) const {
        return x * v.y - y * v.x;
    }
    __host__ __device__ float length() const { return sqrtf(x * x + y * y); }
};

struct Mat2 {
    float m[4]; // Let's choose row-major

    __host__ __device__ float& operator()(int r, int c) { return m[r * 2 + c]; } // Used for modification/assignment (row-major)
    __host__ __device__ const float& operator()(int r, int c) const { return m[r * 2 + c]; } // Used for read-only reference

    __host__ __device__ Mat2 transpose() const {
        Mat2 result;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result(i, j) = (*this)(j, i);
            }
        }

        return result;
    }
};

// row-major matrix multiplication
__host__ __device__ inline Mat2 operator*(const Mat2& a, const Mat2& b) {
    Mat2 result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j);
        }
    }

    return result;
}

// row-major matrix-vector multiplication
__host__ __device__ inline Vec2 operator*(const Mat2& a, const Vec2& v) {
    Vec2 result{};
    result.x = a(0, 0) * v.x + a(0, 1) * v.y;
    result.y = a(1, 0) * v.x + a(1, 1) * v.y;

    return result;
}

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3 operator+(Vec3 v) const { return { x + v.x, y + v.y, z + v.z }; }
    __host__ __device__ Vec3 operator-(Vec3 v) const { return { x - v.x, y - v.y, z - v.z }; }
    __host__ __device__ Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
    __host__ __device__ float dot(Vec3 v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ Vec3 cross(Vec3 v) const {
        return { y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x };
    }
    __host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
};

struct Mat3 {
    float m[9];

    __host__ __device__ float& operator()(int r, int c) { return m[r * 3 + c]; }
    __host__ __device__ const float& operator()(int r, int c) const { return m[r * 3 + c]; }

    __host__ __device__ Mat3 transpose() const {
        Mat3 result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result(i, j) = (*this)(j, i);
            }
        }

        return result;
    }
};

__host__ __device__ inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    Mat3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
        }
    }

    return result;
}

__host__ __device__ inline Vec3 operator*(const Mat3& a, const Vec3& v) {
    Vec3 result{};
    result.x = a(0, 0) * v.x + a(0, 1) * v.y + a(0, 2) * v.z;
    result.y = a(1, 0) * v.x + a(1, 1) * v.y + a(1, 2) * v.z;
    result.z = a(2, 0) * v.x + a(2, 1) * v.y + a(2, 2) * v.z;

    return result;
}

__host__ __device__ inline Mat2 get_R_from_theta(float theta) {
    Mat2 R{};

    float c = cosf(theta);
    float s = sinf(theta);
    R(0, 0) = c;
    R(1, 0) = s;
    R(0, 1) = -s;
    R(1, 1) = c;

    return R;
}

__host__ __device__ inline Pair<Mat2, Vec2> get_affine_tx_from_state(Vec3 state) {
    Mat2 R{};
    Vec2 t{};

    // Populate translation vector
    t.x = state.x;
    t.y = state.y;

    // NOTE: because state vector theta represents
    // clockwise rotations, need to keep this in mind
    // when using R downstream. This usually just looks
    // like transposing R before using in practice.
    R = get_R_from_theta(state.z);

    return { R, t };
}

__host__ __device__ inline Vec2 quantize_pt(Vec2 pt, float cell_width) {
    Vec2 qpt{};

    qpt.x = roundf(pt.x / cell_width);
    qpt.y = roundf(pt.y / cell_width);

    return qpt;
}

__host__ __device__ inline Vec2 dequantize_pt(Vec2 qpt, float cell_width) {
    Vec2 pt{};

    pt.x = qpt.x * cell_width;
    pt.y = qpt.y * cell_width;

    return pt;
}

__host__ __device__ inline Vec2 body2map(Vec3 body_state_map, Vec2 body_pt) {
    Mat2 R{};
    Vec2 body_state_pos{ body_state_map.x, body_state_map.y };
    Vec2 result{};

    R = get_R_from_theta(body_state_map.z);
    result = body_state_pos + R.transpose() * body_pt;

    return result;
}

// ============================================================================
// ParticleSlam Data Structures
// ============================================================================

struct Particle {
    Vec3 state;
    double timestamp;
};

struct ChunkCell {
    uint8_t num_pos;
    uint8_t num_neg;
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
    uint8_t measurement_saturation;
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
                 float beta_prior = 1.5f,
                 uint8_t measurement_saturation = 200);
    
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
    void download_current_particle_states(Particle* h_current_states) const;
    
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

} // namespace pswarm