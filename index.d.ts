/**
 * TypeScript definitions for @novadynamics/pswarm
 */

export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export interface Particle {
    state: Vec3;
    timestamp: number;
}

export interface ChunkCell {
    num_pos: number;
    num_neg: number;
}

export interface Chunk {
    cells: ChunkCell[][];  // 60x60 array
    timestamp: number;
}

export interface Map {
    cells: ChunkCell[];
    width: number;
    height: number;
    min_x: number;
    min_y: number;
    max_x: number;
    max_y: number;
    cell_size: number;
}

export interface Mat3 {
    m: number[];  // 9 elements, row-major
}

export interface Measurement {
    covariance: Mat3;
    mean: Vec3;
    is_gaussian: boolean;
}

export class ParticleSlam {
    /**
     * Construct particle filter with specified parameters
     * @param num_particles Number of particles in the filter
     * @param max_trajectory_length Maximum trajectory history per particle
     * @param max_chunk_length Maximum number of visual measurement chunks to buffer
     * @param cell_size_m Size of map cells in meters (default: 0.1)
     * @param pos_weight Weight for positive observations (default: 0.7)
     * @param neg_weight Weight for negative observations (default: 0.4)
     * @param alpha_prior Alpha prior for Beta distribution (default: 1.0)
     * @param beta_prior Beta prior for Beta distribution (default: 1.5)
     * @param measurement_saturation Max value for observation counters (default: 200)
     */
    constructor(
        num_particles: number,
        max_trajectory_length: number,
        max_chunk_length: number,
        cell_size_m?: number,
        pos_weight?: number,
        neg_weight?: number,
        alpha_prior?: number,
        beta_prior?: number,
        measurement_saturation?: number
    );

    /**
     * Initialize CUDA memory and random number generators
     * @param random_seed Seed for reproducible random number generation (default: 1234)
     */
    init(random_seed?: number): void;

    /**
     * Apply dead reckoning motion step to all particles with noise
     * @param dx_step Mean motion delta (x, y, theta)
     * @param timestamp Timestamp for this step
     * @param pos_std Standard deviation for position noise (default: 1.6e-3)
     * @param yaw_std Standard deviation for yaw angle noise (default: 1e-3)
     */
    apply_step(dx_step: Vec3, timestamp: number, pos_std?: number, yaw_std?: number): void;

    /**
     * Ingest new visual measurement chunk
     * @param chunk Visual measurement chunk
     * @returns Index where chunk was stored, or -1 on error
     */
    ingest_visual_measurement(chunk: Chunk): number;

    /**
     * Evaluate particle likelihoods and resample
     * @param chunk_index Index of chunk to use for evaluation
     */
    evaluate_and_resample(chunk_index: number): void;

    /**
     * Download particle states at all chunk timestamps
     * @param max_chunks Maximum number of chunks to download
     * @returns Array of [chunk][particle] states
     */
    download_chunk_states(max_chunks: number): Vec3[][];

    /**
     * Download trajectory of single particle
     * @param particle_idx Index of particle to download
     * @param max_chunks Maximum number of chunks to download
     * @returns Array of particle states across chunks
     */
    download_chunk_states_for_particle(particle_idx: number, max_chunks: number): Vec3[];

    /**
     * Download raw particle scores from last evaluation
     * @returns Array of scores per particle
     */
    download_scores(): number[];

    /**
     * Download current particle states efficiently
     * @returns Array of current particle states
     */
    download_current_particle_states(): Particle[];

    /**
     * Get current chunk count
     */
    get_current_chunk_count(): number;

    /**
     * Get current timestep index
     */
    get_current_timestep(): number;

    /**
     * Get number of particles in filter
     */
    get_num_particles(): number;

    /**
     * Generate map from best particle's trajectory (SLAM mode)
     * @param start_chunk_index Starting index for chunks to include (default: 0)
     * @returns Generated map
     */
    bake_best_particle_map(start_chunk_index?: number): Map;

    /**
     * Accumulate predicted map from trajectory history (SLAM mode)
     * @param chunk_index Index of chunk to predict map for
     */
    accumulate_map_from_trajectories(chunk_index: number): void;

    /**
     * Accumulate predicted map from global reference map (MCL mode)
     * @param chunk_index Index of chunk to predict map for
     */
    accumulate_map_from_map(chunk_index: number): void;

    /**
     * Set global reference map for MCL localization
     * @param map Reference map
     */
    set_global_map(map: Map): void;

    /**
     * Initialize particles uniformly across valid map regions
     */
    uniform_initialize_particles(): void;

    /**
     * Prune particles outside valid map and reinitialize
     */
    prune_particles_outside_map(): void;

    /**
     * Merge best particle's map with global reference map
     * @param start_chunk_index Starting index for chunks to include (default: 0)
     * @returns Merged map
     */
    bake_global_map_best_particle(start_chunk_index?: number): Map;

    /**
     * Calculate mean, covariance, and Gaussianity test for particle distribution
     * @returns Measurement statistics
     */
    calculate_measurement(): Measurement;
}

/**
 * Load map from binary file
 * @param filename Path to input file
 * @returns Loaded map, or throws on error
 */
export function load_map_from_file(filename: string): Map;

/**
 * Save map to binary file
 * @param map Map to save
 * @param filename Path to output file
 * @returns true on success, false on failure
 */
export function save_map_to_file(map: Map, filename: string): boolean;
