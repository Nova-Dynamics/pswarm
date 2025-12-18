# @novadynamics/pswarm

GPU-accelerated particle filter for SLAM (Simultaneous Localization and Mapping) and MCL (Monte Carlo Localization) using CUDA.

## Installation

### Prerequisites

- Node.js 18.x or later
- CUDA Toolkit 11.0 or later
- Visual Studio 2019 or later (Windows) / GCC (Linux)
- Python 3.8+ (for node-gyp)

### Install

```bash
npm install @novadynamics/pswarm
```

The package will automatically compile the native addon during installation using node-gyp.

## Usage

### Basic SLAM Example

```javascript
const { ParticleSlam } = require('@novadynamics/pswarm');

// Create particle filter
const slam = new ParticleSlam(
    10000,  // num_particles
    1000,   // max_trajectory_length
    100     // max_chunk_length
);

// Initialize
slam.init(1234);  // random seed

// Apply motion step
slam.apply_step(
    { x: 0.1, y: 0.0, z: 0.0 },  // dx_step
    Date.now() / 1000.0           // timestamp
);

// Ingest visual measurement
const chunk = {
    cells: /* 60x60 array of {num_pos, num_neg} */,
    timestamp: Date.now() / 1000.0
};
const chunk_idx = slam.ingest_visual_measurement(chunk);

// Accumulate and evaluate
slam.accumulate_map_from_trajectories(chunk_idx);
slam.evaluate_and_resample(chunk_idx);

// Get statistics
const measurement = slam.calculate_measurement();
console.log('Mean pose:', measurement.mean);
console.log('Is Gaussian:', measurement.is_gaussian);

// Generate final map
const final_map = slam.bake_best_particle_map();
```

### MCL (Localization) Example

```javascript
const { ParticleSlam, load_map_from_file } = require('@novadynamics/pswarm');

// Load reference map
const global_map = load_map_from_file('reference_map.bin');

// Create particle filter
const slam = new ParticleSlam(10000, 1000, 100);
slam.init(1234);

// Set global map and initialize particles
slam.set_global_map(global_map);
slam.uniform_initialize_particles();

// Localization loop
slam.apply_step({ x: 0.1, y: 0.0, z: 0.0 }, timestamp);
const chunk_idx = slam.ingest_visual_measurement(chunk);
slam.accumulate_map_from_map(chunk_idx);  // Use global map
slam.evaluate_and_resample(chunk_idx);
slam.prune_particles_outside_map();

// Get current pose estimate
const measurement = slam.calculate_measurement();
console.log('Estimated pose:', measurement.mean);
```

## API Reference

See [index.d.ts](index.d.ts) for complete TypeScript definitions.

### ParticleSlam

Main particle filter class for SLAM and MCL.

#### Constructor

```javascript
new ParticleSlam(num_particles, max_trajectory_length, max_chunk_length, 
                 cell_size_m?, pos_weight?, neg_weight?, 
                 alpha_prior?, beta_prior?, measurement_saturation?)
```

#### Methods

**Initialization:**
- `init(random_seed?)` - Initialize CUDA memory and RNG

**Motion & Observation:**
- `apply_step(dx_step, timestamp, pos_std?, yaw_std?)` - Apply motion model
- `ingest_visual_measurement(chunk)` - Ingest observation chunk

**Evaluation:**
- `evaluate_and_resample(chunk_index)` - Compute weights and resample

**SLAM Mode:**
- `accumulate_map_from_trajectories(chunk_index)` - Predict from history
- `bake_best_particle_map()` - Generate final map

**MCL Mode:**
- `set_global_map(map)` - Set reference map
- `uniform_initialize_particles()` - Initialize particles in map
- `accumulate_map_from_map(chunk_index)` - Predict from global map
- `prune_particles_outside_map()` - Reinitialize invalid particles
- `bake_global_map_best_particle()` - Update global map

**Statistics:**
- `calculate_measurement()` - Get mean, covariance, Gaussianity test

**Download (Visualization):**
- `download_current_particle_states()` - Get current particle states
- `download_chunk_states(max_chunks)` - Get all particle trajectories
- `download_scores()` - Get particle weights

**Accessors:**
- `get_num_particles()` - Number of particles
- `get_current_timestep()` - Current timestep index
- `get_current_chunk_count()` - Number of chunks ingested

### Utility Functions

```javascript
load_map_from_file(filename)  // Load map from binary file
save_map_to_file(map, filename)  // Save map to binary file
```

## Data Structures

### Vec3
```javascript
{ x: number, y: number, z: number }
```

### Chunk
```javascript
{
    cells: ChunkCell[][],  // 60x60 array
    timestamp: number
}
```

### Map
```javascript
{
    cells: ChunkCell[],  // Flat array
    width: number,
    height: number,
    min_x: number,
    min_y: number,
    max_x: number,
    max_y: number,
    cell_size: number
}
```

### Measurement
```javascript
{
    mean: Vec3,           // Mean pose
    covariance: Mat3,     // 3x3 covariance matrix
    is_gaussian: boolean  // Gaussianity test result
}
```

## License

MIT

## Author

Nova Dynamics
