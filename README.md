# CUDA Particle SLAM

A high-performance particle filter implementation for Simultaneous Localization and Mapping (SLAM) and Monte Carlo Localization (MCL) using CUDA and exposed as a Node.js native addon.

**⚠️ Linux Only**: This project only supports Linux with NVIDIA GPUs. Windows is not supported.

## Overview

This project implements a GPU-accelerated particle filter that can perform both:

- **SLAM (Simultaneous Localization and Mapping)**: Build a map of an unknown environment while simultaneously tracking the robot's position
- **MCL (Monte Carlo Localization)**: Localize a robot within a known map using particle filtering

The core algorithms run on CUDA-enabled GPUs for real-time performance with thousands of particles, while the Node.js interface makes it easy to integrate into robotics applications.

## Features

- 🚀 **GPU-Accelerated**: All compute-intensive operations run on CUDA
- 🎯 **Particle Filter**: Supports up to 10,000+ particles for robust estimation
- 🗺️ **Occupancy Grid Mapping**: Uses Beta-Bernoulli model for probabilistic mapping
- 📊 **Real-time Visualization**: ASCII terminal visualizer for particle states and maps
- 📈 **Performance Profiling**: Built-in timing for all operations
- 🔄 **Resampling**: Efficient GPU-based particle resampling
- 📦 **Easy Integration**: Node.js API for seamless integration

## Requirements

### System Requirements
- **Linux Operating System** (Ubuntu 20.04+ recommended, WSL2 supported)
- **NVIDIA GPU** with CUDA support (Compute Capability 3.5+)
- **CUDA Toolkit** 10.0 or later (installed at `/usr/local/cuda`)
- **Node.js** 14.x or later recommended
- **GCC** 7+ with C++14 support

### Dependencies
- `node-addon-api`: For Node.js native addon support
- `node-gyp`: For building native addons

## Installation

### 1. Install CUDA Toolkit

Download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads).

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify CUDA installation
/usr/local/cuda/bin/nvcc --version
```

Ensure CUDA is installed at `/usr/local/cuda`. If installed elsewhere, create a symlink:
```bash
sudo ln -s /usr/local/cuda-12.0 /usr/local/cuda
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Build the Native Addon

```bash
npx node-gyp configure
npx node-gyp build
```

## Usage

### SLAM (Mapping)

Build a map of an unknown environment:

```bash
# With visualization
node example_mapping.js

# Without visualization (faster)
node example_mapping.js --no-viz
```

This will:
1. Load sensor data from `bigboi_munged.json`
2. Run particle SLAM to build a map
3. Save the final map to `node_baked_map.bin`
4. Display profiling results

### MCL (Localization)

Localize the robot in a known map:

```bash
# With visualization
node example_mcl.js

# Without visualization (faster)
node example_mcl.js --no-viz
```

This will:
1. Load a pre-built map from `node_baked_map.bin`
2. Load sensor data from `one_loop_munged.json`
3. Localize the robot using MCL
4. Display position estimates and profiling results

## API Reference

### ParticleSlam Class

```javascript
const { ParticleSlam } = require('./index');

// Create instance
const slam = new ParticleSlam(
    num_particles,          // Number of particles (e.g., 5000-10000)
    max_trajectory_length,  // Maximum trajectory history per particle
    max_chunk_length,       // Maximum visual measurement chunks to buffer
    cell_size_m,           // Map cell size in meters (default: 0.1)
    pos_weight,            // Weight for positive observations (default: 0.7)
    neg_weight,            // Weight for negative observations (default: 0.4)
    alpha_prior,           // Alpha prior for Beta distribution (default: 1.0)
    beta_prior,            // Beta prior for Beta distribution (default: 1.5)
    measurement_saturation // Max observation counter value (default: 200)
);
```

### Core Methods

#### Initialization
```javascript
// Initialize CUDA memory and RNG
slam.init(random_seed = 1234);
```

#### Motion Update
```javascript
// Apply dead reckoning motion step
slam.apply_step(
    { x: dx, y: dy, z: dtheta }, // Motion delta in robot frame
    timestamp,                     // Timestamp in seconds
    pos_std = 1.6e-3,             // Position noise std dev
    yaw_std = 1e-3                // Yaw noise std dev
);
```

#### Measurement Update
```javascript
// Ingest visual measurement (60x60 occupancy grid)
const chunk = {
    timestamp: ts,
    cells: [[{ num_pos: 0, num_neg: 0 }, ...], ...]  // 60x60 array
};
const chunk_idx = slam.ingest_visual_measurement(chunk);

// For SLAM: accumulate map from particle trajectories
slam.accumulate_map_from_trajectories(chunk_idx);

// For MCL: accumulate map from global map
slam.accumulate_map_from_map(chunk_idx);

// Evaluate and resample particles
slam.evaluate_and_resample(chunk_idx);
```

#### State Estimation
```javascript
// Calculate mean and covariance
const measurement = slam.calculate_measurement();
console.log(measurement.mean);        // { x, y, z }
console.log(measurement.covariance);  // { m: [9 elements] } (row-major 3x3)
console.log(measurement.is_gaussian); // boolean
```

#### Data Retrieval
```javascript
// Download current particle states
const particles = slam.download_current_particle_states();
// Returns: [{ state: { x, y, z }, timestamp }, ...]

// Download particle scores
const scores = slam.download_scores();
// Returns: [score1, score2, ...]

// Download chunk states for a particle
const trajectory = slam.download_chunk_states_for_particle(
    particle_idx,
    num_chunks
);
// Returns: [{ x, y, z }, ...]
```

#### Map Operations
```javascript
// For MCL: Set global map
slam.set_global_map(map);

// For MCL: Uniformly initialize particles across map
slam.uniform_initialize_particles();

// For MCL: Remove particles outside map bounds
slam.prune_particles_outside_map();

// Bake final map from best particle
const final_map = slam.bake_best_particle_map();

// For MCL: Bake global map from best particle
const updated_map = slam.bake_global_map_best_particle();
```

### Utility Functions

```javascript
// Load map from file
const { load_map_from_file } = require('./index');
const map = load_map_from_file('map.bin');

// Save map to file
const { save_map_to_file } = require('./index');
save_map_to_file(map, 'map.bin');
```

### Map Structure
```javascript
{
    width: 800,           // Map width in cells
    height: 600,          // Map height in cells
    min_x: -40.0,         // Minimum X coordinate (meters)
    min_y: -30.0,         // Minimum Y coordinate (meters)
    max_x: 40.0,          // Maximum X coordinate (meters)
    max_y: 30.0,          // Maximum Y coordinate (meters)
    cell_size: 0.1,       // Cell size in meters
    cells: [              // Flattened array of cells (height * width)
        { num_pos: 10, num_neg: 5 },
        // ...
    ]
}
```

## Terminal Visualizer

The included ASCII terminal visualizer provides real-time feedback:

```javascript
const { TermVisualizer } = require('./lib/term_visualizer');

const visualizer = new TermVisualizer({
    width: 120,           // Terminal width in characters
    height: 50,           // Terminal height in characters
    metersPerChar: 0.2,   // Meters per character cell
    maxParticles: 200     // Max particles to render
});

visualizer.init();
visualizer.setMap(map);  // Optional: set background map

visualizer.render(particles, mean, isGaussian, {
    title: 'SLAM Visualization',
    info: ['Position: (1.23, 4.56)', 'Particles: 5000']
});
```

**Legend:**
- `↑↗→↘↓↙←↖` - Blue arrows show particle positions and orientations
- `↑` (Green) - Mean pose when distribution is Gaussian
- `↑` (Yellow) - Mean pose when distribution is non-Gaussian
- `█▓▒░` - Map occupancy (black = occupied, white = free)

## File Structure

```
.
├── ParticleSlam.h          # Main particle filter header
├── ParticleSlam.cu         # CUDA implementation
├── src/
│   └── pswarm_wrapper.cpp  # Node.js addon wrapper
├── lib/
│   └── term_visualizer.js  # ASCII terminal visualizer
├── example_mapping.js      # SLAM example
├── example_mcl.js          # MCL example
├── binding.gyp             # Build configuration
├── index.js                # Node.js entry point
├── index.d.ts              # TypeScript definitions
└── README.md               # This file
```

## Input Data Format

The examples expect JSON files with the following format:

```json
[
  {
    "type": "dr_step",
    "ts": 1234567.89,
    "value": {
      "x": [x, y, theta]  // Robot state (odometry)
    }
  },
  {
    "type": "map_measurement",
    "ts": 1234568.12,
    "value": {
      "cells": [  // 60x60 grid
        [
          {"num_pos": 5, "num_neg": 2},
          null,  // No observation
          // ...
        ],
        // ...
      ]
    }
  }
]
```

## Performance

Typical performance on an NVIDIA RTX 3080 with 10,000 particles:

| Operation | Average Time |
|-----------|-------------|
| apply_step | 0.5-1.0 ms |
| ingest_visual_measurement | 0.3-0.5 ms |
| accumulate_map_from_trajectories | 2-5 ms |
| evaluate_and_resample | 3-8 ms |
| calculate_measurement | 0.5-1.0 ms |
| download_current_particle_states | 0.5-1.0 ms |

**Tips for Performance:**
- Use `--no-viz` flag to disable visualization for maximum speed
- Adjust `NUM_PARTICLES` based on your GPU capability
- Larger `max_trajectory_length` increases memory usage
- Profile with built-in timing to identify bottlenecks

## Troubleshooting

### Build Issues

**Error: `nvcc: command not found`**
- Ensure CUDA Toolkit is installed at `/usr/local/cuda`
- Verify nvcc is accessible: `/usr/local/cuda/bin/nvcc --version`
- If CUDA is installed elsewhere, create a symlink:
  ```bash
  sudo ln -s /path/to/cuda /usr/local/cuda
  ```

**Error: `undefined symbol: _ZN6pswarm...`**
- The build system automatically sets rpath to `/usr/local/cuda/lib64`
- If issues persist, manually add to library path:
  ```bash
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  ```

**Error: Building on Windows**
- Windows is not supported. Use Linux or WSL2 (Windows Subsystem for Linux 2)

### Runtime Issues

**Error: `CUDA error: out of memory`**
- Reduce `NUM_PARTICLES` or `MAX_TRAJECTORY_LENGTH`
- Close other GPU-intensive applications

**Visualization flickering**
- The alternate screen buffer should eliminate flicker
- If issues persist, use `--no-viz` flag

**Slow performance**
- Profile with built-in timing to identify bottlenecks
- Ensure CUDA drivers are up to date
- Check GPU utilization with `nvidia-smi`

## Algorithm Details

### Particle Filter

Each particle maintains:
- Current state: `(x, y, θ)` in world frame
- Trajectory history of states at measurement times
- Individual occupancy map built from observations

### Motion Model

Dead reckoning with Gaussian noise:
```
x' = x + Δx + noise(σ_pos)
y' = y + Δy + noise(σ_pos)
θ' = θ + Δθ + noise(σ_yaw)
```

### Observation Model

60×60 occupancy grid centered on robot:
- Each cell contains positive/negative observation counts
- Beta-Bernoulli model for occupancy probability:
  - `α = α_prior + pos_weight × num_pos`
  - `β = β_prior + neg_weight × num_neg`
  - `P(occupied) = α / (α + β)`

### Resampling

Systematic resampling based on particle scores:
- SLAM: Score based on map consistency
- MCL: Score based on match with global map

## Contributing

Contributions are welcome! Areas for improvement:
- Additional sensor models (lidar, depth cameras)
- Loop closure detection
- Map serialization formats
- More visualization options

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

This implementation uses CUDA for GPU acceleration and is designed for real-time robotics applications.
