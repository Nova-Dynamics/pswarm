const { ParticleSlam, save_map_to_file } = require('../index');
const { TermVisualizer } = require('../lib/term_visualizer');
const fs = require('fs');

// Check for --no-viz flag to disable visualization
const ENABLE_VISUALIZATION = !process.argv.includes('--no-viz');

// Profiling data
const profiling = {
    apply_step: { total: 0, count: 0 },
    ingest_visual_measurement: { total: 0, count: 0 },
    accumulate_map_from_trajectories: { total: 0, count: 0 },
    evaluate_and_resample: { total: 0, count: 0 },
    download_current_particle_states: { total: 0, count: 0 },
    download_scores: { total: 0, count: 0 },
    download_chunk_states_for_particle: { total: 0, count: 0 },
    calculate_measurement: { total: 0, count: 0 },
    bake_best_particle_map: { total: 0, count: 0 },
    get_current_chunk_count: { total: 0, count: 0 }
};

function profileFunction(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    profiling[name].total += (end - start);
    profiling[name].count++;
    return result;
}

console.log('Starting Particle SLAM Mapping example...');
console.log(`Visualization: ${ENABLE_VISUALIZATION ? 'ENABLED' : 'DISABLED (use --no-viz to disable)'}`);

// Load JSON data
console.log('Loading ./data/full_run.json...');
const data = JSON.parse(fs.readFileSync('./data/full_run.json', 'utf8'));
console.log(`Loaded ${data.length} entries from JSON`);

// Create ParticleSlam instance with SLAM parameters
const NUM_PARTICLES = 10000;
const MAX_TRAJECTORY_LENGTH = 1000;
const MAX_CHUNK_LENGTH = 3600;

console.log(`Creating ParticleSlam with ${NUM_PARTICLES} particles for mapping...`);
const slam = new ParticleSlam(NUM_PARTICLES, MAX_TRAJECTORY_LENGTH, MAX_CHUNK_LENGTH);
console.log('ParticleSlam created successfully');

// Initialize
console.log('Initializing SLAM...');
slam.init(1234);
console.log('Initialization complete');

// Create terminal visualizer for trajectory visualization
let visualizer = null;
if (ENABLE_VISUALIZATION) {
    visualizer = new TermVisualizer({
        width: 120,
        height: 50,
        metersPerChar: 0.15,
        maxParticles: 100  // Show fewer particles for trajectory view
    });

    // Initialize visualizer (enters alternate screen buffer)
    visualizer.init();

    // Cleanup on exit
    process.on('exit', () => visualizer.cleanup());
    process.on('SIGINT', () => {
        visualizer.cleanup();
        process.exit(0);
    });
}

// Track state for dx_step calculation
let prev_state = { x: 0.0, y: 0.0, z: 0.0 };
let first_dr_step = true;
let dr_step_count = 0;
let measurement_count = 0;

// Helper function to create rotation matrix from theta
function get_R_from_theta(theta) {
    const cos_theta = Math.cos(theta);
    const sin_theta = Math.sin(theta);
    return {
        m00: cos_theta, m01: -sin_theta,
        m10: sin_theta, m11: cos_theta
    };
}

// Helper function for matrix transpose and multiply
function transpose_multiply(R, delta) {
    // R^T * delta
    return {
        x: R.m00 * delta.x + R.m10 * delta.y,
        y: R.m01 * delta.x + R.m11 * delta.y
    };
}

console.log('Processing JSON entries for SLAM mapping...');

// Process all entries from JSON
for (let i = 0; i < data.length; i++) {
    const entry = data[i];
    const type = entry.type;
    const ts = entry.ts;

    if (type === 'dr_step') {
        const cur_state = {
            x: entry.value.x[0],
            y: entry.value.x[1],
            z: entry.value.x[2]
        };

        if (!first_dr_step) {
            const theta = -prev_state.z;
            const R = get_R_from_theta(theta);
            const state_delta = {
                x: cur_state.x - prev_state.x,
                y: cur_state.y - prev_state.y
            };
            const mean_delta = transpose_multiply(R, state_delta);
            const theta_delta = cur_state.z - prev_state.z;

            const dx_step = {
                x: mean_delta.x,
                y: mean_delta.y,
                z: theta_delta
            };

            // SLAM noise parameters: small noise for mapping (uses defaults)
            profileFunction('apply_step', () => slam.apply_step(dx_step, ts));
            dr_step_count++;
        } else {
            first_dr_step = false;
        }

        prev_state = cur_state;
    } else if (type === 'map_measurement') {
        // Package map measurement into chunk structure
        const chunk = {
            timestamp: ts,
            cells: []
        };

        // Initialize cells
        for (let i = 0; i < 60; i++) {
            chunk.cells[i] = [];
            for (let j = 0; j < 60; j++) {
                chunk.cells[i][j] = { num_pos: 0, num_neg: 0 };
            }
        }

        // Fill in cells from JSON
        const cells = entry.value.cells;
        for (let i = 0; i < 60; i++) {
            for (let j = 0; j < 60; j++) {
                if (cells[i][j] !== null) {
                    chunk.cells[i][j].num_pos = cells[i][j].num_pos;
                    chunk.cells[i][j].num_neg = cells[i][j].num_neg;
                }
            }
        }

        const c_idx = profileFunction('ingest_visual_measurement', () => slam.ingest_visual_measurement(chunk));
        if (c_idx === -1) {
            console.error(`Failed to ingest chunk at timestamp ${ts}`);
            continue;
        }

        profileFunction('accumulate_map_from_trajectories', () => slam.accumulate_map_from_trajectories(c_idx));
        profileFunction('evaluate_and_resample', () => slam.evaluate_and_resample(c_idx));
        measurement_count++;

        // Visualize trajectories every 5 measurements
        if (ENABLE_VISUALIZATION && measurement_count % 5 === 0) {
            const num_chunks = profileFunction('get_current_chunk_count', () => slam.get_current_chunk_count());
            const scores = profileFunction('download_scores', () => slam.download_scores());
            
            const current_states = profileFunction('download_current_particle_states', () => slam.download_current_particle_states());
            const measurement = profileFunction('calculate_measurement', () => slam.calculate_measurement());
            
            visualizer.render(current_states, measurement.mean, measurement.is_gaussian, {
                title: `Particle SLAM Mapping - Measurement ${measurement_count}/${data.filter(e => e.type === 'map_measurement').length}`,
                info: [
                    `DR Steps: ${dr_step_count} | Chunks: ${num_chunks}`,
                    `Position: (${measurement.mean.x.toFixed(2)}, ${measurement.mean.y.toFixed(2)}, ${measurement.mean.z.toFixed(3)} rad)`,
                    `Particles: ${NUM_PARTICLES} (showing ${Math.min(100, NUM_PARTICLES)})`,
                    `Best Score: ${Math.max(...scores).toFixed(4)}`
                ]
            });
        }
    }
}

console.log('\n========== MAPPING COMPLETE ==========');
console.log(`DR Steps: ${dr_step_count}`);
console.log(`Map Measurements: ${measurement_count}`);

// Bake final map from best particle
console.log('\nBaking final map from best particle...');
const final_map = profileFunction('bake_best_particle_map', () => slam.bake_best_particle_map());
console.log(`Final map size: ${final_map.width}x${final_map.height} cells`);
console.log(`Map bounds: (${final_map.min_x.toFixed(2)}, ${final_map.min_y.toFixed(2)}) to (${final_map.max_x.toFixed(2)}, ${final_map.max_y.toFixed(2)})`);
console.log(`Cell size: ${final_map.cell_size} meters`);

// Save map to file
console.log('\nSaving map to ./data/baked_map.bin...');
try {
    save_map_to_file(final_map, './data/baked_map.bin');
    console.log('Map saved successfully!');
} catch (err) {
    console.error('Failed to save map:', err.message);
}

// Get best particle trajectory for final visualization
const num_chunks = profileFunction('get_current_chunk_count', () => slam.get_current_chunk_count());
const scores = profileFunction('download_scores', () => slam.download_scores());
const best_particle_idx = scores.indexOf(Math.max(...scores));
const best_trajectory = profileFunction('download_chunk_states_for_particle', () => slam.download_chunk_states_for_particle(best_particle_idx, num_chunks));

console.log(`\nBest particle trajectory: ${num_chunks} waypoints`);
console.log(`Start: (${best_trajectory[0].x.toFixed(2)}, ${best_trajectory[0].y.toFixed(2)})`);
console.log(`End: (${best_trajectory[num_chunks-1].x.toFixed(2)}, ${best_trajectory[num_chunks-1].y.toFixed(2)})`);

// Final visualization with map
if (ENABLE_VISUALIZATION) {
    visualizer.setMap(final_map);
    const final_particles = profileFunction('download_current_particle_states', () => slam.download_current_particle_states());
    const final_measurement = profileFunction('calculate_measurement', () => slam.calculate_measurement());

    visualizer.render(final_particles, final_measurement.mean, final_measurement.is_gaussian, {
        title: 'FINAL MAPPED ENVIRONMENT',
        info: [
            `Total DR Steps: ${dr_step_count}`,
            `Total Map Measurements: ${measurement_count}`,
            `Map: ${final_map.width}x${final_map.height} cells (${final_map.cell_size}m resolution)`,
            `Final Position: (${final_measurement.mean.x.toFixed(2)}, ${final_measurement.mean.y.toFixed(2)}, ${final_measurement.mean.z.toFixed(3)} rad)`,
            `Best Score: ${Math.max(...scores).toFixed(4)}`,
            '',
            'Map saved to: ./data/baked_map.bin'
        ]
    });
}

// Print profiling results
console.log('\n========== PROFILING RESULTS ==========');
for (const [name, data] of Object.entries(profiling)) {
    if (data.count > 0) {
        const avg = data.total / data.count;
        console.log(`${name}:`);
        console.log(`  Calls: ${data.count}`);
        console.log(`  Avg: ${avg.toFixed(3)} ms`);
        console.log(`  Total: ${data.total.toFixed(3)} ms`);
    }
}
console.log('========================================\n');

if (ENABLE_VISUALIZATION) {
    console.log('Press Ctrl+C to exit...');
} else {
    console.log('Mapping complete!');
}
