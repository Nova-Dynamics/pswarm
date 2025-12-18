const { ParticleSlam, load_map_from_file } = require('../index');
const { TermVisualizer } = require('../lib/term_visualizer');
const fs = require('fs');

// Check for --no-viz flag to disable visualization
const ENABLE_VISUALIZATION = !process.argv.includes('--no-viz');

// Profiling data
const profiling = {
    apply_step: { total: 0, count: 0 },
    ingest_visual_measurement: { total: 0, count: 0 },
    accumulate_map_from_map: { total: 0, count: 0 },
    accumulate_map_from_trajectories: { total: 0, count: 0 },
    evaluate_and_resample: { total: 0, count: 0 },
    prune_particles_outside_map: { total: 0, count: 0 },
    download_current_particle_states: { total: 0, count: 0 },
    calculate_measurement: { total: 0, count: 0 },
    set_global_map: { total: 0, count: 0 },
    uniform_initialize_particles: { total: 0, count: 0 }
};

function profileFunction(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    profiling[name].total += (end - start);
    profiling[name].count++;
    return result;
}

console.log('Starting MCL Localization example...');
console.log(`Visualization: ${ENABLE_VISUALIZATION ? 'ENABLED' : 'DISABLED (use --no-viz to disable)'}`);

// Load JSON data
console.log('Loading ./data/partial_run.json...');
const data = JSON.parse(fs.readFileSync('./data/partial_run.json', 'utf8'));
console.log(`Loaded ${data.length} entries from JSON`);

// Load pre-existing map if available
let use_global_map = false;
let global_map = null;
const map_filename = './data/baked_map.bin';
if (fs.existsSync(map_filename)) {
    console.log(`Loading map from ${map_filename}...`);
    try {
        global_map = load_map_from_file(map_filename);
        console.log(`Map loaded: ${global_map.width}x${global_map.height} cells`);
        use_global_map = true;
    } catch (err) {
        console.log(`Failed to load map: ${err.message}, continuing without global map`);
    }
}

// Create terminal visualizer
let visualizer = null;
if (ENABLE_VISUALIZATION) {
    visualizer = new TermVisualizer({
        width: 200,
        height: 70,
        metersPerChar: 0.2,
        maxParticles: 1000
    });

    if (use_global_map) {
        visualizer.setMap(global_map);
    }

    // Initialize visualizer (enters alternate screen buffer)
    visualizer.init();

    // Cleanup on exit
    process.on('exit', () => visualizer.cleanup());
    process.on('SIGINT', () => {
        visualizer.cleanup();
        process.exit(0);
    });
}

// Create ParticleSlam instance with MCL parameters
const NUM_PARTICLES = 5000;
const MAX_TRAJECTORY_LENGTH = 300;
const MAX_CHUNK_LENGTH = 60;

console.log(`Creating ParticleSlam with ${NUM_PARTICLES} particles...`);
const mcl_slam = new ParticleSlam(NUM_PARTICLES, MAX_TRAJECTORY_LENGTH, MAX_CHUNK_LENGTH);
console.log('ParticleSlam created successfully');

// Initialize
console.log('Initializing MCL SLAM...');
mcl_slam.init(1234);
console.log('Initialization complete');

// Set global map and uniform initialize particles if map is available
if (use_global_map) {
    console.log('Setting global map for MCL SLAM...');
    profileFunction('set_global_map', () => mcl_slam.set_global_map(global_map));
    
    console.log('Uniformly initializing particles across map...');
    profileFunction('uniform_initialize_particles', () => mcl_slam.uniform_initialize_particles());
} else {
    console.log('No global map available - particles will start from origin');
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

console.log('Processing JSON entries...');

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

            profileFunction('apply_step', () => mcl_slam.apply_step(dx_step, ts, 0.1, 0.003));
            
            if (use_global_map) {
                profileFunction('prune_particles_outside_map', () => mcl_slam.prune_particles_outside_map());
            }
            
            const measurement = profileFunction('calculate_measurement', () => mcl_slam.calculate_measurement());
            dr_step_count++;
            
            // Visualize every 10 steps
            if (ENABLE_VISUALIZATION && dr_step_count % 10 === 0) {
                const particles = profileFunction('download_current_particle_states', () => mcl_slam.download_current_particle_states());
                
                visualizer.render(particles, measurement.mean, measurement.is_gaussian, {
                    title: `MCL Localization - Step ${dr_step_count}`,
                    info: [
                        `Position: (${measurement.mean.x.toFixed(2)}, ${measurement.mean.y.toFixed(2)}, ${measurement.mean.z.toFixed(3)} rad)`,
                        `Particles: ${NUM_PARTICLES} (showing ${Math.min(200, NUM_PARTICLES)})`,
                        `Distribution: ${measurement.is_gaussian ? 'Gaussian ✓' : 'Non-Gaussian'}`,
                        `Measurements processed: ${measurement_count}`
                    ]
                });
            }
            
            // // Print detailed stats every 100 steps
            // if (dr_step_count % 100 === 0) {
            //     console.log(`\nStep ${dr_step_count}:`);
            //     console.log('  Mean:', measurement.mean);
            //     console.log('  Covariance:');
            //     console.log('    [', measurement.covariance.m.slice(0, 3).join(', '), ']');
            //     console.log('    [', measurement.covariance.m.slice(3, 6).join(', '), ']');
            //     console.log('    [', measurement.covariance.m.slice(6, 9).join(', '), ']');
            //     console.log('  Is Gaussian:', measurement.is_gaussian);
            // }
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

        const c_idx = profileFunction('ingest_visual_measurement', () => mcl_slam.ingest_visual_measurement(chunk));
        if (c_idx === -1) {
            console.error(`Failed to ingest chunk at timestamp ${ts}`);
            continue;
        }

        if (use_global_map) {
            profileFunction('accumulate_map_from_map', () => mcl_slam.accumulate_map_from_map(c_idx));
        } else {
            profileFunction('accumulate_map_from_trajectories', () => mcl_slam.accumulate_map_from_trajectories(c_idx));
        }
        profileFunction('evaluate_and_resample', () => mcl_slam.evaluate_and_resample(c_idx));
        measurement_count++;
        
    }
}

console.log('\n========== RESULTS ==========');
console.log(`DR Steps: ${dr_step_count}`);
console.log(`Map Measurements: ${measurement_count}`);

// Final measurement and visualization
const final_measurement = profileFunction('calculate_measurement', () => mcl_slam.calculate_measurement());
const final_particles = profileFunction('download_current_particle_states', () => mcl_slam.download_current_particle_states());

if (ENABLE_VISUALIZATION) {
    visualizer.render(final_particles, final_measurement.mean, final_measurement.is_gaussian, {
        title: 'FINAL STATE - MCL Localization Complete',
        info: [
            `Total DR Steps: ${dr_step_count}`,
            `Total Map Measurements: ${measurement_count}`,
            `Final Position: (${final_measurement.mean.x.toFixed(2)}, ${final_measurement.mean.y.toFixed(2)}, ${final_measurement.mean.z.toFixed(3)} rad)`,
            `Distribution: ${final_measurement.is_gaussian ? 'Gaussian ✓' : 'Non-Gaussian'}`,
            ''
        ]
    });
}

console.log('\nFinal Measurement:');
console.log('  Mean:', final_measurement.mean);
console.log('  Covariance:');
console.log('    [', final_measurement.covariance.m.slice(0, 3).join(', '), ']');
console.log('    [', final_measurement.covariance.m.slice(3, 6).join(', '), ']');
console.log('    [', final_measurement.covariance.m.slice(6, 9).join(', '), ']');
console.log('  Is Gaussian:', final_measurement.is_gaussian);

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
    console.log('Example complete - Press Ctrl+C to exit...');
} else {
    console.log('Example complete!');
}