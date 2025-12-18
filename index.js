/**
 * @novadynamics/pswarm - GPU-accelerated particle filter for SLAM and MCL
 * 
 * Native Node.js addon wrapping CUDA-based particle SLAM implementation.
 */

const pswarm = require('./build/Release/pswarm.node');

module.exports = {
    ParticleSlam: pswarm.ParticleSlam,
    load_map_from_file: pswarm.load_map_from_file,
    save_map_to_file: pswarm.save_map_to_file
};
