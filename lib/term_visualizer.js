/**
 * Terminal-based ASCII visualizer for particle SLAM
 * Renders occupancy grid map and particle positions in the console
 */

class TermVisualizer {
    /**
     * @param {Object} options - Configuration options
     * @param {number} options.width - Terminal viewport width in characters (default: 100)
     * @param {number} options.height - Terminal viewport height in characters (default: 40)
     * @param {number} options.metersPerChar - Meters per character cell (default: 0.3)
     * @param {number} options.maxParticles - Maximum particles to render (default: 200)
     */
    constructor(options = {}) {
        this.width = options.width || 100;
        this.height = options.height || 40;
        this.metersPerChar = options.metersPerChar || 0.3;
        this.maxParticles = options.maxParticles || 200;
        
        // Occupancy probability to ASCII character mapping
        this.occupancyChars = [' ', '░', '▒', '▓', '█'];
        
        // Directional arrows for particles (8 directions)
        this.arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
        
        this.map = null;
        this.buffer = null;
        this.initialized = false;
    }
    
    /**
     * Initialize the visualizer (enter alternate screen buffer)
     */
    init() {
        if (!this.initialized) {
            // Enter alternate screen buffer and hide cursor
            process.stdout.write('\x1b[?1049h\x1b[?25l');
            this.initialized = true;
        }
    }
    
    /**
     * Cleanup the visualizer (exit alternate screen buffer)
     */
    cleanup() {
        if (this.initialized) {
            // Show cursor and exit alternate screen buffer
            process.stdout.write('\x1b[?25h\x1b[?1049l');
            this.initialized = false;
        }
    }
    
    /**
     * Set the global map for background rendering
     * @param {Object} map - Map object with width, height, cells, min_x, min_y, max_x, max_y, cell_size
     */
    setMap(map) {
        this.map = map;
    }
    
    /**
     * Convert world coordinates to terminal character coordinates
     * @param {number} x - World x coordinate
     * @param {number} y - World y coordinate
     * @param {number} centerX - Center world x
     * @param {number} centerY - Center world y
     * @returns {{col: number, row: number}}
     */
    worldToScreen(x, y, centerX, centerY) {
        const screenX = (x - centerX) / this.metersPerChar;
        const screenY = (centerY - y) / this.metersPerChar;  // Flip y for screen coords
        
        const col = Math.floor(this.width / 2 + screenX);
        const row = Math.floor(this.height / 2 + screenY);
        
        return { col, row };
    }
    
    /**
     * Get occupancy character for a cell
     * @param {Object} cell - Cell with num_pos and num_neg
     * @returns {string} ASCII character
     */
    getOccupancyChar(cell) {
        const total = cell.num_pos + cell.num_neg;
        if (total < 2) return ' ';
        
        // Beta-Bernoulli model
        const alpha = 1.0 + 0.7 * cell.num_pos;
        const beta = 1.5 + 0.4 * cell.num_neg;
        const prob = alpha / (alpha + beta);
        
        // Map probability to character
        const index = Math.min(4, Math.floor(prob * 5));
        return this.occupancyChars[index];
    }
    
    /**
     * Get arrow character for particle orientation
     * @param {number} theta - Angle in radians
     * @returns {string} Arrow character
     */
    getArrowChar(theta) {
        // Normalize angle to [0, 2*PI)
        let angle = theta % (2 * Math.PI);
        if (angle < 0) angle += 2 * Math.PI;
        
        // Convert to 8-direction index (0 = up/north)
        const index = Math.round(angle / (Math.PI / 4)) % 8;
        return this.arrows[index];
    }
    
    /**
     * Render the scene
     * @param {Array} particles - Array of particle objects with state: {x, y, z}
     * @param {Object} mean - Mean pose {x, y, z}
     * @param {boolean} isGaussian - Whether distribution is Gaussian
     * @param {Object} options - Render options
     */
    render(particles, mean, isGaussian = false, options = {}) {
        // Center on map if available, otherwise center on origin
        let centerX = 0;
        let centerY = 0;
        
        if (this.map) {
            // Center on the map's center
            centerX = (this.map.min_x + this.map.max_x) / 2;
            centerY = (this.map.min_y + this.map.max_y) / 2;
        }
        
        // Create buffer
        this.buffer = Array(this.height).fill(null).map(() => 
            Array(this.width).fill(' ')
        );
        
        // Render map background if available
        if (this.map) {
            this.renderMap(centerX, centerY);
        }
        
        // Sample particles if there are too many
        let particlesToRender = particles;
        if (particles.length > this.maxParticles) {
            particlesToRender = this.sampleParticles(particles, this.maxParticles);
        }
        
        // Render particles (blue dots with orientation)
        for (const particle of particlesToRender) {
            const { col, row } = this.worldToScreen(
                particle.state.x, 
                particle.state.y, 
                centerX, 
                centerY
            );
            
            if (col >= 0 && col < this.width && row >= 0 && row < this.height) {
                const arrow = this.getArrowChar(particle.state.z);
                this.buffer[row][col] = `\x1b[34m${arrow}\x1b[0m`;  // Blue
            }
        }
        
        // Render mean pose (green if Gaussian, yellow if not)
        if (mean) {
            const { col, row } = this.worldToScreen(mean.x, mean.y, centerX, centerY);
            if (col >= 0 && col < this.width && row >= 0 && row < this.height) {
                const arrow = this.getArrowChar(mean.z);
                const color = isGaussian ? '\x1b[32m' : '\x1b[33m';  // Green or Yellow
                this.buffer[row][col] = `${color}${arrow}\x1b[0m`;
            }
        }
        
        // Draw to terminal
        this.draw(options);
    }
    
    /**
     * Render map cells in the viewport
     */
    renderMap(centerX, centerY) {
        if (!this.map) return;
        
        // Iterate through map cells and render those in viewport
        for (let my = 0; my < this.map.height; my++) {
            for (let mx = 0; mx < this.map.width; mx++) {
                const idx = my * this.map.width + mx;
                const cell = this.map.cells[idx];
                
                // Calculate world position of cell center
                const worldX = this.map.min_x + (mx + 0.5) * this.map.cell_size;
                const worldY = this.map.min_y + (my + 0.5) * this.map.cell_size;
                
                const { col, row } = this.worldToScreen(worldX, worldY, centerX, centerY);
                
                if (col >= 0 && col < this.width && row >= 0 && row < this.height) {
                    const char = this.getOccupancyChar(cell);
                    if (char !== ' ') {
                        this.buffer[row][col] = `\x1b[37m${char}\x1b[0m`;  // White/gray
                    }
                }
            }
        }
    }
    
    /**
     * Sample random subset of particles
     */
    sampleParticles(particles, count) {
        const sampled = [];
        const step = Math.floor(particles.length / count);
        for (let i = 0; i < particles.length; i += step) {
            sampled.push(particles[i]);
            if (sampled.length >= count) break;
        }
        return sampled;
    }
    
    /**
     * Draw buffer to terminal
     */
    draw(options = {}) {
        const { title, info } = options;
        
        // Ensure visualizer is initialized
        if (!this.initialized) {
            this.init();
        }
        
        // Build entire frame as a string first to reduce flicker
        let output = '';
        
        // Move cursor to top-left
        output += '\x1b[H';
        
        // Draw title
        if (title) {
            output += '\x1b[1m' + title + '\x1b[0m\n';
            output += '─'.repeat(this.width) + '\n';
        }
        
        // Draw buffer
        for (let row = 0; row < this.height; row++) {
            output += this.buffer[row].join('') + '\n';
        }
        
        // Draw info footer
        if (info) {
            output += '─'.repeat(this.width) + '\n';
            for (const line of info) {
                output += line + '\n';
            }
        }
        
        // Draw legend
        output += '\nLegend: \x1b[34m↑\x1b[0m Particles  \x1b[32m↑\x1b[0m Mean(Gaussian)  \x1b[33m↑\x1b[0m Mean(Non-Gaussian)  █▓▒░ Map Occupancy\n';
        
        // Clear to end of screen to handle any leftover content
        output += '\x1b[J';
        
        // Write entire frame at once
        process.stdout.write(output);
    }
    
    /**
     * Clear the terminal
     */
    clear() {
        if (this.initialized) {
            process.stdout.write('\x1b[H\x1b[J');
        } else {
            process.stdout.write('\x1b[2J\x1b[H');
        }
    }
}

module.exports = { TermVisualizer };
