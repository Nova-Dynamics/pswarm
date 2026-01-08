#!/usr/bin/env python3
"""
Map Visualization Script
Loads a binary map file and renders it to a PNG image using the same
color scheme as the C++ implementation (HSL interpolation from black to orange).
"""

import argparse
import struct
import sys
import numpy as np
import cv2
from pathlib import Path
import colorsys


MAP_FILE_MAGIC = 0x4D415053  # "MAPS" in ASCII
MAP_FILE_VERSION = 1


class ChunkCell:
    """Represents a single cell in the map."""
    def __init__(self, num_pos=0, num_neg=0):
        self.num_pos = num_pos
        self.num_neg = num_neg


class Map:
    """Map structure matching the C++ implementation."""
    def __init__(self):
        self.cells = None
        self.width = 0
        self.height = 0
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        self.cell_size = 0.1


def load_map_from_file(filename):
    """
    Load map from binary file.
    
    File format:
    - uint32: magic number (0x4D415053 = "MAPS")
    - uint32: version
    - uint32: header size
    - int32: width
    - int32: height
    - float32: cell_size
    - float32: min_x
    - float32: min_y
    - float32: max_x
    - float32: max_y
    - ChunkCell[width*height]: cell data (each cell is 2 uint8: num_pos, num_neg)
    
    Args:
        filename: Path to the binary map file
        
    Returns:
        Map object or None on failure
    """
    try:
        with open(filename, 'rb') as f:
            # Read and verify header
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            header_size = struct.unpack('<I', f.read(4))[0]
            
            if magic != MAP_FILE_MAGIC:
                print(f"Error: Invalid file format (bad magic number: {hex(magic)})", file=sys.stderr)
                return None
            
            if version > MAP_FILE_VERSION:
                print(f"Error: File version {version} is newer than supported version {MAP_FILE_VERSION}", 
                      file=sys.stderr)
                return None
            
            # Skip any extra header data (for forward compatibility)
            expected_header_size = 4 * 3  # magic + version + header_size
            if header_size > expected_header_size:
                f.seek(header_size - expected_header_size, 1)  # Seek relative to current position
            
            # Read map metadata
            width = struct.unpack('<i', f.read(4))[0]
            height = struct.unpack('<i', f.read(4))[0]
            cell_size = struct.unpack('<f', f.read(4))[0]
            min_x = struct.unpack('<f', f.read(4))[0]
            min_y = struct.unpack('<f', f.read(4))[0]
            max_x = struct.unpack('<f', f.read(4))[0]
            max_y = struct.unpack('<f', f.read(4))[0]
            
            if width <= 0 or height <= 0:
                print(f"Error: Invalid map dimensions ({width} x {height})", file=sys.stderr)
                return None
            
            # Allocate map
            map_obj = Map()
            map_obj.width = width
            map_obj.height = height
            map_obj.cell_size = cell_size
            map_obj.min_x = min_x
            map_obj.min_y = min_y
            map_obj.max_x = max_x
            map_obj.max_y = max_y
            
            # Read cell data
            num_cells = width * height
            cell_data = f.read(num_cells * 2)  # Each cell is 2 bytes (uint8 + uint8)
            
            if len(cell_data) != num_cells * 2:
                print(f"Error: Incomplete cell data (expected {num_cells * 2} bytes, got {len(cell_data)})", 
                      file=sys.stderr)
                return None
            
            # Parse cell data into 2D array
            map_obj.cells = np.zeros((height, width), dtype=[('num_pos', np.uint8), ('num_neg', np.uint8)])
            for i in range(num_cells):
                y = i // width
                x = i % width
                map_obj.cells[y, x]['num_pos'] = cell_data[i * 2]
                map_obj.cells[y, x]['num_neg'] = cell_data[i * 2 + 1]
            
            print(f"Loaded map: {width} x {height} ({num_cells} cells, {num_cells * 2 / 1024:.2f} KB)")
            print(f"  Cell size: {cell_size} meters")
            print(f"  Bounds: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
            
            return map_obj
            
    except FileNotFoundError:
        print(f"Error: File not found: {filename}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading map: {e}", file=sys.stderr)
        return None


def interpolate_hue(h1, h2, frac):
    """Interpolate hue with circular interpolation (0-360 degrees)."""
    d = h2 - h1
    if h1 > h2:
        # Swap
        h1, h2 = h2, h1
        d = -d
        frac = 1.0 - frac
    
    if d > 180.0:
        # Go the other way around the color wheel
        h1 = h1 + 360.0
        h = h1 + frac * (h2 - h1)
        if h > 360.0:
            h -= 360.0
        return h
    
    return h1 + frac * d


def interpolate_hsl(c1_bgr, c2_bgr, frac):
    """
    Interpolate between two BGR colors in HSL space.
    
    Args:
        c1_bgr: Tuple (B, G, R) for first color
        c2_bgr: Tuple (B, G, R) for second color
        frac: Interpolation fraction (0.0 to 1.0)
        
    Returns:
        Tuple (B, G, R) for interpolated color
    """
    # Convert BGR to RGB (0-1 range)
    c1_r, c1_g, c1_b = c1_bgr[2] / 255.0, c1_bgr[1] / 255.0, c1_bgr[0] / 255.0
    c2_r, c2_g, c2_b = c2_bgr[2] / 255.0, c2_bgr[1] / 255.0, c2_bgr[0] / 255.0
    
    # Convert to HLS
    c1_h, c1_l, c1_s = colorsys.rgb_to_hls(c1_r, c1_g, c1_b)
    c2_h, c2_l, c2_s = colorsys.rgb_to_hls(c2_r, c2_g, c2_b)
    
    # Interpolate in HLS space
    h = interpolate_hue(c1_h * 360.0, c2_h * 360.0, frac) / 360.0
    l = c1_l * (1.0 - frac) + c2_l * frac
    s = c1_s * (1.0 - frac) + c2_s * frac
    
    # Convert back to RGB
    res_r, res_g, res_b = colorsys.hls_to_rgb(h, l, s)
    
    # Return as BGR for OpenCV
    return (int(res_b * 255), int(res_g * 255), int(res_r * 255))


def render_map(map_obj, pixels_per_cell=5, min_observations=2):
    """
    Render the map to an OpenCV image using HSL interpolation from black to orange.
    
    Args:
        map_obj: Map object to render
        pixels_per_cell: Number of pixels per map cell
        min_observations: Minimum number of observations to render a cell
        
    Returns:
        OpenCV image (numpy array)
    """
    img_width = map_obj.width * pixels_per_cell
    img_height = map_obj.height * pixels_per_cell
    
    # Create white background
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Color scheme: black (unoccupied) to orange (occupied)
    start_color = (0, 0, 0)       # Black (BGR)
    stop_color = (0, 165, 255)    # Orange (BGR)
    
    # Render each cell
    for y in range(map_obj.height):
        for x in range(map_obj.width):
            cell = map_obj.cells[y, x]
            num_pos = int(cell['num_pos'])
            num_neg = int(cell['num_neg'])
            total_obs = num_pos + num_neg
            
            if total_obs >= min_observations:
                # Calculate occupancy probability using Beta-Bernoulli model
                alpha = 1.0 + 0.7 * num_pos
                beta = 1.5 + 0.4 * num_neg
                prob = alpha / (alpha + beta)
                
                # Interpolate color
                color = interpolate_hsl(start_color, stop_color, prob)
                
                # Draw rectangle (note: y is flipped for image coordinates)
                px = x * pixels_per_cell
                py = (map_obj.height - 1 - y) * pixels_per_cell
                cv2.rectangle(img,
                            (px, py),
                            (px + pixels_per_cell, py + pixels_per_cell),
                            color,
                            -1)  # Filled rectangle
    
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Visualize binary map files as PNG images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input_bin', type=str,
                       help='Path to input binary map file (.bin)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Path to output PNG file (default: input_name.png)')
    parser.add_argument('-p', '--pixels-per-cell', type=int, default=5,
                       help='Number of pixels per map cell')
    parser.add_argument('-m', '--min-observations', type=int, default=2,
                       help='Minimum observations required to render a cell')
    parser.add_argument('-s', '--show', action='store_true',
                       help='Display the rendered map in a window')
    
    args = parser.parse_args()
    
    # Load map
    print(f"Loading map from {args.input_bin}...")
    map_obj = load_map_from_file(args.input_bin)
    
    if map_obj is None:
        print("Failed to load map!", file=sys.stderr)
        return 1
    
    # Render map
    print("Rendering map...")
    img = render_map(map_obj, args.pixels_per_cell, args.min_observations)
    
    # Determine output path
    if args.output is None:
        input_path = Path(args.input_bin)
        output_path = input_path.with_suffix('.png')
    else:
        output_path = Path(args.output)
    
    # Save image
    print(f"Saving to {output_path}...")
    cv2.imwrite(str(output_path), img)
    print(f"Map visualization saved successfully! ({img.shape[1]}x{img.shape[0]} pixels)")
    
    # Optionally display
    if args.show:
        print("Displaying map (press any key to close)...")
        cv2.imshow("Map Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
