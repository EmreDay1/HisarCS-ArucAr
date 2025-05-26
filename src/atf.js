/**
 * Adaptive Threshold Filter (ATF)
 * 
 * This library provides adaptive thresholding functionality to improve
 * marker detection in AR applications, especially when markers are viewed
 * from sharp angles where lighting across the marker is uneven.
 * 
 * The adaptive thresholding approach calculates thresholds locally for
 * each pixel based on the surrounding region, making it much more robust
 * to lighting variations than global thresholding.
 */

class AdaptiveThresholdFilter {
    /**
     * Create a new adaptive threshold filter
     * @param {Object} options - Configuration options
     * @param {number} options.blockSize - Size of local region for calculating threshold (must be odd)
     * @param {number} options.C - Constant subtracted from mean (positive = darker threshold)
     * @param {boolean} options.useIntegralImage - Whether to use integral image optimization (faster but more memory)
     * @param {boolean} options.debug - Enable debug mode to visualize filter steps
     */
    constructor(options = {}) {
      this.blockSize = options.blockSize || 11;
      this.C = options.C !== undefined ? options.C : 5;
      this.useIntegralImage = options.useIntegralImage !== undefined ? options.useIntegralImage : true;
      this.debug = options.debug || false;
      
      // Ensure block size is odd
      if (this.blockSize % 2 === 0) {
        this.blockSize += 1;
      }
      
      // For performance tracking
      this.lastProcessTime = 0;
      
      // Debug data
      this.debugData = null;
    }
    
    /**
     * Apply adaptive thresholding to image data
     * @param {ImageData} imageData - Input image data
     * @param {Object} options - Override default options
     * @returns {ImageData} Processed binary image
     */
    process(imageData, options = {}) {
      const startTime = performance.now();
      
      // Override instance options with method options if provided
      const blockSize = options.blockSize || this.blockSize;
      const C = options.C !== undefined ? options.C : this.C;
      const useIntegralImage = options.useIntegralImage !== undefined ? 
        options.useIntegralImage : this.useIntegralImage;
      
      // Convert to grayscale first
      const grayData = this._toGrayscale(imageData);
      
      // Process with appropriate method
      let result;
      if (useIntegralImage) {
        result = this._processWithIntegralImage(grayData, blockSize, C);
      } else {
        result = this._processWithSlidingWindow(grayData, blockSize, C);
      }
      
      this.lastProcessTime = performance.now() - startTime;
      
      if (this.debug) {
        console.log(`Adaptive threshold took ${this.lastProcessTime.toFixed(2)}ms`);
        // Store gray image for debug visualization
        this.debugData = {
          grayscale: grayData,
          result: result
        };
      }
      
      return result;
    }
    
    /**
     * Process image using sliding window approach
     * Simpler but slower implementation
     * @private
     */
    _processWithSlidingWindow(grayData, blockSize, C) {
      const width = grayData.width;
      const height = grayData.height;
      const radius = Math.floor(blockSize / 2);
      const result = new ImageData(width, height);
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          // Calculate local mean in the block around (x,y)
          let sum = 0, count = 0;
          
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              const nx = x + dx;
              const ny = y + dy;
              
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                sum += grayData.data[(ny * width + nx) * 4];
                count++;
              }
            }
          }
          
          const mean = sum / count;
          const i = (y * width + x) * 4;
          const pixel = grayData.data[i];
          
          // Apply threshold (pixel < mean-C ? 0 : 255)
          const value = pixel < (mean - C) ? 0 : 255;
          result.data[i] = result.data[i+1] = result.data[i+2] = value;
          result.data[i+3] = 255;
        }
      }
      
      return result;
    }
    
    /**
     * Process image using integral image approach
     * More complex but significantly faster for large block sizes
     * @private
     */
    _processWithIntegralImage(grayData, blockSize, C) {
      const width = grayData.width;
      const height = grayData.height;
      const radius = Math.floor(blockSize / 2);
      const result = new ImageData(width, height);
      
      // Create integral image (summed area table)
      const integral = new Float32Array(width * height);
      
      // First pixel
      integral[0] = grayData.data[0];
      
      // First row
      for (let x = 1; x < width; x++) {
        integral[x] = integral[x-1] + grayData.data[x * 4];
      }
      
      // First column
      for (let y = 1; y < height; y++) {
        integral[y * width] = integral[(y-1) * width] + grayData.data[y * width * 4];
      }
      
      // Rest of the image
      for (let y = 1; y < height; y++) {
        for (let x = 1; x < width; x++) {
          integral[y * width + x] = 
            grayData.data[(y * width + x) * 4] +
            integral[y * width + (x-1)] +
            integral[(y-1) * width + x] -
            integral[(y-1) * width + (x-1)];
        }
      }
      
      // Apply adaptive threshold using integral image
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          // Define bounds for local area
          const x1 = Math.max(0, x - radius);
          const y1 = Math.max(0, y - radius);
          const x2 = Math.min(width - 1, x + radius);
          const y2 = Math.min(height - 1, y + radius);
          
          // Calculate area and sum using integral image
          const area = (x2 - x1 + 1) * (y2 - y1 + 1);
          const sum = integral[y2 * width + x2] -
                     (x1 > 0 ? integral[y2 * width + (x1-1)] : 0) -
                     (y1 > 0 ? integral[(y1-1) * width + x2] : 0) +
                     (x1 > 0 && y1 > 0 ? integral[(y1-1) * width + (x1-1)] : 0);
          
          const mean = sum / area;
          const i = (y * width + x) * 4;
          const pixel = grayData.data[i];
          
          // Apply threshold (pixel < mean-C ? 0 : 255)
          const value = pixel < (mean - C) ? 0 : 255;
          result.data[i] = result.data[i+1] = result.data[i+2] = value;
          result.data[i+3] = 255;
        }
      }
      
      return result;
    }
    
    /**
     * Convert RGB image data to grayscale
     * @private
     */
    _toGrayscale(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const result = new ImageData(width, height);
      
      for (let i = 0; i < imageData.data.length; i += 4) {
        // Use standard luminance formula
        const gray = (
          0.299 * imageData.data[i] + 
          0.587 * imageData.data[i+1] + 
          0.114 * imageData.data[i+2]
        );
        
        result.data[i] = result.data[i+1] = result.data[i+2] = gray;
        result.data[i+3] = 255;
      }
      
      return result;
    }
    
    /**
     * Creates a visualization of the thresholding process for debugging
     * @param {CanvasRenderingContext2D} context - Canvas context to draw on
     * @param {number} x - X position to draw at
     * @param {number} y - Y position to draw at
     */
    visualizeDebug(context, x = 0, y = 0) {
      if (!this.debug || !this.debugData) {
        console.warn('Debug mode is not enabled or no data has been processed yet');
        return;
      }
      
      const { grayscale, result } = this.debugData;
      const width = grayscale.width;
      const height = grayscale.height;
      
      // Draw original grayscale
      context.putImageData(grayscale, x, y);
      
      // Draw processed result
      context.putImageData(result, x + width + 10, y);
      
      // Add labels
      context.fillStyle = 'white';
      context.font = '12px monospace';
      context.fillText('Grayscale', x, y + height + 15);
      context.fillText('Thresholded', x + width + 10, y + height + 15);
      context.fillText(`Process time: ${this.lastProcessTime.toFixed(2)}ms`, x, y + height + 35);
    }
    
    /**
     * Set new filter parameters
     * @param {Object} options - Parameters to update
     */
    setOptions(options = {}) {
      if (options.blockSize !== undefined) {
        this.blockSize = options.blockSize % 2 === 0 ? 
          options.blockSize + 1 : options.blockSize;
      }
      
      if (options.C !== undefined) {
        this.C = options.C;
      }
      
      if (options.useIntegralImage !== undefined) {
        this.useIntegralImage = options.useIntegralImage;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
    }
    
    /**
     * Get the processing time of the last operation
     * @returns {number} Processing time in milliseconds
     */
    getProcessTime() {
      return this.lastProcessTime;
    }
  }
  
  /**
   * Utility function to quickly apply adaptive thresholding
   * @param {ImageData} imageData - Input image data
   * @param {number} blockSize - Size of local region (must be odd)
   * @param {number} C - Constant subtracted from mean
   * @returns {ImageData} Processed binary image
   */
  function adaptiveThreshold(imageData, blockSize = 11, C = 5) {
    const filter = new AdaptiveThresholdFilter({
      blockSize: blockSize,
      C: C
    });
    
    return filter.process(imageData);
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      AdaptiveThresholdFilter,
      adaptiveThreshold
    };
  } else {
    // Browser global
    window.AdaptiveThresholdFilter = AdaptiveThresholdFilter;
    window.adaptiveThreshold = adaptiveThreshold;
  }