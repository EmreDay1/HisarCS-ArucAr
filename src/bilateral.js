/**
 * Bilateral Filter for AR Applications
 * 
 * This module provides edge-preserving noise reduction to improve marker detection
 * in low-quality camera feeds while preserving important edge information.
 * 
 * The bilateral filter smooths images while preserving edges by combining
 * domain filtering (based on pixel distance) and range filtering (based on pixel value differences).
 * 
 * Key features:
 * - Edge-preserving noise reduction
 * - Removes noise while maintaining marker boundaries
 * - Improves marker detection in textured backgrounds
 * - Configurable intensity and spatial parameters
 */

class BilateralFilter {
    /**
     * Create a new bilateral filter
     * @param {Object} options - Configuration options
     * @param {Number} options.diameter - Diameter of each pixel neighborhood (must be odd)
     * @param {Number} options.sigmaColor - Standard deviation in the color space
     * @param {Number} options.sigmaSpace - Standard deviation in the coordinate space
     * @param {Boolean} options.useOptimizedKernel - Use pre-computed kernels for better performance
     * @param {Boolean} options.greyOnly - Process only greyscale information (faster)
     * @param {Boolean} options.debug - Enable debug mode
     */
    constructor(options = {}) {
      this.diameter = options.diameter || 9;
      this.sigmaColor = options.sigmaColor || 75;
      this.sigmaSpace = options.sigmaSpace || 75;
      this.useOptimizedKernel = options.useOptimizedKernel !== undefined ? options.useOptimizedKernel : true;
      this.greyOnly = options.greyOnly !== undefined ? options.greyOnly : true;
      this.debug = options.debug || false;
      
      // Ensure diameter is odd
      if (this.diameter % 2 === 0) {
        this.diameter += 1;
      }
      
      // Pre-compute the spatial kernel for optimization
      this.radius = Math.floor(this.diameter / 2);
      this.spatialKernel = this.useOptimizedKernel ? this.precomputeSpatialKernel() : null;
      
      // Cache for intermediate processing steps (for debug visualization)
      this.debugData = {
        original: null,
        greyscale: null,
        filtered: null,
        processingTime: 0
      };
    }
    
    /**
     * Apply bilateral filtering to an image
     * @param {ImageData} imageData - Original image data
     * @param {Object} options - Override default options
     * @returns {ImageData} Filtered image
     */
    process(imageData, options = {}) {
      const startTime = performance.now();
      
      // Store original for debugging
      if (this.debug) {
        this.debugData.original = new ImageData(
          new Uint8ClampedArray(imageData.data), 
          imageData.width, 
          imageData.height
        );
      }
      
      // Convert to grayscale if needed
      let processedImageData;
      if (this.greyOnly) {
        const greyscale = this.toGreyscale(imageData);
        if (this.debug) {
          this.debugData.greyscale = greyscale;
        }
        processedImageData = this.applyBilateralFilter(greyscale);
      } else {
        processedImageData = this.applyBilateralFilterRGB(imageData);
      }
      
      // Store processing time for debugging
      if (this.debug) {
        this.debugData.processingTime = performance.now() - startTime;
        this.debugData.filtered = processedImageData;
      }
      
      return processedImageData;
    }
    
    /**
     * Pre-compute the spatial kernel for optimization
     * @private
     */
    precomputeSpatialKernel() {
      const kernel = new Float32Array((this.diameter * 2 + 1) * (this.diameter * 2 + 1));
      const twoSigmaSquare = 2 * this.sigmaSpace * this.sigmaSpace;
      
      let idx = 0;
      for (let y = -this.radius; y <= this.radius; y++) {
        for (let x = -this.radius; x <= this.radius; x++) {
          const distance = x * x + y * y;
          kernel[idx++] = Math.exp(-distance / twoSigmaSquare);
        }
      }
      
      return kernel;
    }
    
    /**
     * Convert RGB image to greyscale
     * @private
     */
    toGreyscale(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const result = new ImageData(width, height);
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          
          // Standard luminance formula
          const grey = Math.round(
            0.299 * imageData.data[i] + 
            0.587 * imageData.data[i+1] + 
            0.114 * imageData.data[i+2]
          );
          
          result.data[i] = result.data[i+1] = result.data[i+2] = grey;
          result.data[i+3] = imageData.data[i+3]; // Preserve alpha
        }
      }
      
      return result;
    }
    
    /**
     * Apply bilateral filter to a greyscale image
     * @private
     */
    applyBilateralFilter(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const result = new ImageData(width, height);
      
      // Copy alpha channel
      for (let i = 0; i < width * height; i++) {
        result.data[i * 4 + 3] = imageData.data[i * 4 + 3];
      }
      
      // Filter parameters
      const twoSigmaColorSquare = 2 * this.sigmaColor * this.sigmaColor;
      const radius = this.radius;
      
      // For each pixel in the image
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const centerIdx = (y * width + x) * 4;
          const centerValue = imageData.data[centerIdx];
          
          let sum = 0;
          let totalWeight = 0;
          
          // Apply filter kernel
          for (let ky = -radius; ky <= radius; ky++) {
            const ny = y + ky;
            
            // Skip out-of-bounds pixels
            if (ny < 0 || ny >= height) continue;
            
            for (let kx = -radius; kx <= radius; kx++) {
              const nx = x + kx;
              
              // Skip out-of-bounds pixels
              if (nx < 0 || nx >= width) continue;
              
              const neighborIdx = (ny * width + nx) * 4;
              const neighborValue = imageData.data[neighborIdx];
              
              // Calculate spatial weight
              let spatialWeight;
              if (this.useOptimizedKernel) {
                const kernelIdx = (ky + radius) * this.diameter + (kx + radius);
                spatialWeight = this.spatialKernel[kernelIdx];
              } else {
                const distanceSquare = kx * kx + ky * ky;
                spatialWeight = Math.exp(-distanceSquare / (2 * this.sigmaSpace * this.sigmaSpace));
              }
              
              // Calculate range weight
              const diff = centerValue - neighborValue;
              const rangeWeight = Math.exp(-(diff * diff) / twoSigmaColorSquare);
              
              // Combine weights
              const weight = spatialWeight * rangeWeight;
              
              // Accumulate weighted values
              sum += neighborValue * weight;
              totalWeight += weight;
            }
          }
          
          // Set output value
          const finalValue = Math.round(sum / totalWeight);
          result.data[centerIdx] = result.data[centerIdx + 1] = result.data[centerIdx + 2] = finalValue;
        }
      }
      
      return result;
    }
    
    /**
     * Apply bilateral filter to an RGB image
     * @private
     */
    applyBilateralFilterRGB(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const result = new ImageData(width, height);
      
      // Copy alpha channel
      for (let i = 0; i < width * height; i++) {
        result.data[i * 4 + 3] = imageData.data[i * 4 + 3];
      }
      
      // Filter parameters
      const twoSigmaColorSquare = 2 * this.sigmaColor * this.sigmaColor;
      const radius = this.radius;
      
      // For each pixel in the image
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const centerIdx = (y * width + x) * 4;
          const centerR = imageData.data[centerIdx];
          const centerG = imageData.data[centerIdx + 1];
          const centerB = imageData.data[centerIdx + 2];
          
          let sumR = 0, sumG = 0, sumB = 0;
          let totalWeight = 0;
          
          // Apply filter kernel
          for (let ky = -radius; ky <= radius; ky++) {
            const ny = y + ky;
            
            // Skip out-of-bounds pixels
            if (ny < 0 || ny >= height) continue;
            
            for (let kx = -radius; kx <= radius; kx++) {
              const nx = x + kx;
              
              // Skip out-of-bounds pixels
              if (nx < 0 || nx >= width) continue;
              
              const neighborIdx = (ny * width + nx) * 4;
              const neighborR = imageData.data[neighborIdx];
              const neighborG = imageData.data[neighborIdx + 1];
              const neighborB = imageData.data[neighborIdx + 2];
              
              // Calculate spatial weight
              let spatialWeight;
              if (this.useOptimizedKernel) {
                const kernelIdx = (ky + radius) * this.diameter + (kx + radius);
                spatialWeight = this.spatialKernel[kernelIdx];
              } else {
                const distanceSquare = kx * kx + ky * ky;
                spatialWeight = Math.exp(-distanceSquare / (2 * this.sigmaSpace * this.sigmaSpace));
              }
              
              // Calculate range weight (Euclidean distance in RGB space)
              const diffR = centerR - neighborR;
              const diffG = centerG - neighborG;
              const diffB = centerB - neighborB;
              const colorDistance = diffR * diffR + diffG * diffG + diffB * diffB;
              const rangeWeight = Math.exp(-colorDistance / twoSigmaColorSquare);
              
              // Combine weights
              const weight = spatialWeight * rangeWeight;
              
              // Accumulate weighted values
              sumR += neighborR * weight;
              sumG += neighborG * weight;
              sumB += neighborB * weight;
              totalWeight += weight;
            }
          }
          
          // Set output values
          result.data[centerIdx] = Math.round(sumR / totalWeight);
          result.data[centerIdx + 1] = Math.round(sumG / totalWeight);
          result.data[centerIdx + 2] = Math.round(sumB / totalWeight);
        }
      }
      
      return result;
    }
    
    /**
     * Apply a simple 3x3 edge detection filter to highlight edges in debug mode
     * @param {ImageData} imageData - Input image
     * @returns {ImageData} Edge-highlighted image
     * @private
     */
    detectEdges(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const result = new ImageData(width, height);
      
      // Sobel operators
      const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
      const sobelY = [1, 2, 1, 0, 0, 0, -1, -2, -1];
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let gx = 0, gy = 0;
          
          // Apply Sobel operators
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const pixelIdx = ((y + ky) * width + (x + kx)) * 4;
              const kernelIdx = (ky + 1) * 3 + (kx + 1);
              
              // Use only red channel for simplicity
              gx += imageData.data[pixelIdx] * sobelX[kernelIdx];
              gy += imageData.data[pixelIdx] * sobelY[kernelIdx];
            }
          }
          
          // Calculate gradient magnitude
          const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy));
          
          // Set result
          const resultIdx = (y * width + x) * 4;
          result.data[resultIdx] = result.data[resultIdx + 1] = result.data[resultIdx + 2] = magnitude;
          result.data[resultIdx + 3] = 255;
        }
      }
      
      return result;
    }
    
    /**
     * Draw debug visualization on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    drawDebug(ctx) {
      if (!this.debug || !this.debugData.original) return;
      
      const width = this.debugData.original.width;
      const height = this.debugData.original.height;
      
      // Create temporary canvases for visualization
      const originalCanvas = document.createElement('canvas');
      originalCanvas.width = width;
      originalCanvas.height = height;
      const originalCtx = originalCanvas.getContext('2d');
      originalCtx.putImageData(this.debugData.original, 0, 0);
      
      const filteredCanvas = document.createElement('canvas');
      filteredCanvas.width = width;
      filteredCanvas.height = height;
      const filteredCtx = filteredCanvas.getContext('2d');
      filteredCtx.putImageData(this.debugData.filtered, 0, 0);
      
      // Detect edges in both original and filtered images for comparison
      const edgesOriginal = this.detectEdges(this.debugData.original);
      const edgesFiltered = this.detectEdges(this.debugData.filtered);
      
      const edgesOriginalCanvas = document.createElement('canvas');
      edgesOriginalCanvas.width = width;
      edgesOriginalCanvas.height = height;
      const edgesOriginalCtx = edgesOriginalCanvas.getContext('2d');
      edgesOriginalCtx.putImageData(edgesOriginal, 0, 0);
      
      const edgesFilteredCanvas = document.createElement('canvas');
      edgesFilteredCanvas.width = width;
      edgesFilteredCanvas.height = height;
      const edgesFilteredCtx = edgesFilteredCanvas.getContext('2d');
      edgesFilteredCtx.putImageData(edgesFiltered, 0, 0);
      
      // Draw comparison on debug canvas
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      
      // Draw original and filtered
      ctx.drawImage(originalCanvas, 0, 0);
      ctx.drawImage(filteredCanvas, width, 0);
      
      // Draw edges
      ctx.drawImage(edgesOriginalCanvas, 0, height);
      ctx.drawImage(edgesFilteredCanvas, width, height);
      
      // Add labels
      ctx.fillStyle = 'white';
      ctx.font = '14px Arial';
      ctx.fillText('Original', 10, 20);
      ctx.fillText('Bilateral Filtered', width + 10, 20);
      ctx.fillText('Edges (Original)', 10, height + 20);
      ctx.fillText('Edges (Filtered)', width + 10, height + 20);
      
      // Show performance info
      ctx.fillStyle = 'yellow';
      ctx.fillText(`Processing time: ${this.debugData.processingTime.toFixed(2)}ms`, 10, ctx.canvas.height - 10);
      ctx.fillText(`Diameter: ${this.diameter}, σColor: ${this.sigmaColor}, σSpace: ${this.sigmaSpace}`, 10, ctx.canvas.height - 30);
    }
    
    /**
     * Update filter options
     * @param {Object} options - New options
     */
    setOptions(options = {}) {
      let rebuildKernel = false;
      
      if (options.diameter !== undefined) {
        this.diameter = options.diameter;
        if (this.diameter % 2 === 0) {
          this.diameter += 1;
        }
        this.radius = Math.floor(this.diameter / 2);
        rebuildKernel = true;
      }
      
      if (options.sigmaColor !== undefined) {
        this.sigmaColor = options.sigmaColor;
      }
      
      if (options.sigmaSpace !== undefined) {
        this.sigmaSpace = options.sigmaSpace;
        rebuildKernel = true;
      }
      
      if (options.useOptimizedKernel !== undefined) {
        this.useOptimizedKernel = options.useOptimizedKernel;
        rebuildKernel = true;
      }
      
      if (options.greyOnly !== undefined) {
        this.greyOnly = options.greyOnly;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
      
      // Rebuild spatial kernel if needed
      if (rebuildKernel && this.useOptimizedKernel) {
        this.spatialKernel = this.precomputeSpatialKernel();
      }
    }
    
    /**
     * Get current settings
     * @returns {Object} Current filter settings
     */
    getSettings() {
      return {
        diameter: this.diameter,
        sigmaColor: this.sigmaColor,
        sigmaSpace: this.sigmaSpace,
        useOptimizedKernel: this.useOptimizedKernel,
        greyOnly: this.greyOnly,
        debug: this.debug,
        processingTime: this.debugData.processingTime || 0
      };
    }
  }
  
  /**
   * Helper function to easily create and apply a bilateral filter
   * @param {ImageData} imageData - Input image data
   * @param {Number} diameter - Filter diameter
   * @param {Number} sigmaColor - Filter sigma in color space
   * @param {Number} sigmaSpace - Filter sigma in coordinate space
   * @returns {ImageData} Filtered image data
   */
  function applyBilateralFilter(imageData, diameter = 9, sigmaColor = 75, sigmaSpace = 75) {
    const filter = new BilateralFilter({
      diameter: diameter,
      sigmaColor: sigmaColor,
      sigmaSpace: sigmaSpace
    });
    
    return filter.process(imageData);
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      BilateralFilter,
      applyBilateralFilter
    };
  } else {
    // Browser global
    window.BilateralFilter = BilateralFilter;
    window.applyBilateralFilter = applyBilateralFilter;
  }