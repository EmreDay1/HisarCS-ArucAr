/**
 * Edge-Enhanced Marker Detection for AR Applications
 * 
 * This module provides edge detection and corner enhancement techniques to improve 
 * marker detection in challenging conditions, especially at extreme viewing angles.
 * 
 * Key features:
 * - Canny edge detection to highlight marker boundaries
 * - Harris corner detection to enhance marker corners
 * - Gradient analysis for better feature extraction
 * - Adaptive thresholding to improve marker visibility
 */

class EdgeEnhancedDetector {
    /**
     * Create a new edge-enhanced detector
     * @param {Object} options - Configuration options
     * @param {Object} options.detector - Base detector to enhance (e.g., AR.Detector)
     * @param {Number} options.edgeThreshold - Threshold for edge detection (0-255)
     * @param {Number} options.cannyThreshold1 - Low threshold for Canny detector
     * @param {Number} options.cannyThreshold2 - High threshold for Canny detector
     * @param {Number} options.harrisK - Harris corner detector sensitivity parameter
     * @param {Boolean} options.debug - Enable debug mode
     */
    constructor(options = {}) {
      this.baseDetector = options.detector;
      this.edgeThreshold = options.edgeThreshold || 30;
      this.cannyThreshold1 = options.cannyThreshold1 || 50;
      this.cannyThreshold2 = options.cannyThreshold2 || 150;
      this.harrisK = options.harrisK || 0.04;
      this.blurRadius = options.blurRadius || 3;
      this.sobelKernelSize = options.sobelKernelSize || 3;
      this.debug = options.debug || false;
      
      // Cache for intermediate processing steps (for debug visualization)
      this.debugData = {
        grayscale: null,
        blurred: null,
        edges: null,
        corners: null,
        enhanced: null
      };
    }
    
    /**
     * Detect markers using edge enhancement
     * @param {ImageData} imageData - Original image data
     * @returns {Array} Detected markers
     */
    detect(imageData) {
      // Convert to grayscale for processing
      const grayscale = this.toGrayscale(imageData);
      if (this.debug) this.debugData.grayscale = grayscale;
      
      // Apply Gaussian blur to reduce noise
      const blurred = this.applyGaussianBlur(grayscale, this.blurRadius);
      if (this.debug) this.debugData.blurred = blurred;
      
      // Apply edge detection (Canny algorithm)
      const edges = this.applyCannyEdgeDetection(blurred);
      if (this.debug) this.debugData.edges = edges;
      
      // Enhance corners (Harris corner detector)
      const corners = this.detectHarrisCorners(blurred);
      if (this.debug) this.debugData.corners = corners;
      
      // Combine edges and corners to enhance marker features
      const enhanced = this.combineEdgesAndCorners(grayscale, edges, corners);
      if (this.debug) this.debugData.enhanced = enhanced;
      
      // Run the base detector on the enhanced image
      const markers = this.baseDetector.detect(this.imageDataFromGrayscale(enhanced));
      
      return markers;
    }
    
    /**
     * Convert RGB image to grayscale
     * @private
     */
    toGrayscale(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const grayscale = new Uint8ClampedArray(width * height);
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          // Use standard luminance formula
          grayscale[y * width + x] = Math.round(
            0.299 * imageData.data[idx] + 
            0.587 * imageData.data[idx + 1] + 
            0.114 * imageData.data[idx + 2]
          );
        }
      }
      
      return {
        data: grayscale,
        width: width,
        height: height
      };
    }
    
    /**
     * Apply Gaussian blur to reduce noise
     * @private
     */
    applyGaussianBlur(grayscale, radius) {
      const width = grayscale.width;
      const height = grayscale.height;
      const result = new Uint8ClampedArray(width * height);
      const sigma = radius / 3;
      const twoSigmaSquare = 2 * sigma * sigma;
      const piTwoSigmaSquare = Math.PI * twoSigmaSquare;
      
      // Create Gaussian kernel
      const kernelSize = radius * 2 + 1;
      const kernel = new Float32Array(kernelSize);
      let sum = 0;
      
      for (let i = 0; i < kernelSize; i++) {
        const x = i - radius;
        kernel[i] = Math.exp(-(x * x) / twoSigmaSquare) / piTwoSigmaSquare;
        sum += kernel[i];
      }
      
      // Normalize kernel
      for (let i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
      }
      
      // Apply horizontal blur
      const temp = new Uint8ClampedArray(width * height);
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let val = 0;
          let weightSum = 0;
          
          for (let i = -radius; i <= radius; i++) {
            const xi = x + i;
            if (xi >= 0 && xi < width) {
              const weight = kernel[i + radius];
              val += grayscale.data[y * width + xi] * weight;
              weightSum += weight;
            }
          }
          
          temp[y * width + x] = Math.round(val / weightSum);
        }
      }
      
      // Apply vertical blur
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let val = 0;
          let weightSum = 0;
          
          for (let i = -radius; i <= radius; i++) {
            const yi = y + i;
            if (yi >= 0 && yi < height) {
              const weight = kernel[i + radius];
              val += temp[yi * width + x] * weight;
              weightSum += weight;
            }
          }
          
          result[y * width + x] = Math.round(val / weightSum);
        }
      }
      
      return {
        data: result,
        width: width,
        height: height
      };
    }
    
    /**
     * Apply Canny edge detection
     * @private
     */
    applyCannyEdgeDetection(grayscale) {
      const width = grayscale.width;
      const height = grayscale.height;
      
      // 1. Calculate gradients (Sobel)
      const [gradientMagnitude, gradientDirection] = this.calculateGradients(grayscale);
      
      // 2. Non-maximum suppression
      const suppressed = this.nonMaximumSuppression(gradientMagnitude, gradientDirection, width, height);
      
      // 3. Hysteresis thresholding
      const edges = this.hysteresisThresholding(
        suppressed, 
        this.cannyThreshold1, 
        this.cannyThreshold2, 
        width, 
        height
      );
      
      return {
        data: edges,
        width: width,
        height: height
      };
    }
    
    /**
     * Calculate image gradients using Sobel operator
     * @private
     */
    calculateGradients(grayscale) {
      const width = grayscale.width;
      const height = grayscale.height;
      const data = grayscale.data;
      
      // Sobel kernels
      const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
      const sobelY = [1, 2, 1, 0, 0, 0, -1, -2, -1];
      
      const gradientMagnitude = new Uint8ClampedArray(width * height);
      const gradientDirection = new Float32Array(width * height);
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let gx = 0;
          let gy = 0;
          
          // Apply convolution
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = (y + ky) * width + (x + kx);
              const kernelIdx = (ky + 1) * 3 + (kx + 1);
              
              gx += data[idx] * sobelX[kernelIdx];
              gy += data[idx] * sobelY[kernelIdx];
            }
          }
          
          // Calculate magnitude and direction
          const magnitude = Math.sqrt(gx * gx + gy * gy);
          const direction = Math.atan2(gy, gx);
          
          const idx = y * width + x;
          gradientMagnitude[idx] = Math.min(255, magnitude);
          gradientDirection[idx] = direction;
        }
      }
      
      return [gradientMagnitude, gradientDirection];
    }
    
    /**
     * Apply non-maximum suppression to thin edges
     * @private
     */
    nonMaximumSuppression(gradientMagnitude, gradientDirection, width, height) {
      const result = new Uint8ClampedArray(width * height);
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const idx = y * width + x;
          const direction = gradientDirection[idx];
          const magnitude = gradientMagnitude[idx];
          
          // Normalize angle to 0-180 degrees (π radians)
          let angle = direction * 180 / Math.PI;
          if (angle < 0) angle += 180;
          
          // Get neighboring pixels along gradient direction
          let neighbor1, neighbor2;
          
          if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
            // Horizontal edge (0 degrees)
            neighbor1 = gradientMagnitude[idx - 1];  // West
            neighbor2 = gradientMagnitude[idx + 1];  // East
          } else if (angle >= 22.5 && angle < 67.5) {
            // Diagonal edge (45 degrees)
            neighbor1 = gradientMagnitude[idx - width - 1];  // North-West
            neighbor2 = gradientMagnitude[idx + width + 1];  // South-East
          } else if (angle >= 67.5 && angle < 112.5) {
            // Vertical edge (90 degrees)
            neighbor1 = gradientMagnitude[idx - width];  // North
            neighbor2 = gradientMagnitude[idx + width];  // South
          } else if (angle >= 112.5 && angle < 157.5) {
            // Diagonal edge (135 degrees)
            neighbor1 = gradientMagnitude[idx - width + 1];  // North-East
            neighbor2 = gradientMagnitude[idx + width - 1];  // South-West
          }
          
          // Keep only local maxima
          if (magnitude >= neighbor1 && magnitude >= neighbor2) {
            result[idx] = magnitude;
          } else {
            result[idx] = 0;
          }
        }
      }
      
      return result;
    }
    
    /**
     * Apply hysteresis thresholding to connect edges
     * @private
     */
    hysteresisThresholding(suppressed, lowThreshold, highThreshold, width, height) {
      const result = new Uint8ClampedArray(width * height);
      const visited = new Uint8ClampedArray(width * height);
      const stack = [];
      
      // First pass: identify strong edges
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          if (suppressed[idx] >= highThreshold) {
            // Strong edge
            result[idx] = 255;
            visited[idx] = 1;
            stack.push({x, y});
          } else if (suppressed[idx] < lowThreshold) {
            // Non-edge
            visited[idx] = 1;
          }
        }
      }
      
      // Second pass: trace weak edges connected to strong edges
      const dx = [-1, 0, 1, -1, 1, -1, 0, 1];
      const dy = [-1, -1, -1, 0, 0, 1, 1, 1];
      
      while (stack.length > 0) {
        const {x, y} = stack.pop();
        
        // Check 8 neighbors
        for (let i = 0; i < 8; i++) {
          const nx = x + dx[i];
          const ny = y + dy[i];
          
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nidx = ny * width + nx;
            
            // If this is a weak edge and not yet processed
            if (!visited[nidx] && suppressed[nidx] >= lowThreshold) {
              result[nidx] = 255;
              visited[nidx] = 1;
              stack.push({x: nx, y: ny});
            }
          }
        }
      }
      
      return result;
    }
    
    /**
     * Detect corners using Harris corner detector
     * @private
     */
    detectHarrisCorners(grayscale) {
      const width = grayscale.width;
      const height = grayscale.height;
      const data = grayscale.data;
      const k = this.harrisK;
      const blockSize = 3;
      const derivativeKernelSize = 3;
      
      // Calculate gradients
      const [Ix, Iy] = this.calculateSobelDerivatives(grayscale);
      
      // Calculate products of derivatives for each pixel
      const Ixx = new Float32Array(width * height);
      const Iyy = new Float32Array(width * height);
      const Ixy = new Float32Array(width * height);
      
      for (let i = 0; i < width * height; i++) {
        Ixx[i] = Ix[i] * Ix[i];
        Iyy[i] = Iy[i] * Iy[i];
        Ixy[i] = Ix[i] * Iy[i];
      }
      
      // Apply Gaussian blur to derivative products
      const blurredIxx = this.boxBlur(Ixx, width, height, blockSize);
      const blurredIyy = this.boxBlur(Iyy, width, height, blockSize);
      const blurredIxy = this.boxBlur(Ixy, width, height, blockSize);
      
      // Calculate corner response for each pixel
      const cornerResponse = new Float32Array(width * height);
      let maxResponse = 0;
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = y * width + x;
          const a = blurredIxx[idx];
          const b = blurredIxy[idx];
          const c = blurredIyy[idx];
          
          // Harris response: det(M) - k * trace(M)^2
          const response = (a * c - b * b) - (k * (a + c) * (a + c));
          cornerResponse[idx] = response;
          
          maxResponse = Math.max(maxResponse, response);
        }
      }
      
      // Threshold and perform non-maximum suppression
      const corners = new Uint8ClampedArray(width * height);
      const cornerThreshold = 0.01 * maxResponse;
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          const idx = y * width + x;
          const response = cornerResponse[idx];
          
          // Skip weak corners
          if (response < cornerThreshold) continue;
          
          // Non-maximum suppression in a 3x3 neighborhood
          let isMax = true;
          for (let ny = -1; ny <= 1 && isMax; ny++) {
            for (let nx = -1; nx <= 1 && isMax; nx++) {
              if (nx === 0 && ny === 0) continue;
              
              const nidx = (y + ny) * width + (x + nx);
              if (cornerResponse[nidx] > response) {
                isMax = false;
              }
            }
          }
          
          if (isMax) {
            corners[idx] = 255;
          }
        }
      }
      
      return {
        data: corners,
        width: width,
        height: height
      };
    }
    
    /**
     * Calculate Sobel derivatives
     * @private
     */
    calculateSobelDerivatives(grayscale) {
      const width = grayscale.width;
      const height = grayscale.height;
      const data = grayscale.data;
      
      const Ix = new Float32Array(width * height);
      const Iy = new Float32Array(width * height);
      
      // Sobel kernels
      const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
      const sobelY = [1, 2, 1, 0, 0, 0, -1, -2, -1];
      
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          let gx = 0;
          let gy = 0;
          
          // Apply convolution
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = (y + ky) * width + (x + kx);
              const kernelIdx = (ky + 1) * 3 + (kx + 1);
              
              gx += data[idx] * sobelX[kernelIdx];
              gy += data[idx] * sobelY[kernelIdx];
            }
          }
          
          const idx = y * width + x;
          Ix[idx] = gx;
          Iy[idx] = gy;
        }
      }
      
      return [Ix, Iy];
    }
    
    /**
     * Apply box blur (simplified Gaussian)
     * @private
     */
    boxBlur(data, width, height, radius) {
      const result = new Float32Array(width * height);
      const temp = new Float32Array(width * height);
      
      // Horizontal blur
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let count = 0;
          
          for (let i = -radius; i <= radius; i++) {
            const xi = x + i;
            if (xi >= 0 && xi < width) {
              sum += data[y * width + xi];
              count++;
            }
          }
          
          temp[y * width + x] = sum / count;
        }
      }
      
      // Vertical blur
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = 0;
          let count = 0;
          
          for (let i = -radius; i <= radius; i++) {
            const yi = y + i;
            if (yi >= 0 && yi < height) {
              sum += temp[yi * width + x];
              count++;
            }
          }
          
          result[y * width + x] = sum / count;
        }
      }
      
      return result;
    }
    
    /**
     * Combine edges and corners to enhance marker features
     * @private
     */
    combineEdgesAndCorners(grayscale, edges, corners) {
      const width = grayscale.width;
      const height = grayscale.height;
      const result = new Uint8ClampedArray(width * height);
      
      // Dilate corners slightly to increase their influence
      const dilatedCorners = this.dilate(corners.data, width, height, 1);
      
      for (let i = 0; i < width * height; i++) {
        // Start with the original grayscale image
        const origValue = grayscale.data[i];
        
        // Edge information (binary)
        const edgeValue = edges.data[i];
        
        // Corner information (binary)
        const cornerValue = dilatedCorners[i];
        
        // Combine information:
        // 1. If it's an edge, increase contrast
        // 2. If it's a corner, enhance it further
        // 3. Otherwise, apply mild contrast enhancement
        
        if (cornerValue > 0) {
          // Strong enhancement for corners
          result[i] = 0; // Make corners black (marker corners are usually dark)
        } else if (edgeValue > 0) {
          // Medium enhancement for edges
          result[i] = 0; // Make edges black too
        } else {
          // Apply contrast enhancement to original
          const enhancedValue = this.enhanceContrast(origValue);
          result[i] = enhancedValue;
        }
      }
      
      return result;
    }
    
    /**
     * Apply contrast enhancement to a pixel value
     * @private
     */
    enhanceContrast(value) {
      // Simple contrast stretch using a non-linear function
      return value < 128 ? 
        Math.max(0, value - this.edgeThreshold) : 
        Math.min(255, value + this.edgeThreshold);
    }
    
    /**
     * Dilate a binary image
     * @private
     */
    dilate(data, width, height, radius) {
      const result = new Uint8ClampedArray(width * height);
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let maxVal = 0;
          
          // Check neighborhood
          for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
              const nx = x + dx;
              const ny = y + dy;
              
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                maxVal = Math.max(maxVal, data[ny * width + nx]);
              }
            }
          }
          
          result[y * width + x] = maxVal;
        }
      }
      
      return result;
    }
    
    /**
     * Convert grayscale data back to ImageData
     * @private
     */
    imageDataFromGrayscale(grayscale) {
      const width = grayscale.width;
      const height = grayscale.height;
      const data = new Uint8ClampedArray(width * height * 4);
      
      for (let i = 0; i < width * height; i++) {
        const val = grayscale.data[i];
        data[i * 4] = val;     // R
        data[i * 4 + 1] = val; // G
        data[i * 4 + 2] = val; // B
        data[i * 4 + 3] = 255; // A
      }
      
      return new ImageData(data, width, height);
    }
    
    /**
     * Draw debug visualization on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    drawDebug(ctx) {
      if (!this.debug || !this.debugData.grayscale) return;
      
      const width = this.debugData.grayscale.width;
      const height = this.debugData.grayscale.height;
      
      // Draw original grayscale
      this.drawGrayscaleToCanvas(ctx, this.debugData.grayscale, 0, 0);
      
      // Draw edge detection result
      if (this.debugData.edges) {
        this.drawGrayscaleToCanvas(ctx, this.debugData.edges, width, 0);
      }
      
      // Draw corner detection result
      if (this.debugData.corners) {
        this.drawGrayscaleToCanvas(ctx, this.debugData.corners, 0, height);
      }
      
      // Draw combined enhanced image
      if (this.debugData.enhanced) {
        this.drawGrayscaleToCanvas(ctx, this.debugData.enhanced, width, height);
      }
      
      // Add labels
      ctx.fillStyle = 'white';
      ctx.font = '12px Arial';
      ctx.fillText('Original Grayscale', 10, 15);
      ctx.fillText('Edge Detection', width + 10, 15);
      ctx.fillText('Corner Detection', 10, height + 15);
      ctx.fillText('Enhanced Image', width + 10, height + 15);
    }
    
    /**
     * Helper to draw grayscale data to canvas
     * @private
     */
    drawGrayscaleToCanvas(ctx, grayscale, offsetX, offsetY) {
      const width = grayscale.width;
      const height = grayscale.height;
      const imageData = ctx.createImageData(width, height);
      
      for (let i = 0; i < width * height; i++) {
        const val = grayscale.data[i];
        imageData.data[i * 4] = val;     // R
        imageData.data[i * 4 + 1] = val; // G
        imageData.data[i * 4 + 2] = val; // B
        imageData.data[i * 4 + 3] = 255; // A
      }
      
      // Create a temporary canvas to draw the imageData
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.putImageData(imageData, 0, 0);
      
      // Draw to the main canvas
      ctx.drawImage(tempCanvas, offsetX, offsetY);
    }
    
    /**
     * Get processing statistics and information
     * @returns {Object} Processing statistics
     */
    getStatistics() {
      // This could be expanded to include more detailed stats
      return {
        enabled: this.debug,
        edgeThreshold: this.edgeThreshold,
        cannyThresholds: [this.cannyThreshold1, this.cannyThreshold2],
        harrisK: this.harrisK
      };
    }
    
    /**
     * Update detector options
     * @param {Object} options - New options
     */
    setOptions(options = {}) {
      if (options.edgeThreshold !== undefined) {
        this.edgeThreshold = options.edgeThreshold;
      }
      
      if (options.cannyThreshold1 !== undefined) {
        this.cannyThreshold1 = options.cannyThreshold1;
      }
      
      if (options.cannyThreshold2 !== undefined) {
        this.cannyThreshold2 = options.cannyThreshold2;
      }
      
      if (options.harrisK !== undefined) {
        this.harrisK = options.harrisK;
      }
      
      if (options.blurRadius !== undefined) {
        this.blurRadius = options.blurRadius;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
    }
  }
  
  /**
   * Create a wrapped detector with edge enhancement
   * @param {Object} baseDetector - Base detector to enhance
   * @param {Object} options - Configuration options
   * @returns {Object} Edge-enhanced detector
   */
  function createEdgeEnhancedDetector(baseDetector, options = {}) {
    return new EdgeEnhancedDetector({
      detector: baseDetector,
      ...options
    });
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      EdgeEnhancedDetector,
      createEdgeEnhancedDetector
    };
  } else {
    // Browser global
    window.EdgeEnhancedDetector = EdgeEnhancedDetector;
    window.createEdgeEnhancedDetector = createEdgeEnhancedDetector;
  }