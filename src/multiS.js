/**
 * Multi-Scale Processing for AR Marker Detection
 * 
 * This module provides multi-scale detection capabilities to improve marker detection
 * under challenging conditions:
 * - Detects markers at different scales to handle varying distances
 * - Reduces the impact of motion blur through downsampling
 * - Enhances detection of small or distant markers through upsampling
 * - Intelligently combines results from multiple scales
 */

class MultiScaleDetector {
    /**
     * Create a new multi-scale detector
     * @param {Object} options - Configuration options
     * @param {Object} options.detector - The base marker detector (e.g., AR.Detector)
     * @param {Array} options.scales - Array of scales to try (default: [1.0, 0.5, 2.0])
     * @param {number} options.confidenceThreshold - Confidence threshold for accepting detections (0-1)
     * @param {boolean} options.preferLargerMarkers - Whether to prefer markers with larger areas
     * @param {boolean} options.debug - Enable debug mode
     */
    constructor(options = {}) {
      this.detector = options.detector;
      this.scales = options.scales || [1.0, 0.5, 2.0];
      this.confidenceThreshold = options.confidenceThreshold || 0.6;
      this.preferLargerMarkers = options.preferLargerMarkers !== undefined ? 
        options.preferLargerMarkers : true;
      this.debug = options.debug || false;
      
      // Cache for scaled images
      this.imageCache = new Map();
      this.cacheTimeout = 500; // Cache timeout in ms
      this.lastCacheCleanup = Date.now();
      
      // Detection statistics
      this.stats = {
        detectionsByScale: {},
        totalDetections: 0,
        scaleHitRate: {},
        successfulScales: []
      };
      
      // Initialize statistics
      this.scales.forEach(scale => {
        this.stats.detectionsByScale[scale] = 0;
        this.stats.scaleHitRate[scale] = 0;
      });
    }
    
    /**
     * Detect markers using multi-scale processing
     * @param {ImageData} imageData - Original image data
     * @param {Object} options - Override default options
     * @returns {Array} Detected markers
     */
    detect(imageData, options = {}) {
      // Clean the cache periodically
      if (Date.now() - this.lastCacheCleanup > this.cacheTimeout) {
        this.imageCache.clear();
        this.lastCacheCleanup = Date.now();
      }
      
      const startTime = performance.now();
      const opts = Object.assign({}, {
        scales: this.scales,
        confidenceThreshold: this.confidenceThreshold,
        preferLargerMarkers: this.preferLargerMarkers
      }, options);
      
      // Reset successful scales for this detection
      this.stats.successfulScales = [];
      
      // Perform detection at different scales
      const detectionResults = [];
      
      for (const scale of opts.scales) {
        // Process image at current scale
        const scaledImageData = this._scaleImage(imageData, scale);
        
        // Detect markers
        const markers = this.detector.detect(scaledImageData);
        
        // If markers detected, scale coordinates back to original image
        if (markers.length > 0) {
          const scaledMarkers = this._scaleMarkers(markers, 1/scale);
          
          // Add scale and confidence information
          scaledMarkers.forEach(marker => {
            marker.detectedAtScale = scale;
            marker.confidence = marker.confidence || 1.0; // If no confidence provided by detector
            
            // Adjust confidence based on scale (slightly prefer original scale)
            if (scale !== 1.0) {
              marker.confidence *= 0.9;
            }
            
            // Add area as metric for preferring larger markers
            marker.area = this._calculateMarkerArea(marker.corners);
          });
          
          detectionResults.push(...scaledMarkers);
          
          // Update statistics
          this.stats.detectionsByScale[scale]++;
          this.stats.successfulScales.push(scale);
        }
      }
      
      // Combine and filter results
      const finalMarkers = this._combineResults(detectionResults, opts);
      
      // Update total detections statistics
      if (finalMarkers.length > 0) {
        this.stats.totalDetections++;
        
        // Update hit rate for successful scales
        this.stats.successfulScales.forEach(scale => {
          this.stats.scaleHitRate[scale] = 
            this.stats.detectionsByScale[scale] / this.stats.totalDetections;
        });
      }
      
      // Debug info
      if (this.debug) {
        console.log(`Multi-scale detection completed in ${(performance.now() - startTime).toFixed(2)}ms`);
        console.log(`Detected ${finalMarkers.length} markers from ${detectionResults.length} candidates`);
        console.log(`Successful scales: ${this.stats.successfulScales.join(', ')}`);
      }
      
      return finalMarkers;
    }
    
    /**
     * Scale image to a different size
     * @private
     */
    _scaleImage(imageData, scale) {
      // Return original image for scale = 1
      if (scale === 1.0) {
        return imageData;
      }
      
      // Check cache first
      const cacheKey = `${imageData.width}x${imageData.height}:${scale}`;
      if (this.imageCache.has(cacheKey)) {
        return this.imageCache.get(cacheKey);
      }
      
      // Calculate new dimensions
      const newWidth = Math.round(imageData.width * scale);
      const newHeight = Math.round(imageData.height * scale);
      
      // Create canvas for scaling
      const canvas = document.createElement('canvas');
      canvas.width = newWidth;
      canvas.height = newHeight;
      const ctx = canvas.getContext('2d');
      
      // Create a temporary canvas with the original image
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = imageData.width;
      tempCanvas.height = imageData.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.putImageData(imageData, 0, 0);
      
      // Apply appropriate scaling algorithm
      ctx.imageSmoothingEnabled = true;
      if (scale < 1.0) {
        // For downscaling, use high-quality interpolation
        ctx.imageSmoothingQuality = 'high';
      } else {
        // For upscaling, medium quality is usually sufficient
        ctx.imageSmoothingQuality = 'medium';
      }
      
      // Draw the scaled image
      ctx.drawImage(tempCanvas, 0, 0, newWidth, newHeight);
      
      // Get the scaled image data
      const scaledImageData = ctx.getImageData(0, 0, newWidth, newHeight);
      
      // Cache the result
      this.imageCache.set(cacheKey, scaledImageData);
      
      return scaledImageData;
    }
    
    /**
     * Scale marker coordinates back to original image
     * @private
     */
    _scaleMarkers(markers, scaleFactor) {
      return markers.map(marker => {
        // Clone the marker to avoid modifying the original
        const scaledMarker = Object.assign({}, marker);
        
        // Scale corner coordinates
        scaledMarker.corners = marker.corners.map(corner => ({
          x: corner.x * scaleFactor,
          y: corner.y * scaleFactor
        }));
        
        return scaledMarker;
      });
    }
    
    /**
     * Calculate area of a marker
     * @private
     */
    _calculateMarkerArea(corners) {
      // Use shoelace formula to calculate polygon area
      let area = 0;
      const n = corners.length;
      
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        area += corners[i].x * corners[j].y;
        area -= corners[j].x * corners[i].y;
      }
      
      return Math.abs(area) / 2;
    }
    
    /**
     * Combine results from multiple scales
     * @private
     */
    _combineResults(detectionResults, options) {
      if (detectionResults.length === 0) {
        return [];
      }
      
      // Group detections by marker ID
      const markerGroups = new Map();
      
      detectionResults.forEach(marker => {
        if (!markerGroups.has(marker.id)) {
          markerGroups.set(marker.id, []);
        }
        markerGroups.get(marker.id).push(marker);
      });
      
      // For each marker ID, select the best detection
      const finalMarkers = [];
      
      markerGroups.forEach((markers, id) => {
        // Filter by confidence threshold
        const validMarkers = markers.filter(
          m => m.confidence >= options.confidenceThreshold
        );
        
        if (validMarkers.length === 0) {
          return;
        }
        
        // Pick best marker based on configured preference
        let bestMarker;
        
        if (options.preferLargerMarkers) {
          // Prefer markers with larger area (often more accurate)
          bestMarker = validMarkers.reduce((best, current) => 
            current.area > best.area ? current : best
          );
        } else {
          // Prefer markers with higher confidence
          bestMarker = validMarkers.reduce((best, current) => 
            current.confidence > best.confidence ? current : best
          );
        }
        
        finalMarkers.push(bestMarker);
      });
      
      return finalMarkers;
    }
    
    /**
     * Get detection statistics
     * @returns {Object} Current detection statistics
     */
    getStatistics() {
      return {
        totalDetections: this.stats.totalDetections,
        detectionsByScale: this.stats.detectionsByScale,
        hitRateByScale: this.stats.scaleHitRate
      };
    }
    
    /**
     * Reset detection statistics
     */
    resetStatistics() {
      this.stats.totalDetections = 0;
      this.scales.forEach(scale => {
        this.stats.detectionsByScale[scale] = 0;
        this.stats.scaleHitRate[scale] = 0;
      });
    }
    
    /**
     * Update detector options
     * @param {Object} options - New options
     */
    setOptions(options = {}) {
      if (options.scales) {
        this.scales = options.scales;
        // Reset statistics for scales
        this.scales.forEach(scale => {
          if (!this.stats.detectionsByScale[scale]) {
            this.stats.detectionsByScale[scale] = 0;
            this.stats.scaleHitRate[scale] = 0;
          }
        });
      }
      
      if (options.confidenceThreshold !== undefined) {
        this.confidenceThreshold = options.confidenceThreshold;
      }
      
      if (options.preferLargerMarkers !== undefined) {
        this.preferLargerMarkers = options.preferLargerMarkers;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
    }
    
    /**
     * Draw debug visualization on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     * @param {Array} markers - Detected markers
     */
    drawDebug(ctx, markers) {
      if (!this.debug) return;
      
      // Draw scale information for each marker
      markers.forEach(marker => {
        if (!marker.detectedAtScale) return;
        
        // Get center point of marker
        const centerX = marker.corners.reduce((sum, c) => sum + c.x, 0) / marker.corners.length;
        const centerY = marker.corners.reduce((sum, c) => sum + c.y, 0) / marker.corners.length;
        
        // Draw scale information
        ctx.fillStyle = 'lime';
        ctx.font = 'bold 10px Arial';
        ctx.fillText(
          `Scale: ${marker.detectedAtScale.toFixed(1)}x`, 
          centerX, 
          centerY
        );
        
        // Draw confidence
        ctx.fillText(
          `Conf: ${(marker.confidence * 100).toFixed(0)}%`, 
          centerX, 
          centerY + 12
        );
        
        // Draw area
        ctx.fillText(
          `Area: ${Math.round(marker.area)}`, 
          centerX, 
          centerY + 24
        );
      });
      
      // Draw statistics
      ctx.fillStyle = 'white';
      ctx.font = '12px monospace';
      ctx.fillText(`Total detections: ${this.stats.totalDetections}`, 10, 20);
      
      let yPos = 40;
      for (const scale of this.scales) {
        const hitRate = this.stats.scaleHitRate[scale] || 0;
        ctx.fillStyle = hitRate > 0.5 ? 'lime' : hitRate > 0.2 ? 'yellow' : 'white';
        ctx.fillText(
          `Scale ${scale.toFixed(1)}x: ${(hitRate * 100).toFixed(1)}% (${this.stats.detectionsByScale[scale] || 0})`, 
          10, 
          yPos
        );
        yPos += 20;
      }
    }
  }
  
  /**
   * Create a multiscale detector wrapper around an existing detector
   * @param {Object} detector - Base detector (e.g., AR.Detector)
   * @param {Array} scales - Scales to use for detection
   * @returns {function} Wrapped detection function
   */
  function createMultiScaleDetector(detector, scales = [1.0, 0.5, 1.5]) {
    const multiScaleDetector = new MultiScaleDetector({
      detector: detector,
      scales: scales
    });
    
    return function(imageData) {
      return multiScaleDetector.detect(imageData);
    };
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      MultiScaleDetector,
      createMultiScaleDetector
    };
  } else {
    // Browser global
    window.MultiScaleDetector = MultiScaleDetector;
    window.createMultiScaleDetector = createMultiScaleDetector;
  }