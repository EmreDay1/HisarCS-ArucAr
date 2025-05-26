/**
 * Optical Flow Tracking for AR Applications
 * 
 * This library provides optical flow tracking to maintain marker detection
 * during rapid camera movements when standard detection might fail due to
 * motion blur or marker distortion.
 * 
 * It implements a simplified version of the Lucas-Kanade method for
 * tracking corner points between frames.
 */

class OpticalFlowTracker {
    /**
     * Create a new optical flow tracker
     * @param {Object} options - Configuration options
     * @param {number} options.winSize - Window size for tracking (default: 15)
     * @param {number} options.maxLevel - Pyramid levels for multiscale tracking (default: 2)
     * @param {number} options.maxError - Maximum error threshold for valid tracking (default: 10000)
     * @param {number} options.epsilon - Convergence criteria for iterative tracking (default: 0.01)
     * @param {number} options.maxIter - Maximum iterations for L-K tracking (default: 10)
     * @param {number} options.maxMotion - Maximum allowed motion between frames (default: 50)
     * @param {boolean} options.debug - Enable debug mode (default: false)
     */
    constructor(options = {}) {
      this.winSize = options.winSize || 15;
      this.maxLevel = options.maxLevel || 2;
      this.maxError = options.maxError || 10000;
      this.epsilon = options.epsilon || 0.01;
      this.maxIter = options.maxIter || 10;
      this.maxMotion = options.maxMotion || 50;
      this.debug = options.debug || false;
      
      // State for tracking
      this.lastCorners = null;         // Last detected marker corners
      this.lastDescriptors = null;     // Descriptors for last corners (for reidentification)
      this.lastGray = null;            // Last grayscale image
      this.trackingActive = false;     // Is tracking currently active?
      this.trackedFrames = 0;          // Number of consecutive tracked frames
      this.maxTrackedFrames = 30;      // Maximum number of frames to track without detection
      this.trackingQuality = 1.0;      // Quality metric (1.0 = best)
      
      // Motion estimation
      this.motionVector = { x: 0, y: 0 };
      this.motionHistory = [];
      this.motionHistorySize = 5;
      
      // Debug data
      this.debugData = {
        trackingPoints: [],
        trackingErrors: [],
        searchWindows: []
      };
    }
    
    /**
     * Process a new frame to either use direct detection or optical flow tracking
     * @param {ImageData} imageData - Current frame image data
     * @param {Array} detectedMarkers - Markers detected in the current frame
     * @returns {Array} Markers (either detected or tracked)
     */
    track(imageData, detectedMarkers) {
      // Convert current frame to grayscale for optical flow
      const currentGray = this._convertToGrayscale(imageData);
      
      // If markers were detected in this frame, use them directly
      if (detectedMarkers && detectedMarkers.length > 0) {
        // Store for future tracking
        this.lastCorners = this._cloneCorners(detectedMarkers[0].corners);
        this.lastGray = currentGray;
        this.trackingActive = true;
        this.trackedFrames = 0;
        this.trackingQuality = 1.0;
        
        // Update motion estimation
        if (this.debug) {
          this._updateMotionVector(this.lastCorners, detectedMarkers[0].corners);
        }
        
        return detectedMarkers; // Return the actually detected markers
      } 
      // No markers detected, try optical flow tracking
      else if (this.trackingActive && this.lastCorners && this.lastGray && this.trackedFrames < this.maxTrackedFrames) {
        // Track corners using optical flow
        const { corners: newCorners, status, errors } = this._trackLK(
          this.lastGray, 
          currentGray, 
          this.lastCorners
        );
        
        // Debug data collection
        if (this.debug) {
          this.debugData.trackingPoints = newCorners;
          this.debugData.trackingErrors = errors;
        }
        
        // If tracking succeeded and corners form a valid quadrilateral
        if (this._trackingSucceeded(status, errors) && this._isValidQuadrilateral(newCorners)) {
          // Update motion estimation
          this._updateMotionVector(this.lastCorners, newCorners);
          
          // Update tracking state
          this.lastCorners = newCorners;
          this.lastGray = currentGray;
          this.trackedFrames++;
          
          // Decrease quality slightly with each tracked frame
          this.trackingQuality = Math.max(0.5, 1.0 - (this.trackedFrames / this.maxTrackedFrames));
          
          // Return synthetic marker using tracked corners
          return [{
            id: -1, // Special ID to indicate this is from tracking
            corners: newCorners,
            confidence: this.trackingQuality
          }];
        } else {
          // Tracking failed, reset
          if (this.debug) {
            console.log("Tracking failed: invalid quadrilateral or tracking error");
          }
          this.trackingActive = false;
          return [];
        }
      } else {
        // No previous tracking data or exceeded max tracked frames
        this.trackingActive = false;
        return [];
      }
    }
    
    /**
     * Lucas-Kanade optical flow implementation for corner tracking
     * @private
     */
    _trackLK(prevImg, nextImg, prevPts) {
      const trackedPts = [];
      const statusArr = [];
      const errorsArr = [];
      
      // For each corner point
      for (let i = 0; i < prevPts.length; i++) {
        const pt = prevPts[i];
        
        // Initial guess at same position
        let newPt = { x: pt.x, y: pt.y };
        
        // Track using Lucas-Kanade method
        const result = this._trackPoint(prevImg, nextImg, pt, newPt);
        
        trackedPts.push(result.point);
        statusArr.push(result.status);
        errorsArr.push(result.error);
      }
      
      return {
        corners: trackedPts,
        status: statusArr,
        errors: errorsArr
      };
    }
    
    /**
     * Track a single point using Lucas-Kanade optical flow
     * @private
     */
    _trackPoint(prevImg, nextImg, prevPt, initialGuess) {
      const width = prevImg.width;
      const height = prevImg.height;
      const halfWin = Math.floor(this.winSize / 2);
      
      // Make a copy of initial guess
      let pt = { x: initialGuess.x, y: initialGuess.y };
      
      // Check if the point is within trackable bounds
      if (prevPt.x < halfWin || prevPt.x >= width - halfWin || 
          prevPt.y < halfWin || prevPt.y >= height - halfWin ||
          pt.x < halfWin || pt.x >= width - halfWin || 
          pt.y < halfWin || pt.y >= height - halfWin) {
        return { 
          point: { x: prevPt.x, y: prevPt.y }, 
          status: 0, 
          error: Infinity 
        };
      }
      
      // The implementation below is a simplified version of Lucas-Kanade
      // In a production environment, you'd likely use a more optimized version
      
      // Use sum of squared differences (SSD) for patch matching
      let bestError = Infinity;
      let bestPoint = { x: pt.x, y: pt.y };
      
      // Define search range based on expected motion
      const searchRange = Math.min(20, this.maxMotion);
      
      // Store search window for debugging
      if (this.debug) {
        this.debugData.searchWindows.push({
          x: pt.x - searchRange,
          y: pt.y - searchRange,
          width: searchRange * 2,
          height: searchRange * 2
        });
      }
      
      // Extract template patch from previous frame
      const templatePatch = this._extractPatch(prevImg, prevPt, this.winSize);
      
      // Exhaustive search in window (simplified approach)
      for (let dy = -searchRange; dy <= searchRange; dy += 1) {
        for (let dx = -searchRange; dx <= searchRange; dx += 1) {
          const testPt = { 
            x: Math.round(initialGuess.x + dx), 
            y: Math.round(initialGuess.y + dy) 
          };
          
          // Skip if out of bounds
          if (testPt.x < halfWin || testPt.x >= width - halfWin || 
              testPt.y < halfWin || testPt.y >= height - halfWin) {
            continue;
          }
          
          // Extract candidate patch
          const candidatePatch = this._extractPatch(nextImg, testPt, this.winSize);
          
          // Calculate SSD error
          const error = this._calculateSSD(templatePatch, candidatePatch);
          
          if (error < bestError) {
            bestError = error;
            bestPoint = { x: testPt.x, y: testPt.y };
          }
        }
      }
      
      // Check if the tracking error is acceptable
      const status = bestError < this.maxError ? 1 : 0;
      
      return {
        point: bestPoint,
        status: status,
        error: bestError
      };
    }
    
    /**
     * Extract a patch from an image around a center point
     * @private
     */
    _extractPatch(img, center, size) {
      const halfSize = Math.floor(size / 2);
      const patch = new Float32Array(size * size);
      
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const imgX = Math.round(center.x + (x - halfSize));
          const imgY = Math.round(center.y + (y - halfSize));
          
          // Bounds checking
          if (imgX >= 0 && imgX < img.width && imgY >= 0 && imgY < img.height) {
            const imgIdx = (imgY * img.width + imgX);
            patch[y * size + x] = img.data[imgIdx];
          }
        }
      }
      
      return patch;
    }
    
    /**
     * Calculate Sum of Squared Differences between two patches
     * @private
     */
    _calculateSSD(patch1, patch2) {
      let sum = 0;
      
      for (let i = 0; i < patch1.length; i++) {
        const diff = patch1[i] - patch2[i];
        sum += diff * diff;
      }
      
      return sum;
    }
    
    /**
     * Check if optical flow tracking was successful based on status and errors
     * @private
     */
    _trackingSucceeded(status, errors) {
      // Check if all points were successfully tracked
      const allTracked = status.every(s => s === 1);
      
      // Check if average error is below threshold
      const avgError = errors.reduce((sum, err) => sum + err, 0) / errors.length;
      const lowError = avgError < this.maxError;
      
      return allTracked && lowError;
    }
    
    /**
     * Check if points form a valid quadrilateral (not too distorted)
     * @private
     */
    _isValidQuadrilateral(corners) {
      if (corners.length !== 4) {
        return false;
      }
      
      // Check minimum area
      const area = this._calculateQuadArea(corners);
      if (area < 100) { // Minimum area threshold
        return false;
      }
      
      // Check that it's roughly convex
      if (!this._isRoughlyConvex(corners)) {
        return false;
      }
      
      // Check aspect ratio is not too extreme
      const aspectRatio = this._calculateAspectRatio(corners);
      if (aspectRatio > 5 || aspectRatio < 0.2) { // Not too stretched
        return false;
      }
      
      return true;
    }
    
    /**
     * Calculate area of a quadrilateral
     * @private
     */
    _calculateQuadArea(quad) {
      // Split into two triangles and sum their areas
      const area1 = this._triangleArea(quad[0], quad[1], quad[2]);
      const area2 = this._triangleArea(quad[0], quad[2], quad[3]);
      return Math.abs(area1) + Math.abs(area2);
    }
    
    /**
     * Calculate area of a triangle using cross product
     * @private
     */
    _triangleArea(p1, p2, p3) {
      return 0.5 * ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y));
    }
    
    /**
     * Check if a quadrilateral is roughly convex
     * @private
     */
    _isRoughlyConvex(quad) {
      // Check sign of cross products for all consecutive vertices
      let allPositive = true;
      let allNegative = true;
      
      for (let i = 0; i < 4; i++) {
        const p1 = quad[i];
        const p2 = quad[(i + 1) % 4];
        const p3 = quad[(i + 2) % 4];
        
        const crossProduct = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x);
        
        if (crossProduct < 0) allPositive = false;
        if (crossProduct > 0) allNegative = false;
      }
      
      // If all cross products have the same sign, it's convex
      return allPositive || allNegative;
    }
    
    /**
     * Calculate aspect ratio of a quadrilateral
     * @private
     */
    _calculateAspectRatio(quad) {
      // Calculate average width and height
      const width1 = this._distance(quad[0], quad[1]);
      const width2 = this._distance(quad[3], quad[2]);
      const height1 = this._distance(quad[0], quad[3]);
      const height2 = this._distance(quad[1], quad[2]);
      
      const avgWidth = (width1 + width2) / 2;
      const avgHeight = (height1 + height2) / 2;
      
      return avgWidth > avgHeight ? avgWidth / avgHeight : avgHeight / avgWidth;
    }
    
    /**
     * Calculate Euclidean distance between two points
     * @private
     */
    _distance(p1, p2) {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Convert RGB image data to grayscale
     * @private
     */
    _convertToGrayscale(imageData) {
      const width = imageData.width;
      const height = imageData.height;
      const grayData = {
        width: width,
        height: height,
        data: new Uint8Array(width * height)
      };
      
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          // Standard luminance formula
          grayData.data[y * width + x] = Math.round(
            0.299 * imageData.data[i] + 
            0.587 * imageData.data[i+1] + 
            0.114 * imageData.data[i+2]
          );
        }
      }
      
      return grayData;
    }
    
    /**
     * Clone corners array to avoid reference issues
     * @private
     */
    _cloneCorners(corners) {
      return corners.map(c => ({ x: c.x, y: c.y }));
    }
    
    /**
     * Update motion vector based on point movement
     * @private
     */
    _updateMotionVector(prevCorners, newCorners) {
      if (!prevCorners || !newCorners || prevCorners.length !== newCorners.length) {
        return;
      }
      
      // Calculate average motion
      let sumX = 0, sumY = 0;
      for (let i = 0; i < prevCorners.length; i++) {
        sumX += newCorners[i].x - prevCorners[i].x;
        sumY += newCorners[i].y - prevCorners[i].y;
      }
      
      const avgMotion = {
        x: sumX / prevCorners.length,
        y: sumY / prevCorners.length
      };
      
      // Add to motion history
      this.motionHistory.push(avgMotion);
      
      // Keep history at fixed size
      if (this.motionHistory.length > this.motionHistorySize) {
        this.motionHistory.shift();
      }
      
      // Calculate smoothed motion vector
      sumX = 0;
      sumY = 0;
      for (const motion of this.motionHistory) {
        sumX += motion.x;
        sumY += motion.y;
      }
      
      this.motionVector = {
        x: sumX / this.motionHistory.length,
        y: sumY / this.motionHistory.length
      };
    }
    
    /**
     * Reset tracking state
     */
    reset() {
      this.lastCorners = null;
      this.lastGray = null;
      this.trackingActive = false;
      this.trackedFrames = 0;
      this.trackingQuality = 1.0;
      this.motionVector = { x: 0, y: 0 };
      this.motionHistory = [];
      
      if (this.debug) {
        this.debugData = {
          trackingPoints: [],
          trackingErrors: [],
          searchWindows: []
        };
      }
    }
    
    /**
     * Get current motion vector
     * @returns {Object} Motion vector {x, y}
     */
    getMotionVector() {
      return { x: this.motionVector.x, y: this.motionVector.y };
    }
    
    /**
     * Get tracking quality (1.0 = best, 0.0 = worst)
     * @returns {number} Tracking quality
     */
    getTrackingQuality() {
      return this.trackingQuality;
    }
    
    /**
     * Check if tracking is currently active
     * @returns {boolean} Tracking state
     */
    isTracking() {
      return this.trackingActive;
    }
    
    /**
     * Set new tracker options
     * @param {Object} options - New option values
     */
    setOptions(options = {}) {
      if (options.winSize !== undefined) {
        this.winSize = options.winSize;
      }
      
      if (options.maxLevel !== undefined) {
        this.maxLevel = options.maxLevel;
      }
      
      if (options.maxError !== undefined) {
        this.maxError = options.maxError;
      }
      
      if (options.epsilon !== undefined) {
        this.epsilon = options.epsilon;
      }
      
      if (options.maxIter !== undefined) {
        this.maxIter = options.maxIter;
      }
      
      if (options.maxMotion !== undefined) {
        this.maxMotion = options.maxMotion;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
      
      if (options.maxTrackedFrames !== undefined) {
        this.maxTrackedFrames = options.maxTrackedFrames;
      }
    }
    
    /**
     * Draw debug visualization on a canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    drawDebug(ctx) {
      if (!this.debug) return;
      
      // Draw tracked points
      ctx.fillStyle = 'rgba(0, 255, 0, 0.5)';
      for (const pt of this.debugData.trackingPoints) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
      
      // Draw search windows
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
      for (const win of this.debugData.searchWindows) {
        ctx.strokeRect(win.x, win.y, win.width, win.height);
      }
      
      // Draw motion vector
      if (this.trackingActive) {
        const centerX = ctx.canvas.width / 2;
        const centerY = ctx.canvas.height / 2;
        const scale = 5; // Scale up for visibility
        
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX + this.motionVector.x * scale, centerY + this.motionVector.y * scale);
        ctx.stroke();
        
        // Add motion vector text
        ctx.fillStyle = 'white';
        ctx.font = '10px Arial';
        ctx.fillText(
          `Motion: (${this.motionVector.x.toFixed(1)}, ${this.motionVector.y.toFixed(1)})`,
          centerX + 10, centerY + 10
        );
        
        // Add tracking quality indicator
        ctx.fillStyle = this.trackingQuality > 0.7 ? 'green' : 
                        this.trackingQuality > 0.4 ? 'orange' : 'red';
        ctx.fillText(
          `Quality: ${(this.trackingQuality * 100).toFixed(0)}%`,
          centerX + 10, centerY + 25
        );
      }
    }
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      OpticalFlowTracker
    };
  } else {
    // Browser global
    window.OpticalFlowTracker = OpticalFlowTracker;
  }