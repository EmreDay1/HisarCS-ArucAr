/**
 * Multi-Tag Pose Tracking System for AR Applications
 * 
 * This module provides enhanced pose tracking by:
 * 1. Tracking multiple markers simultaneously
 * 2. Combining information from multiple pose estimations
 * 3. Maintaining spatial relationships between markers
 * 4. Providing pose prediction during partial occlusion
 * 5. Utilizing both POSIT implementations for optimal results
 */

class MultiTagPoseTracker {
    /**
     * Create a new multi-tag pose tracking system
     * @param {Object} options - Configuration options
     * @param {Number} options.markerSize - Physical size of markers in real-world units (e.g., mm)
     * @param {Number} options.focalLength - Camera focal length in pixels
     * @param {Boolean} options.useKalmanFilter - Whether to use Kalman filtering for pose smoothing
     * @param {Boolean} options.useOpticalFlow - Whether to use optical flow for tracking in between detections
     * @param {Boolean} options.useAdaptiveAlgorithm - Whether to dynamically select between POSIT algorithms
     * @param {Array} options.knownMarkerPositions - Known physical positions of markers in real-world space
     * @param {Boolean} options.debug - Enable debug mode
     */
    constructor(options = {}) {
      // Core tracking parameters
      this.markerSize = options.markerSize || 50; // Default 50mm marker size
      this.focalLength = options.focalLength || 700; // Default focal length (typical for webcams)
      this.useKalmanFilter = options.useKalmanFilter !== undefined ? options.useKalmanFilter : true;
      this.useOpticalFlow = options.useOpticalFlow !== undefined ? options.useOpticalFlow : true;
      this.useAdaptiveAlgorithm = options.useAdaptiveAlgorithm !== undefined ? options.useAdaptiveAlgorithm : true;
      this.debug = options.debug || false;
      
      // Known spatial relationships between markers (if provided)
      this.knownMarkerPositions = options.knownMarkerPositions || null;
      
      // Initialize pose estimators (both implementations for adaptive selection)
      this.posit1 = new POS.Posit(this.markerSize, this.focalLength);
      this.posit2 = new POS.Posit(this.markerSize, this.focalLength);
      
      // Initialize Kalman filter for pose tracking if enabled
      if (this.useKalmanFilter) {
        // Create 6 state variables (3 for position, 3 for rotation)
        this.kalmanFilters = {};
      }
      
      // Initialize optical flow tracker if enabled
      if (this.useOpticalFlow) {
        this.opticalFlowTracker = new OpticalFlowTracker({
          maxTrackedFrames: 30,
          winSize: 15,
          maxLevel: 2,
          maxError: 10000
        });
      }
      
      // State variables for tracking
      this.markers = new Map(); // Map of marker IDs to their tracking info
      this.lastImageData = null; // Last processed image for optical flow
      this.lastProcessTime = performance.now();
      this.frameCount = 0;
      
      // Spatial reference model for multiple markers
      this.spatialModel = null;
      
      // Debug data
      this.debugData = {
        algorithmUsed: {},
        positionError: {},
        rotationError: {},
        trackingTime: []
      };
    }
    
    /**
     * Process detected markers and estimate poses
     * @param {ImageData} imageData - Current camera frame
     * @param {Array} detectedMarkers - Markers detected in the current frame
     * @returns {Map} Map of marker IDs to their estimated poses
     */
    update(imageData, detectedMarkers) {
      const startTime = performance.now();
      this.frameCount++;
      
      // Initialize collections for this frame
      const currentMarkers = new Map();
      const newlyDetectedIds = detectedMarkers.map(m => m.id);
      
      // Phase 1: Process markers detected in current frame
      for (const marker of detectedMarkers) {
        // Convert marker corners to format expected by POSIT
        const corners = this._prepareCorners(marker.corners);
        
        // Decide which POSIT implementation to use
        const usePosit1 = this.useAdaptiveAlgorithm ? 
          this._shouldUsePosit1(marker) : true;
        
        // Estimate pose
        const pose = usePosit1 ? 
          this.posit1.pose(corners) : 
          this.posit2.pose(corners);
        
        // Store algorithm choice for debugging
        if (this.debug) {
          this.debugData.algorithmUsed[marker.id] = usePosit1 ? 'posit1' : 'posit2';
        }
        
        // Apply Kalman filtering if enabled
        if (this.useKalmanFilter && this.kalmanFilters[marker.id]) {
          this._applyKalmanFilter(marker.id, pose);
        } else if (this.useKalmanFilter) {
          // Initialize Kalman filter for new marker
          this._initKalmanFilter(marker.id, pose);
        }
        
        // Store updated pose with marker info
        currentMarkers.set(marker.id, {
          id: marker.id,
          corners: marker.corners,
          pose: pose,
          lastSeen: this.frameCount,
          confidence: 1.0
        });
      }
      
      // Phase 2: Update markers not detected in current frame using optical flow
      if (this.useOpticalFlow && this.lastImageData) {
        for (const [id, markerInfo] of this.markers.entries()) {
          // Skip if marker was detected in current frame
          if (newlyDetectedIds.includes(id)) continue;
          
          // Only track recently seen markers (within last 30 frames)
          if (this.frameCount - markerInfo.lastSeen > 30) continue;
          
          // Track marker corners using optical flow
          const trackedCorners = this._trackCorners(
            this.lastImageData, 
            imageData, 
            markerInfo.corners
          );
          
          // If tracking successful, estimate pose from tracked corners
          if (trackedCorners) {
            // Convert tracked corners to format expected by POSIT
            const corners = this._prepareCorners(trackedCorners);
            
            // Use the same POSIT implementation as last time for consistency
            const usePosit1 = this.debugData.algorithmUsed[id] === 'posit1';
            
            // Estimate pose
            const pose = usePosit1 ? 
              this.posit1.pose(corners) : 
              this.posit2.pose(corners);
            
            // Apply Kalman filtering if enabled
            if (this.useKalmanFilter && this.kalmanFilters[id]) {
              this._applyKalmanFilter(id, pose);
            }
            
            // Reduce confidence based on frames since last detection
            const framesSinceDetection = this.frameCount - markerInfo.lastSeen;
            const confidence = Math.max(0.3, 1.0 - (framesSinceDetection / 30));
            
            // Store updated pose with marker info
            currentMarkers.set(id, {
              id: id,
              corners: trackedCorners,
              pose: pose,
              lastSeen: markerInfo.lastSeen, // Keep original lastSeen (not current frame)
              confidence: confidence,
              tracked: true // Flag as optically tracked
            });
          }
        }
      }
      
      // Phase 3: Update spatial model with newly detected marker positions
      if (this.knownMarkerPositions && currentMarkers.size > 0) {
        this._updateSpatialModel(currentMarkers);
      }
      
      // Phase 4: Predict poses for missing markers using spatial relationships
      if (this.spatialModel && currentMarkers.size > 0) {
        this._predictMissingMarkers(currentMarkers);
      }
      
      // Update state
      this.markers = currentMarkers;
      this.lastImageData = imageData;
      this.lastProcessTime = performance.now() - startTime;
      
      // Store processing time for debug
      if (this.debug) {
        this.debugData.trackingTime.push(this.lastProcessTime);
        if (this.debugData.trackingTime.length > 100) {
          this.debugData.trackingTime.shift();
        }
      }
      
      return this.markers;
    }
    
    /**
     * Track marker corners using optical flow
     * @private
     */
    _trackCorners(prevImage, currentImage, prevCorners) {
      if (!this.opticalFlowTracker) return null;
      
      // Track single marker using optical flow
      const trackedPoints = this.opticalFlowTracker.track(
        prevImage, 
        currentImage, 
        prevCorners
      );
      
      if (trackedPoints && trackedPoints.length === 4) {
        return trackedPoints;
      }
      
      return null;
    }
    
    /**
     * Convert marker corners to the format expected by POSIT
     * @private
     */
    _prepareCorners(corners) {
      // POSIT expects {x, y} format in a specific order
      // Make sure corners are ordered: top-left, top-right, bottom-right, bottom-left
      return corners.map(corner => ({ x: corner.x, y: corner.y }));
    }
    
    /**
     * Decide whether to use POSIT1 or POSIT2 implementation
     * Strategy: Use POSIT1 for markers at sharp angles (better for perspective distortion)
     * and POSIT2 for more frontal views (better numerical stability)
     * @private
     */
    _shouldUsePosit1(marker) {
      // Compute aspect ratio of marker as a proxy for perspective distortion
      const width1 = Math.sqrt(
        Math.pow(marker.corners[1].x - marker.corners[0].x, 2) +
        Math.pow(marker.corners[1].y - marker.corners[0].y, 2)
      );
      
      const width2 = Math.sqrt(
        Math.pow(marker.corners[2].x - marker.corners[3].x, 2) +
        Math.pow(marker.corners[2].y - marker.corners[3].y, 2)
      );
      
      const height1 = Math.sqrt(
        Math.pow(marker.corners[3].x - marker.corners[0].x, 2) +
        Math.pow(marker.corners[3].y - marker.corners[0].y, 2)
      );
      
      const height2 = Math.sqrt(
        Math.pow(marker.corners[2].x - marker.corners[1].x, 2) +
        Math.pow(marker.corners[2].y - marker.corners[1].y, 2)
      );
      
      // Calculate aspect ratio difference as a measure of perspective distortion
      const aspectRatio1 = width1 / height1;
      const aspectRatio2 = width2 / height2;
      const aspectDiff = Math.abs(aspectRatio1 - aspectRatio2);
      
      // If marker appears significantly distorted, use POSIT1
      return aspectDiff > 0.2;
    }
    
    /**
     * Initialize Kalman filter for a marker
     * @private
     */
    _initKalmanFilter(markerId, pose) {
      if (!this.useKalmanFilter) return;
      
      // Extract rotation and translation from pose
      const rotation = pose.bestRotation;
      const translation = pose.bestTranslation;
      
      // Create 12-state Kalman filter (6 for position/rotation, 6 for velocity)
      this.kalmanFilters[markerId] = new KalmanFilter(12, 6);
      
      // Initialize with first measurement
      // Convert rotation matrix to Euler angles for easier state representation
      const eulerAngles = this._rotationMatrixToEuler(rotation);
      
      // Initial measurement: [x, y, z, rotX, rotY, rotZ]
      const initialMeasurement = [
        translation[0], translation[1], translation[2],
        eulerAngles[0], eulerAngles[1], eulerAngles[2]
      ];
      
      // Initialize filter with first pose
      this.kalmanFilters[markerId].correct(initialMeasurement);
    }
    
    /**
     * Apply Kalman filtering to pose estimate
     * @private
     */
    _applyKalmanFilter(markerId, pose) {
      if (!this.useKalmanFilter || !this.kalmanFilters[markerId]) return;
      
      // Extract rotation and translation from pose
      const rotation = pose.bestRotation;
      const translation = pose.bestTranslation;
      
      // Convert rotation matrix to Euler angles
      const eulerAngles = this._rotationMatrixToEuler(rotation);
      
      // Current measurement: [x, y, z, rotX, rotY, rotZ]
      const measurement = [
        translation[0], translation[1], translation[2],
        eulerAngles[0], eulerAngles[1], eulerAngles[2]
      ];
      
      // Predict next state
      this.kalmanFilters[markerId].predict();
      
      // Correct with measurement
      const corrected = this.kalmanFilters[markerId].correct(measurement);
      
      // Update pose with filtered values
      // First 3 values are translation
      pose.bestTranslation[0] = corrected[0];
      pose.bestTranslation[1] = corrected[1];
      pose.bestTranslation[2] = corrected[2];
      
      // Convert Euler angles back to rotation matrix
      const filteredRotation = this._eulerToRotationMatrix(
        corrected[3], corrected[4], corrected[5]
      );
      
      // Update rotation matrix
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          pose.bestRotation[i][j] = filteredRotation[i][j];
        }
      }
    }
    
    /**
     * Convert rotation matrix to Euler angles
     * @private
     */
    _rotationMatrixToEuler(matrix) {
      // Implementation of rotation matrix to Euler angles conversion
      // Using ZYX convention (Tait-Bryan angles)
      
      let x, y, z;
      
      // Check for gimbal lock
      if (Math.abs(matrix[2][0]) >= 0.99999) {
        // Gimbal lock case
        x = 0;
        y = Math.PI / 2 * Math.sign(matrix[2][0]);
        z = Math.atan2(-matrix[0][1], matrix[1][1]);
      } else {
        // Regular case
        x = Math.atan2(matrix[2][1], matrix[2][2]);
        y = -Math.asin(matrix[2][0]);
        z = Math.atan2(matrix[1][0], matrix[0][0]);
      }
      
      return [x, y, z];
    }
    
    /**
     * Convert Euler angles to rotation matrix
     * @private
     */
    _eulerToRotationMatrix(x, y, z) {
      // Implementation of Euler angles to rotation matrix conversion
      // Using ZYX convention (Tait-Bryan angles)
      
      const cx = Math.cos(x);
      const sx = Math.sin(x);
      const cy = Math.cos(y);
      const sy = Math.sin(y);
      const cz = Math.cos(z);
      const sz = Math.sin(z);
      
      // Construct rotation matrix
      const matrix = [
        [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy]
      ];
      
      return matrix;
    }
    
    /**
     * Update spatial model with newly detected marker positions
     * @private
     */
    _updateSpatialModel(currentMarkers) {
      // If no spatial model exists, initialize it
      if (!this.spatialModel) {
        this.spatialModel = {
          referenceId: null,
          relativePoses: new Map()
        };
      }
      
      // Find best marker to use as reference (highest confidence)
      let bestConfidence = 0;
      let referenceId = null;
      let referencePose = null;
      
      for (const [id, markerInfo] of currentMarkers.entries()) {
        if (markerInfo.confidence > bestConfidence) {
          bestConfidence = markerInfo.confidence;
          referenceId = id;
          referencePose = markerInfo.pose;
        }
      }
      
      if (!referenceId) return; // No reliable marker found
      
      // Update reference marker
      this.spatialModel.referenceId = referenceId;
      
      // Calculate relative poses for all other markers
      for (const [id, markerInfo] of currentMarkers.entries()) {
        if (id === referenceId) continue;
        
        // Calculate relative pose from reference to this marker
        const relativePose = this._calculateRelativePose(
          referencePose, 
          markerInfo.pose
        );
        
        // Update or initialize relative pose in model
        if (this.spatialModel.relativePoses.has(id)) {
          // Blend with existing pose for stability (70% existing, 30% new)
          const existingPose = this.spatialModel.relativePoses.get(id);
          const blendedPose = this._blendPoses(existingPose, relativePose, 0.7);
          this.spatialModel.relativePoses.set(id, blendedPose);
        } else {
          // First time seeing this marker, store its relative pose
          this.spatialModel.relativePoses.set(id, relativePose);
        }
      }
    }
    
    /**
     * Calculate relative pose from reference marker to target marker
     * @private
     */
    _calculateRelativePose(referencePose, targetPose) {
      // Relative translation = targetTranslation - referenceTranslation in reference coordinate system
      // Need to transform targetTranslation to reference coordinate system first
      
      // Extract components
      const refR = referencePose.bestRotation;
      const refT = referencePose.bestTranslation;
      const targetR = targetPose.bestRotation;
      const targetT = targetPose.bestTranslation;
      
      // Calculate inverse rotation of reference (transpose for orthogonal matrix)
      const refRInv = this._transposeMatrix(refR);
      
      // Calculate relative rotation: R_rel = R_target * R_ref^(-1)
      const relativeR = this._multiplyMatrices(targetR, refRInv);
      
      // Calculate relative translation in reference coordinate system
      // T_rel = R_ref^(-1) * (T_target - T_ref)
      const tDiff = [
        targetT[0] - refT[0],
        targetT[1] - refT[1],
        targetT[2] - refT[2]
      ];
      
      const relativeT = this._multiplyMatrixVector(refRInv, tDiff);
      
      return {
        rotation: relativeR,
        translation: relativeT
      };
    }
    
    /**
     * Blend two poses for smooth transitions
     * @private
     */
    _blendPoses(pose1, pose2, weight1) {
      const weight2 = 1.0 - weight1;
      
      // Blend translations
      const blendedT = [
        pose1.translation[0] * weight1 + pose2.translation[0] * weight2,
        pose1.translation[1] * weight1 + pose2.translation[1] * weight2,
        pose1.translation[2] * weight1 + pose2.translation[2] * weight2
      ];
      
      // Blend rotations (convert to quaternions, slerp, convert back)
      // For simplicity, we'll just do a linear blend of the matrices
      // A proper implementation would use quaternions for rotation blending
      const blendedR = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ];
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          blendedR[i][j] = pose1.rotation[i][j] * weight1 + pose2.rotation[i][j] * weight2;
        }
      }
      
      // Orthogonalize the resulting matrix to ensure it's a valid rotation
      // Using Gram-Schmidt process
      const orthogonalR = this._orthogonalizeMatrix(blendedR);
      
      return {
        rotation: orthogonalR,
        translation: blendedT
      };
    }
    
    /**
     * Orthogonalize a matrix to ensure it's a valid rotation matrix
     * @private
     */
    _orthogonalizeMatrix(matrix) {
      // Extract rows
      const row1 = [matrix[0][0], matrix[0][1], matrix[0][2]];
      const row2 = [matrix[1][0], matrix[1][1], matrix[1][2]];
      
      // Normalize first row
      const norm1 = Math.sqrt(row1[0] * row1[0] + row1[1] * row1[1] + row1[2] * row1[2]);
      row1[0] /= norm1;
      row1[1] /= norm1;
      row1[2] /= norm1;
      
      // Make row2 orthogonal to row1
      const dot = row1[0] * row2[0] + row1[1] * row2[1] + row1[2] * row2[2];
      row2[0] -= dot * row1[0];
      row2[1] -= dot * row1[1];
      row2[2] -= dot * row1[2];
      
      // Normalize row2
      const norm2 = Math.sqrt(row2[0] * row2[0] + row2[1] * row2[1] + row2[2] * row2[2]);
      row2[0] /= norm2;
      row2[1] /= norm2;
      row2[2] /= norm2;
      
      // Compute row3 as cross product of row1 and row2
      const row3 = [
        row1[1] * row2[2] - row1[2] * row2[1],
        row1[2] * row2[0] - row1[0] * row2[2],
        row1[0] * row2[1] - row1[1] * row2[0]
      ];
      
      // Construct orthogonal matrix
      return [
        [row1[0], row1[1], row1[2]],
        [row2[0], row2[1], row2[2]],
        [row3[0], row3[1], row3[2]]
      ];
    }
    
    /**
     * Predict poses for markers not detected in current frame
     * @private
     */
    _predictMissingMarkers(currentMarkers) {
      // Check if we have a reference marker in current frame
      if (!this.spatialModel.referenceId || !currentMarkers.has(this.spatialModel.referenceId)) {
        return;
      }
      
      const referenceId = this.spatialModel.referenceId;
      const referencePose = currentMarkers.get(referenceId).pose;
      
      // Predict poses for all markers in spatial model
      for (const [id, relativePose] of this.spatialModel.relativePoses.entries()) {
        // Skip if marker is already in current frame
        if (currentMarkers.has(id)) continue;
        
        // Predict pose from reference pose and relative pose
        const predictedPose = this._predictPoseFromReference(
          referencePose, 
          relativePose
        );
        
        // Create a synthetic marker with predicted pose
        // Note: We can't predict corners accurately, so we'll skip those
        currentMarkers.set(id, {
          id: id,
          pose: predictedPose,
          lastSeen: this.markers.has(id) ? this.markers.get(id).lastSeen : 0,
          confidence: 0.5, // Reduced confidence for predicted poses
          predicted: true  // Flag as predicted
        });
      }
    }
    
    /**
     * Predict pose from reference pose and relative pose
     * @private
     */
    _predictPoseFromReference(referencePose, relativePose) {
      // Extract components
      const refR = referencePose.bestRotation;
      const refT = referencePose.bestTranslation;
      const relR = relativePose.rotation;
      const relT = relativePose.translation;
      
      // Calculate predicted rotation: R_pred = R_ref * R_rel
      const predictedR = this._multiplyMatrices(refR, relR);
      
      // Calculate predicted translation: T_pred = T_ref + R_ref * T_rel
      const rotatedRelT = this._multiplyMatrixVector(refR, relT);
      const predictedT = [
        refT[0] + rotatedRelT[0],
        refT[1] + rotatedRelT[1],
        refT[2] + rotatedRelT[2]
      ];
      
      // Create a synthetic pose object
      return {
        bestRotation: predictedR,
        bestTranslation: predictedT,
        bestError: 0, // Placeholder
        alternativeRotation: predictedR, // Same as best for simplicity
        alternativeTranslation: predictedT,
        alternativeError: 0 // Placeholder
      };
    }
    
    /**
     * Matrix multiplication helper
     * @private
     */
    _multiplyMatrices(a, b) {
      const result = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ];
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          for (let k = 0; k < 3; k++) {
            result[i][j] += a[i][k] * b[k][j];
          }
        }
      }
      
      return result;
    }
    
    /**
     * Matrix transpose helper
     * @private
     */
    _transposeMatrix(matrix) {
      const result = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ];
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          result[i][j] = matrix[j][i];
        }
      }
      
      return result;
    }
    
    /**
     * Matrix-vector multiplication helper
     * @private
     */
    _multiplyMatrixVector(matrix, vector) {
      const result = [0, 0, 0];
      
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          result[i] += matrix[i][j] * vector[j];
        }
      }
      
      return result;
    }
    
    /**
     * Reset tracking state
     */
    reset() {
      this.markers = new Map();
      this.lastImageData = null;
      this.frameCount = 0;
      this.spatialModel = null;
      
      // Reset Kalman filters
      if (this.useKalmanFilter) {
        this.kalmanFilters = {};
      }
      
      // Reset optical flow tracker
      if (this.useOpticalFlow && this.opticalFlowTracker) {
        this.opticalFlowTracker.reset();
      }
      
      // Reset debug data
      if (this.debug) {
        this.debugData = {
          algorithmUsed: {},
          positionError: {},
          rotationError: {},
          trackingTime: []
        };
      }
    }
    
    /**
     * Get current tracking statistics
     * @returns {Object} Tracking statistics
     */
    getStatistics() {
      // Calculate average tracking time
      const avgTrackingTime = this.debugData.trackingTime.length > 0 ?
        this.debugData.trackingTime.reduce((a, b) => a + b, 0) / this.debugData.trackingTime.length : 0;
      
      return {
        markerCount: this.markers.size,
        trackedFrames: this.frameCount,
        averageTrackingTimeMs: avgTrackingTime,
        algorithmUsage: this._calculateAlgorithmUsage(),
        trackedMarkerIds: Array.from(this.markers.keys()),
        spatialModelSize: this.spatialModel ? this.spatialModel.relativePoses.size : 0
      };
    }
    
    /**
     * Calculate algorithm usage statistics
     * @private
     */
    _calculateAlgorithmUsage() {
      if (!this.debug) return {};
      
      const counts = {
        posit1: 0,
        posit2: 0,
        total: Object.keys(this.debugData.algorithmUsed).length
      };
      
      for (const algo of Object.values(this.debugData.algorithmUsed)) {
        if (algo === 'posit1') counts.posit1++;
        else if (algo === 'posit2') counts.posit2++;
      }
      
      return {
        posit1Percentage: counts.total > 0 ? (counts.posit1 / counts.total) * 100 : 0,
        posit2Percentage: counts.total > 0 ? (counts.posit2 / counts.total) * 100 : 0,
        adaptiveSelection: this.useAdaptiveAlgorithm
      };
    }
    
    /**
     * Draw debug visualization on canvas
     * @param {CanvasRenderingContext2D} ctx - Canvas context
     */
    drawDebug(ctx) {
      if (!this.debug) return;
      
      // Draw marker poses
      for (const [id, markerInfo] of this.markers.entries()) {
        // Draw marker ID and tracking status
        const status = markerInfo.predicted ? 'PREDICTED' : 
                       markerInfo.tracked ? 'TRACKED' : 
                       'DETECTED';
        
        // Calculate center of marker (if corners available)
        let centerX = 0, centerY = 0;
        
        if (markerInfo.corners) {
          for (const corner of markerInfo.corners) {
            centerX += corner.x;
            centerY += corner.y;
          }
          centerX /= markerInfo.corners.length;
          centerY /= markerInfo.corners.length;
        } else if (markerInfo.predicted) {
          // For predicted markers, just place text in center of screen
          centerX = ctx.canvas.width / 2;
          centerY = ctx.canvas.height / 2 + id * 30;
        }
        
        // Draw marker ID and status
        ctx.fillStyle = markerInfo.predicted ? 'orange' : 
                       markerInfo.tracked ? 'yellow' : 
                       'lime';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(`ID: ${id} (${status})`, centerX, centerY);
        
        // Draw confidence
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(`Confidence: ${(markerInfo.confidence * 100).toFixed(0)}%`, centerX, centerY + 20);
        
        // Draw which POSIT algorithm was used
        if (this.debugData.algorithmUsed[id]) {
          ctx.fillText(`Algorithm: ${this.debugData.algorithmUsed[id]}`, centerX, centerY + 40);
        }
        
        // Draw marker axes if pose is available
        if (markerInfo.pose && !markerInfo.predicted) {
          this._drawMarkerAxes(ctx, markerInfo.pose, markerInfo.corners[0]);
        }
      }
      
      // Draw tracking statistics
      const stats = this.getStatistics();
      ctx.fillStyle = 'white';
      ctx.font = '14px monospace';
      ctx.fillText(`Tracked markers: ${stats.markerCount}`, 10, ctx.canvas.height - 70);
      ctx.fillText(`Frame #: ${this.frameCount}`, 10, ctx.canvas.height - 50);
      ctx.fillText(`Tracking time: ${stats.averageTrackingTimeMs.toFixed(2)}ms`, 10, ctx.canvas.height - 30);
      
      // Draw algorithm usage statistics
      if (stats.algorithmUsage.total > 0) {
        ctx.fillText(
          `POSIT1: ${stats.algorithmUsage.posit1Percentage.toFixed(1)}%, POSIT2: ${stats.algorithmUsage.posit2Percentage.toFixed(1)}%`,
          10, ctx.canvas.height - 10
        );
      }
      
      // If we have a spatial model, visualize it
      if (this.spatialModel && this.spatialModel.relativePoses.size > 0) {
        ctx.fillStyle = 'cyan';
        ctx.fillText(`Spatial model: ${this.spatialModel.relativePoses.size} markers`, 10, ctx.canvas.height - 90);
        ctx.fillText(`Reference: ${this.spatialModel.referenceId}`, 250, ctx.canvas.height - 90);
      }
    }
    
    /**
     * Draw 3D axes for a marker
     * @private
     */
    _drawMarkerAxes(ctx, pose, origin) {
      const rotation = pose.bestRotation;
      const translation = pose.bestTranslation;
      const size = this.markerSize / 2; // Half marker size
      
      // Define the 3D axes endpoints
      const axes = [
        [size, 0, 0],  // X-axis (red)
        [0, size, 0],  // Y-axis (green)
        [0, 0, size]   // Z-axis (blue)
      ];
      
      // Project origin
      const originProj = {
        x: this.focalLength * translation[0] / translation[2] + origin.x,
        y: this.focalLength * translation[1] / translation[2] + origin.y
      };
      
      // Colors for the axes
      const colors = ['red', 'green', 'blue'];
      
      // Draw each axis
      for (let i = 0; i < axes.length; i++) {
        // Apply rotation and translation to the axis endpoint
        const endpoint = [0, 0, 0];
        for (let j = 0; j < 3; j++) {
          for (let k = 0; k < 3; k++) {
            endpoint[j] += rotation[j][k] * axes[i][k];
          }
          endpoint[j] += translation[j];
        }
        
        // Project endpoint to 2D
        const endpointProj = {
          x: this.focalLength * endpoint[0] / endpoint[2] + origin.x,
          y: this.focalLength * endpoint[1] / endpoint[2] + origin.y
        };
        
        // Draw the axis line
        ctx.strokeStyle = colors[i];
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(originProj.x, originProj.y);
        ctx.lineTo(endpointProj.x, endpointProj.y);
        ctx.stroke();
      }
    }
    
    /**
     * Set new tracker options
     * @param {Object} options - New options
     */
    setOptions(options = {}) {
      // Update core parameters
      if (options.markerSize !== undefined) {
        this.markerSize = options.markerSize;
        // Reinitialize POSIT
        this.posit1 = new POS.Posit(this.markerSize, this.focalLength);
        this.posit2 = new POS.Posit(this.markerSize, this.focalLength);
      }
      
      if (options.focalLength !== undefined) {
        this.focalLength = options.focalLength;
        // Reinitialize POSIT
        this.posit1 = new POS.Posit(this.markerSize, this.focalLength);
        this.posit2 = new POS.Posit(this.markerSize, this.focalLength);
      }
      
      if (options.useKalmanFilter !== undefined) {
        const wasEnabled = this.useKalmanFilter;
        this.useKalmanFilter = options.useKalmanFilter;
        
        // Reset Kalman filters if toggling on
        if (!wasEnabled && this.useKalmanFilter) {
          this.kalmanFilters = {};
        }
      }
      
      if (options.useOpticalFlow !== undefined) {
        const wasEnabled = this.useOpticalFlow;
        this.useOpticalFlow = options.useOpticalFlow;
        
        // Initialize optical flow if toggling on
        if (!wasEnabled && this.useOpticalFlow && !this.opticalFlowTracker) {
          this.opticalFlowTracker = new OpticalFlowTracker({
            maxTrackedFrames: 30,
            winSize: 15,
            maxLevel: 2,
            maxError: 10000
          });
        }
      }
      
      if (options.useAdaptiveAlgorithm !== undefined) {
        this.useAdaptiveAlgorithm = options.useAdaptiveAlgorithm;
      }
      
      if (options.debug !== undefined) {
        this.debug = options.debug;
      }
      
      // Update known marker positions
      if (options.knownMarkerPositions !== undefined) {
        this.knownMarkerPositions = options.knownMarkerPositions;
        // Reset spatial model when changing marker positions
        this.spatialModel = null;
      }
      
      // Update optical flow options
      if (this.useOpticalFlow && this.opticalFlowTracker && options.opticalFlowOptions) {
        this.opticalFlowTracker.setOptions(options.opticalFlowOptions);
      }
    }
  }
  
  /**
   * Helper class to create a multi-tag pose tracking system with default options
   */
  class ARMultiTracker {
    /**
     * Create a new AR multi-tracker with default settings
     * @param {Object} options - Configuration options
     * @returns {MultiTagPoseTracker} Configured tracker
     */
    static create(options = {}) {
      // Default options for general AR use
      const defaultOptions = {
        markerSize: 50, // 50mm markers
        focalLength: 700, // Typical webcam focal length
        useKalmanFilter: true,
        useOpticalFlow: true,
        useAdaptiveAlgorithm: true
      };
      
      // Merge default options with provided options
      const mergedOptions = {...defaultOptions, ...options};
      
      return new MultiTagPoseTracker(mergedOptions);
    }
    
    /**
     * Create a lightweight tracker for mobile devices
     * @param {Object} options - Configuration options
     * @returns {MultiTagPoseTracker} Configured lightweight tracker
     */
    static createLightweight(options = {}) {
      // Default options optimized for mobile performance
      const defaultOptions = {
        markerSize: 50,
        focalLength: 700,
        useKalmanFilter: false, // Disable Kalman for performance
        useOpticalFlow: true,
        useAdaptiveAlgorithm: false // Just use POSIT1 for simplicity
      };
      
      // Merge default options with provided options
      const mergedOptions = {...defaultOptions, ...options};
      
      return new MultiTagPoseTracker(mergedOptions);
    }
  }
  
  // Export for both browser and Node.js environments
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      MultiTagPoseTracker,
      ARMultiTracker
    };
  } else {
    // Browser global
    window.MultiTagPoseTracker = MultiTagPoseTracker;
    window.ARMultiTracker = ARMultiTracker;
  }