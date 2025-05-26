/**
 * Kalman Filter for AR pose tracking
 * This helps smooth out position and rotation during movement
 */
class KalmanFilter {
    constructor(stateSize, measSize) {
      this.stateSize = stateSize;  // State vector size (position, velocity)
      this.measSize = measSize;    // Measurement vector size (position)
      
      // State vector [x, y, z, vx, vy, vz]
      this.statePost = new Array(stateSize).fill(0);
      this.statePre = new Array(stateSize).fill(0);
      
      // Process noise covariance matrix (how much we expect the process to change)
      this.processNoiseCov = new Array(stateSize * stateSize).fill(0);
      for (let i = 0; i < stateSize; i++) {
        // Add some process noise to the diagonal elements
        this.processNoiseCov[i * stateSize + i] = 0.01;
      }
      // Higher process noise for velocities
      for (let i = measSize; i < stateSize; i++) {
        this.processNoiseCov[i * stateSize + i] = 0.1;
      }
      
      // Measurement noise covariance (how much to trust measurements)
      this.measurementNoiseCov = new Array(measSize * measSize).fill(0);
      for (let i = 0; i < measSize; i++) {
        this.measurementNoiseCov[i * measSize + i] = 0.1;
      }
      
      // Error covariance matrices
      this.errorCovPre = new Array(stateSize * stateSize).fill(0);
      this.errorCovPost = new Array(stateSize * stateSize).fill(0);
      for (let i = 0; i < stateSize; i++) {
        this.errorCovPost[i * stateSize + i] = 1;
      }
      
      // Create transition matrix (state update matrix)
      // For position and velocity, we use a constant velocity model:
      // [1 0 0 dt 0  0 ]
      // [0 1 0 0  dt 0 ]
      // [0 0 1 0  0  dt]
      // [0 0 0 1  0  0 ]
      // [0 0 0 0  1  0 ]
      // [0 0 0 0  0  1 ]
      this.transitionMatrix = new Array(stateSize * stateSize).fill(0);
      for (let i = 0; i < stateSize; i++) {
        this.transitionMatrix[i * stateSize + i] = 1; // Identity matrix
      }
      
      // Positions are updated by velocities
      const dt = 1.0; // Time step (can be adjusted)
      for (let i = 0; i < measSize; i++) {
        this.transitionMatrix[i * stateSize + (i + measSize)] = dt;
      }
      
      // Measurement matrix - we only measure position, not velocities
      this.measurementMatrix = new Array(measSize * stateSize).fill(0);
      for (let i = 0; i < measSize; i++) {
        this.measurementMatrix[i * stateSize + i] = 1;
      }
      
      // Kalman gain
      this.gain = new Array(stateSize * measSize).fill(0);
      
      this.initialized = false;
      this.lastUpdateTime = 0;
    }
    
    // Update the time delta in the transition matrix
    updateDeltaTime() {
      const currentTime = Date.now();
      if (this.lastUpdateTime === 0) {
        this.lastUpdateTime = currentTime;
        return;
      }
      
      const dt = (currentTime - this.lastUpdateTime) / 1000.0; // Convert to seconds
      this.lastUpdateTime = currentTime;
      
      // Update time-dependent elements in transition matrix
      for (let i = 0; i < this.measSize; i++) {
        this.transitionMatrix[i * this.stateSize + (i + this.measSize)] = dt;
      }
    }
    
    // Predict next state
    predict() {
      this.updateDeltaTime();
      
      if (!this.initialized) {
        return this.statePre;
      }
      
      // statePre = transitionMatrix * statePost
      for (let i = 0; i < this.stateSize; i++) {
        this.statePre[i] = 0;
        for (let j = 0; j < this.stateSize; j++) {
          this.statePre[i] += this.transitionMatrix[i * this.stateSize + j] * this.statePost[j];
        }
      }
      
      // errorCovPre = transitionMatrix * errorCovPost * transitionMatrix' + processNoiseCov
      const temp = new Array(this.stateSize * this.stateSize).fill(0);
      
      // temp = transitionMatrix * errorCovPost
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.stateSize; j++) {
          temp[i * this.stateSize + j] = 0;
          for (let k = 0; k < this.stateSize; k++) {
            temp[i * this.stateSize + j] += 
              this.transitionMatrix[i * this.stateSize + k] * 
              this.errorCovPost[k * this.stateSize + j];
          }
        }
      }
      
      // errorCovPre = temp * transitionMatrix' + processNoiseCov
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.stateSize; j++) {
          this.errorCovPre[i * this.stateSize + j] = this.processNoiseCov[i * this.stateSize + j];
          for (let k = 0; k < this.stateSize; k++) {
            this.errorCovPre[i * this.stateSize + j] += 
              temp[i * this.stateSize + k] * 
              this.transitionMatrix[j * this.stateSize + k]; // Using transposed index
          }
        }
      }
      
      return this.statePre;
    }
    
    // Correct state based on measurement
    correct(measurement) {
      if (!this.initialized) {
        // Initialize with first measurement
        for (let i = 0; i < this.measSize; i++) {
          this.statePost[i] = measurement[i];
        }
        this.initialized = true;
        return this.statePost;
      }
      
      // temp1 = errorCovPre * measurementMatrix'
      const temp1 = new Array(this.stateSize * this.measSize).fill(0);
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.measSize; j++) {
          temp1[i * this.measSize + j] = 0;
          for (let k = 0; k < this.stateSize; k++) {
            temp1[i * this.measSize + j] += 
              this.errorCovPre[i * this.stateSize + k] * 
              this.measurementMatrix[j * this.stateSize + k]; // Using transposed index
          }
        }
      }
      
      // temp2 = measurementMatrix * errorCovPre * measurementMatrix' + measurementNoiseCov
      const temp2 = new Array(this.measSize * this.measSize).fill(0);
      for (let i = 0; i < this.measSize; i++) {
        for (let j = 0; j < this.measSize; j++) {
          temp2[i * this.measSize + j] = this.measurementNoiseCov[i * this.measSize + j];
          for (let k = 0; k < this.stateSize; k++) {
            for (let l = 0; l < this.stateSize; l++) {
              temp2[i * this.measSize + j] += 
                this.measurementMatrix[i * this.stateSize + k] * 
                this.errorCovPre[k * this.stateSize + l] * 
                this.measurementMatrix[j * this.stateSize + l]; // Using transposed index
            }
          }
        }
      }
      
      // Invert temp2 (simplified for 3x3 or smaller matrix)
      const invTemp2 = this.invert3x3(temp2, this.measSize);
      
      // gain = temp1 * invTemp2
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.measSize; j++) {
          this.gain[i * this.measSize + j] = 0;
          for (let k = 0; k < this.measSize; k++) {
            this.gain[i * this.measSize + j] += temp1[i * this.measSize + k] * invTemp2[k * this.measSize + j];
          }
        }
      }
      
      // Calculate innovation: measurement - measurementMatrix * statePre
      const innovation = new Array(this.measSize).fill(0);
      for (let i = 0; i < this.measSize; i++) {
        innovation[i] = measurement[i];
        for (let j = 0; j < this.stateSize; j++) {
          innovation[i] -= this.measurementMatrix[i * this.stateSize + j] * this.statePre[j];
        }
      }
      
      // statePost = statePre + gain * innovation
      for (let i = 0; i < this.stateSize; i++) {
        this.statePost[i] = this.statePre[i];
        for (let j = 0; j < this.measSize; j++) {
          this.statePost[i] += this.gain[i * this.measSize + j] * innovation[j];
        }
      }
      
      // Update error covariance: errorCovPost = (I - gain * measurementMatrix) * errorCovPre
      const temp3 = new Array(this.stateSize * this.stateSize).fill(0);
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.stateSize; j++) {
          temp3[i * this.stateSize + j] = (i === j ? 1 : 0);
          for (let k = 0; k < this.measSize; k++) {
            temp3[i * this.stateSize + j] -= this.gain[i * this.measSize + k] * this.measurementMatrix[k * this.stateSize + j];
          }
        }
      }
      
      for (let i = 0; i < this.stateSize; i++) {
        for (let j = 0; j < this.stateSize; j++) {
          this.errorCovPost[i * this.stateSize + j] = 0;
          for (let k = 0; k < this.stateSize; k++) {
            this.errorCovPost[i * this.stateSize + j] += temp3[i * this.stateSize + k] * this.errorCovPre[k * this.stateSize + j];
          }
        }
      }
      
      return this.statePost;
    }
    
    // Reset the filter
    reset() {
      this.initialized = false;
      this.lastUpdateTime = 0;
      this.statePost.fill(0);
      this.statePre.fill(0);
      
      // Reset error covariance matrix to identity
      this.errorCovPost = new Array(this.stateSize * this.stateSize).fill(0);
      for (let i = 0; i < this.stateSize; i++) {
        this.errorCovPost[i * this.stateSize + i] = 1;
      }
    }
    
    // Invert a small matrix (specifically for 3x3 or smaller)
    invert3x3(matrix, size) {
      if (size > 3) {
        throw new Error("This simplified inversion only works for 3x3 or smaller matrices");
      }
      
      const result = new Array(size * size).fill(0);
      
      if (size === 1) {
        result[0] = 1.0 / matrix[0];
        return result;
      }
      
      if (size === 2) {
        const det = matrix[0] * matrix[3] - matrix[1] * matrix[2];
        const invDet = 1.0 / det;
        
        result[0] = matrix[3] * invDet;
        result[1] = -matrix[1] * invDet;
        result[2] = -matrix[2] * invDet;
        result[3] = matrix[0] * invDet;
        
        return result;
      }
      
      if (size === 3) {
        const a = matrix[0], b = matrix[1], c = matrix[2];
        const d = matrix[3], e = matrix[4], f = matrix[5];
        const g = matrix[6], h = matrix[7], i = matrix[8];
        
        const A = e * i - f * h;
        const B = -(d * i - f * g);
        const C = d * h - e * g;
        const D = -(b * i - c * h);
        const E = a * i - c * g;
        const F = -(a * h - b * g);
        const G = b * f - c * e;
        const H = -(a * f - c * d);
        const I = a * e - b * d;
        
        const det = a * A + b * B + c * C;
        const invDet = 1.0 / det;
        
        result[0] = A * invDet;
        result[1] = D * invDet;
        result[2] = G * invDet;
        result[3] = B * invDet;
        result[4] = E * invDet;
        result[5] = H * invDet;
        result[6] = C * invDet;
        result[7] = F * invDet;
        result[8] = I * invDet;
        
        return result;
      }
    }
  }