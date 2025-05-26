/**
 * Extended Kalman Filter (EKF) for AR Marker Tracking
 * 
 * This implementation is specifically designed to handle non-linear motion
 * like rotating crank mechanisms with sudden direction changes.
 * 
 * Key advantages over standard Kalman Filter:
 * - Handles non-linear motion models through linearization
 * - Better acceleration modeling for sudden movements
 * - Can incorporate physical constraints and mechanical knowledge
 * - Improves tracking during transitions (rising to falling)
 */

class ExtendedKalmanFilter {
    /**
     * Create a new Extended Kalman Filter
     * @param {Object} options - Configuration options
     * @param {Number} options.stateSize - Size of state vector (typically 9 for position+velocity+acceleration in 3D)
     * @param {Number} options.measurementSize - Size of measurement vector (typically 3 for position in 3D)
     * @param {Function} options.stateTransitionFn - Non-linear state transition function f(x,dt)
     * @param {Function} options.measurementFn - Non-linear measurement function h(x)
     * @param {Function} options.stateTransitionJacobian - Jacobian of state transition Df/dx
     * @param {Function} options.measurementJacobian - Jacobian of measurement function Dh/dx
     * @param {Array} options.initialState - Initial state vector [x,y,z,vx,vy,vz,ax,ay,az]
     * @param {Boolean} options.useGravityModel - Whether to use gravity model for vertical motion
     * @param {Boolean} options.useCrankModel - Whether to use specialized crank motion model
     */
    constructor(options = {}) {
      // State dimensions
      this.stateSize = options.stateSize || 9; // position, velocity, acceleration in 3D
      this.measurementSize = options.measurementSize || 3; // position in 3D
      
      // Model functions (non-linear)
      this.stateTransitionFn = options.stateTransitionFn || this.defaultStateTransition;
      this.measurementFn = options.measurementFn || this.defaultMeasurement;
      
      // Jacobian functions (linearization)
      this.stateTransitionJacobian = options.stateTransitionJacobian || this.defaultStateTransitionJacobian;
      this.measurementJacobian = options.measurementJacobian || this.defaultMeasurementJacobian;
      
      // Physical model options
      this.useGravityModel = options.useGravityModel !== undefined ? options.useGravityModel : true;
      this.useCrankModel = options.useCrankModel !== undefined ? options.useCrankModel : false;
      this.crankRadius = options.crankRadius || 50; // radius of crank in world units
      this.crankCenter = options.crankCenter || [0, 0, 0]; // center of rotation for crank
      this.crankPlane = options.crankPlane || 'yz'; // plane of rotation
      this.gravityAcceleration = options.gravityAcceleration || 9.81; // m/s^2
      
      // State vector and covariance
      this.state = options.initialState ? [...options.initialState] : new Array(this.stateSize).fill(0);
      this.errorCovariance = this.createIdentityMatrix(this.stateSize);
      
      // Process noise covariance (Q)
      this.processNoise = options.processNoise || this.createDefaultProcessNoise();
      
      // Measurement noise covariance (R)
      this.measurementNoise = options.measurementNoise || this.createDefaultMeasurementNoise();
      
      // Timing
      this.lastUpdateTime = 0;
      
      // Status flags
      this.initialized = false;
      this.detectedRapidVerticalMovement = false;
      this.pastPositionY = null;
      this.directionChangeThreshold = options.directionChangeThreshold || 10;
      
      // History for tracking
      this.positionHistory = [];
      this.velocityHistory = [];
      this.historyMaxLength = options.historyMaxLength || 10;
      
      // Debug data
      this.debug = options.debug || false;
      this.debugData = {
        predictedState: null,
        correctedState: null,
        innovation: null,
        innovationCovariance: null,
        kalmanGain: null
      };
    }
    
    /**
     * Default non-linear state transition function
     * x_{k+1} = f(x_k, dt)
     * Includes gravity model and optional crank model
     * @param {Array} state - Current state [x,y,z,vx,vy,vz,ax,ay,az]
     * @param {Number} dt - Time step
     * @returns {Array} Next state
     */
    defaultStateTransition(state, dt) {
      const nextState = [...state];
      const [x, y, z, vx, vy, vz, ax, ay, az] = state;
      
      // Position update with acceleration
      nextState[0] = x + vx * dt + 0.5 * ax * dt * dt; // x
      nextState[1] = y + vy * dt + 0.5 * ay * dt * dt; // y
      nextState[2] = z + vz * dt + 0.5 * az * dt * dt; // z
      
      // Velocity update with acceleration
      nextState[3] = vx + ax * dt; // vx
      nextState[4] = vy + ay * dt; // vy
      nextState[5] = vz + az * dt; // vz
      
      // Apply gravity model - only applied to vertical axis (y)
      if (this.useGravityModel) {
        // When moving downward, gravity reinforces acceleration
        if (vy < 0) {
          nextState[4] = vy + (ay - this.gravityAcceleration) * dt; // vy with gravity
          
          // Stronger effect when we detect a rapid vertical movement (downward)
          if (this.detectedRapidVerticalMovement) {
            // Increase the effect of gravity by 50% during rapid downward motion
            nextState[4] = vy + (ay - this.gravityAcceleration * 1.5) * dt;
          }
        }
      }
      
      // Apply crank model if enabled
      if (this.useCrankModel) {
        this.applyCrankModelConstraints(nextState);
      }
      
      return nextState;
    }
    
    /**
     * Apply constraints from the crank model to the state
     * This forces the position to follow a circular path
     * @param {Array} state - State to constrain
     */
    applyCrankModelConstraints(state) {
      // Only apply if we have history to work with
      if (this.positionHistory.length < 2) return state;
      
      const [cx, cy, cz] = this.crankCenter;
      
      // Calculate current angle in the crank plane
      let currentAngle;
      let radius;
      
      if (this.crankPlane === 'xy') {
        // Rotating in xy plane
        const dx = state[0] - cx;
        const dy = state[1] - cy;
        currentAngle = Math.atan2(dy, dx);
        radius = Math.sqrt(dx*dx + dy*dy);
      } else if (this.crankPlane === 'xz') {
        // Rotating in xz plane
        const dx = state[0] - cx;
        const dz = state[2] - cz;
        currentAngle = Math.atan2(dz, dx);
        radius = Math.sqrt(dx*dx + dz*dz);
      } else {
        // Default: yz plane
        const dy = state[1] - cy;
        const dz = state[2] - cz;
        currentAngle = Math.atan2(dy, dz);
        radius = Math.sqrt(dy*dy + dz*dz);
      }
      
      // If radius is way off from expected, bring it back to crank radius
      if (Math.abs(radius - this.crankRadius) > this.crankRadius * 0.2) {
        const targetRadius = this.crankRadius;
        const correctionFactor = targetRadius / radius;
        
        if (this.crankPlane === 'xy') {
          state[0] = cx + (state[0] - cx) * correctionFactor;
          state[1] = cy + (state[1] - cy) * correctionFactor;
        } else if (this.crankPlane === 'xz') {
          state[0] = cx + (state[0] - cx) * correctionFactor;
          state[2] = cz + (state[2] - cz) * correctionFactor;
        } else {
          state[1] = cy + (state[1] - cy) * correctionFactor;
          state[2] = cz + (state[2] - cz) * correctionFactor;
        }
      }
      
      return state;
    }
    
    /**
     * Default non-linear measurement function
     * z = h(x)
     * Simply extracts position from state
     * @param {Array} state - Full state
     * @returns {Array} Measurement (just position)
     */
    defaultMeasurement(state) {
      // Default: just return the position
      return [state[0], state[1], state[2]];
    }
    
    /**
     * Default Jacobian of state transition function
     * Df/dx - linearization of state transition at current state
     * @param {Array} state - Current state
     * @param {Number} dt - Time step
     * @returns {Array} Jacobian matrix as 2D array
     */
    defaultStateTransitionJacobian(state, dt) {
      // For our motion model, the Jacobian is constant
      // This is the linearization of our state transition
      const F = this.createIdentityMatrix(this.stateSize);
      
      // Position rows
      F[0][3] = dt; // dx/dvx
      F[1][4] = dt; // dy/dvy
      F[2][5] = dt; // dz/dvz
      
      F[0][6] = 0.5 * dt * dt; // dx/dax
      F[1][7] = 0.5 * dt * dt; // dy/day
      F[2][8] = 0.5 * dt * dt; // dz/daz
      
      // Velocity rows
      F[3][6] = dt; // dvx/dax
      F[4][7] = dt; // dvy/day
      F[5][8] = dt; // dvz/daz
      
      return F;
    }
    
    /**
     * Default Jacobian of measurement function
     * Dh/dx - linearization of measurement function at current state
     * @param {Array} state - Current state
     * @returns {Array} Jacobian matrix as 2D array
     */
    defaultMeasurementJacobian(state) {
      // For our simple measurement model, the Jacobian is constant
      // It's just extracting the position elements from the state
      const H = new Array(this.measurementSize).fill(0).map(() => 
        new Array(this.stateSize).fill(0));
      
      // Set the measurement Jacobian values
      // We're measuring position directly
      H[0][0] = 1; // dx/dx
      H[1][1] = 1; // dy/dy
      H[2][2] = 1; // dz/dz
      
      return H;
    }
    
    /**
     * Create a default process noise covariance matrix
     * @returns {Array} Process noise covariance matrix
     */
    createDefaultProcessNoise() {
      const Q = new Array(this.stateSize).fill(0).map(() => 
        new Array(this.stateSize).fill(0));
      
      // Set variances for position
      Q[0][0] = 0.01; // x
      Q[1][1] = 0.01; // y
      Q[2][2] = 0.01; // z
      
      // Set variances for velocity
      Q[3][3] = 0.1; // vx
      Q[4][4] = 0.1; // vy
      Q[5][5] = 0.1; // vz
      
      // Set variances for acceleration
      Q[6][6] = 1.0; // ax
      Q[7][7] = 1.0; // ay
      Q[8][8] = 1.0; // az
      
      return Q;
    }
    
    /**
     * Create a default measurement noise covariance matrix
     * @returns {Array} Measurement noise covariance matrix
     */
    createDefaultMeasurementNoise() {
      const R = new Array(this.measurementSize).fill(0).map(() => 
        new Array(this.measurementSize).fill(0));
      
      // Set measurement noise variances
      R[0][0] = 1.0; // x measurement variance
      R[1][1] = 1.0; // y measurement variance
      R[2][2] = 1.0; // z measurement variance
      
      return R;
    }
    
    /**
     * Create an identity matrix of given size
     * @param {Number} size - Size of the matrix
     * @returns {Array} Identity matrix as 2D array
     */
    createIdentityMatrix(size) {
      const matrix = new Array(size).fill(0).map(() => 
        new Array(size).fill(0));
      
      for (let i = 0; i < size; i++) {
        matrix[i][i] = 1;
      }
      
      return matrix;
    }
    
    /**
     * Matrix multiplication: C = A * B
     * @param {Array} A - First matrix (2D array)
     * @param {Array} B - Second matrix (2D array)
     * @returns {Array} Result matrix (2D array)
     */
    matrixMultiply(A, B) {
      const rowsA = A.length;
      const colsA = A[0].length;
      const rowsB = B.length;
      const colsB = B[0].length;
      
      if (colsA !== rowsB) {
        throw new Error('Matrix dimensions do not match for multiplication');
      }
      
      const C = new Array(rowsA).fill(0).map(() => 
        new Array(colsB).fill(0));
      
      for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
          for (let k = 0; k < colsA; k++) {
            C[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      
      return C;
    }
    
    /**
     * Matrix addition: C = A + B
     * @param {Array} A - First matrix (2D array)
     * @param {Array} B - Second matrix (2D array)
     * @returns {Array} Result matrix (2D array)
     */
    matrixAdd(A, B) {
      const rows = A.length;
      const cols = A[0].length;
      
      const C = new Array(rows).fill(0).map(() => 
        new Array(cols).fill(0));
      
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          C[i][j] = A[i][j] + B[i][j];
        }
      }
      
      return C;
    }
    
    /**
     * Matrix subtraction: C = A - B
     * @param {Array} A - First matrix (2D array)
     * @param {Array} B - Second matrix (2D array)
     * @returns {Array} Result matrix (2D array)
     */
    matrixSubtract(A, B) {
      const rows = A.length;
      const cols = A[0].length;
      
      const C = new Array(rows).fill(0).map(() => 
        new Array(cols).fill(0));
      
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          C[i][j] = A[i][j] - B[i][j];
        }
      }
      
      return C;
    }
    
    /**
     * Matrix transpose: B = A^T
     * @param {Array} A - Input matrix (2D array)
     * @returns {Array} Transposed matrix (2D array)
     */
    matrixTranspose(A) {
      const rows = A.length;
      const cols = A[0].length;
      
      const B = new Array(cols).fill(0).map(() => 
        new Array(rows).fill(0));
      
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          B[j][i] = A[i][j];
        }
      }
      
      return B;
    }
    
    /**
     * Matrix inversion (simple implementation for small matrices)
     * @param {Array} A - Input matrix (2D array)
     * @returns {Array} Inverted matrix (2D array)
     */
    matrixInverse(A) {
      const n = A.length;
      
      // For 1x1 matrix, inversion is just the reciprocal
      if (n === 1) {
        return [[1 / A[0][0]]];
      }
      
      // For larger matrices, we'll use a simple approach
      // This is not the most efficient or numerically stable method
      // but works for our small matrices in EKF
      
      // Create augmented matrix [A|I]
      const augmented = [];
      for (let i = 0; i < n; i++) {
        augmented[i] = [...A[i]];
        for (let j = 0; j < n; j++) {
          augmented[i].push(i === j ? 1 : 0);
        }
      }
      
      // Apply Gauss-Jordan elimination
      for (let i = 0; i < n; i++) {
        // Find pivot
        let pivotRow = i;
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(augmented[j][i]) > Math.abs(augmented[pivotRow][i])) {
            pivotRow = j;
          }
        }
        
        // Swap rows if needed
        if (pivotRow !== i) {
          [augmented[i], augmented[pivotRow]] = [augmented[pivotRow], augmented[i]];
        }
        
        // Scale pivot row
        const pivot = augmented[i][i];
        if (pivot === 0) {
          throw new Error('Matrix is singular and cannot be inverted');
        }
        
        for (let j = 0; j < 2 * n; j++) {
          augmented[i][j] /= pivot;
        }
        
        // Eliminate other rows
        for (let j = 0; j < n; j++) {
          if (j !== i) {
            const factor = augmented[j][i];
            for (let k = 0; k < 2 * n; k++) {
              augmented[j][k] -= factor * augmented[i][k];
            }
          }
        }
      }
      
      // Extract the right half as the inverse
      const inverse = new Array(n).fill(0).map(() => 
        new Array(n).fill(0));
      
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          inverse[i][j] = augmented[i][j + n];
        }
      }
      
      return inverse;
    }
    
    /**
     * Predict the next state based on the current state
     * @param {Number} timestamp - Current timestamp in milliseconds (optional)
     * @returns {Array} Predicted state
     */
    predict(timestamp) {
      const currentTime = timestamp || performance.now();
      const dt = this.initialized ? (currentTime - this.lastUpdateTime) / 1000 : 0.033;
      
      // Check for very large time steps (which can cause instability)
      const safeTimeStep = Math.min(dt, 0.1);
      
      // 1. Project state ahead using non-linear state transition function
      const predictedState = this.stateTransitionFn(this.state, safeTimeStep);
      
      // 2. Project error covariance ahead
      // P = F * P * F' + Q
      const F = this.stateTransitionJacobian(this.state, safeTimeStep);
      const Ft = this.matrixTranspose(F);
      
      const FP = this.matrixMultiply(F, this.errorCovariance);
      const FPFt = this.matrixMultiply(FP, Ft);
      
      // Add process noise
      this.errorCovariance = this.matrixAdd(FPFt, this.processNoise);
      
      // Update state
      this.state = predictedState;
      
      // Store debug data
      if (this.debug) {
        this.debugData.predictedState = [...this.state];
      }
      
      if (!this.initialized) {
        this.initialized = true;
      }
      
      // Return the predicted state
      return [...this.state];
    }
    
    /**
     * Update step - correct state based on measurement
     * @param {Array} measurement - Measurement vector [x, y, z]
     * @param {Number} timestamp - Current timestamp in milliseconds (optional)
     * @returns {Array} Corrected state
     */
    correct(measurement, timestamp) {
      const currentTime = timestamp || performance.now();
      
      // Check for rapid vertical movement
      this.detectRapidVerticalMovement(measurement);
      
      // Linearize measurement model at current state
      const H = this.measurementJacobian(this.state);
      const Ht = this.matrixTranspose(H);
      
      // Calculate innovation (measurement residual)
      const predictedMeasurement = this.measurementFn(this.state);
      const innovation = measurement.map((value, index) => 
        value - predictedMeasurement[index]);
      
      // Innovation covariance S = H * P * H' + R
      const HP = this.matrixMultiply(H, this.errorCovariance);
      const HPHt = this.matrixMultiply(HP, Ht);
      const S = this.matrixAdd(HPHt, this.measurementNoise);
      
      // Optimal Kalman gain K = P * H' * S^(-1)
      const PHt = this.matrixMultiply(this.errorCovariance, Ht);
      const Sinv = this.matrixInverse(S);
      const K = this.matrixMultiply(PHt, Sinv);
      
      // Convert innovation to column vector for matrix operations
      const innovationColumn = innovation.map(value => [value]);
      
      // Update state estimate x = x + K * (z - h(x))
      const KInnovation = this.matrixMultiply(K, innovationColumn);
      
      // Add correction to state
      for (let i = 0; i < this.stateSize; i++) {
        this.state[i] += KInnovation[i][0];
      }
      
      // Update error covariance P = (I - K * H) * P
      const I = this.createIdentityMatrix(this.stateSize);
      const KH = this.matrixMultiply(K, H);
      const IminusKH = this.matrixSubtract(I, KH);
      this.errorCovariance = this.matrixMultiply(IminusKH, this.errorCovariance);
      
      // Ensure covariance stays symmetric and positive definite
      this.enforceSymmetricMatrix(this.errorCovariance);
      
      // Update timing
      this.lastUpdateTime = currentTime;
      
      // Add to position history
      this.updateHistory();
      
      // Store debug data
      if (this.debug) {
        this.debugData.correctedState = [...this.state];
        this.debugData.innovation = [...innovation];
        this.debugData.innovationCovariance = S;
        this.debugData.kalmanGain = K;
      }
      
      // Return the corrected state
      return [...this.state];
    }
    
    /**
     * Enforce that a matrix is symmetric by averaging with its transpose
     * @param {Array} matrix - Matrix to make symmetric
     */
    enforceSymmetricMatrix(matrix) {
      const n = matrix.length;
      
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const avg = (matrix[i][j] + matrix[j][i]) / 2;
          matrix[i][j] = avg;
          matrix[j][i] = avg;
        }
      }
    }
    
    /**
     * Detect rapid vertical (Y-axis) movement, especially direction changes
     * @param {Array} measurement - Current position measurement [x, y, z]
     */
    detectRapidVerticalMovement(measurement) {
      const currentY = measurement[1];
      
      // Skip if we don't have history yet
      if (this.pastPositionY === null) {
        this.pastPositionY = currentY;
        return;
      }
      
      // Check for velocity direction change
      if (this.state[4] !== 0 && this.velocityHistory.length > 1) {
        const currentVelocityY = this.state[4];
        const previousVelocityY = this.velocityHistory[this.velocityHistory.length - 1];
        
        // Direction changed from up to down
        if (previousVelocityY > 0 && currentVelocityY < 0) {
          this.detectedRapidVerticalMovement = true;
          
          // We're at the peak of movement, about to fall
          // Adjust process noise to be more responsive in vertical direction
          this.processNoise[4][4] = 1.0; // Increase process noise for vertical velocity
          this.processNoise[7][7] = 2.0; // Increase process noise for vertical acceleration
          
          if (this.debug) {
            console.log("EKF: Detected direction change from rising to falling!");
          }
        } else {
          // Gradually return to normal
          this.detectedRapidVerticalMovement = false;
          
          // Reset process noise to default gradually
          this.processNoise[4][4] = Math.max(0.1, this.processNoise[4][4] * 0.95);
          this.processNoise[7][7] = Math.max(1.0, this.processNoise[7][7] * 0.95);
        }
      }
      
      // Update past position
      this.pastPositionY = currentY;
    }
    
    /**
     * Update position and velocity history
     */
    updateHistory() {
      // Add current position to history
      this.positionHistory.push([
        this.state[0], // x
        this.state[1], // y
        this.state[2]  // z
      ]);
      
      // Add current velocity to history
      this.velocityHistory.push(this.state[4]); // y velocity is most important for tracking
      
      // Limit history size
      if (this.positionHistory.length > this.historyMaxLength) {
        this.positionHistory.shift();
      }
      
      if (this.velocityHistory.length > this.historyMaxLength) {
        this.velocityHistory.shift();
      }
    }
    
    /**
     * Get the current state estimate
     * @returns {Array} Current state estimate
     */
    getState() {
      return [...this.state];
    }
    
    /**
     * Get the position part of the state
     * @returns {Array} Position [x, y, z]
     */
    getPosition() {
      return [this.state[0], this.state[1], this.state[2]];
    }
    
    /**
     * Get the velocity part of the state
     * @returns {Array} Velocity [vx, vy, vz]
     */
    getVelocity() {
      return [this.state[3], this.state[4], this.state[5]];
    }
    
    /**
     * Get the acceleration part of the state
     * @returns {Array} Acceleration [ax, ay, az]
     */
    getAcceleration() {
      return [this.state[6], this.state[7], this.state[8]];
    }
    
    /**
     * Get debug information
     * @returns {Object} Debug data
     */
    getDebugInfo() {
      return this.debugData;
    }
    
    /**
     * Reset the filter to initial state
     * @param {Array} initialState - Optional new initial state
     */
    reset(initialState = null) {
      if (initialState) {
        this.state = [...initialState];
      } else {
        this.state = new Array(this.stateSize).fill(0);
      }
      
      this.errorCovariance = this.createIdentityMatrix(this.stateSize);
      this.lastUpdateTime = 0;
      this.initialized = false;
      this.detectedRapidVerticalMovement = false;
      this.pastPositionY = null;
      this.positionHistory = [];
      this.velocityHistory = [];
      
      this.debugData = {
        predictedState: null,
        correctedState: null,
        innovation: null,
        innovationCovariance: null,
        kalmanGain: null
      };
    }
    
    /**
     * Check if rapid vertical movement is currently detected
     * @returns {Boolean} True if rapid vertical movement detected
     */
    isRapidVerticalMovementDetected() {
      return this.detectedRapidVerticalMovement;
    }
    
    /**
     * Tuning - set new process noise values
     * @param {Array} newProcessNoise - New process noise covariance matrix
     */
    setProcessNoise(newProcessNoise) {
      this.processNoise = newProcessNoise;
    }
    
    /**
     * Tuning - set new measurement noise values
     * @param {Array} newMeasurementNoise - New measurement noise covariance matrix
     */
    setMeasurementNoise(newMeasurementNoise) {
      this.measurementNoise = newMeasurementNoise;
    }
    
    /**
     * Enable or disable debug mode
     * @param {Boolean} enabled - Whether to enable debug mode
     */
    setDebug(enabled) {
      this.debug = enabled;
    }
    
    /**
     * Create optimized EKF specifically for crank mechanism
     * @param {Number} crankRadius - Radius of the crank in world units
     * @param {Array} crankCenter - Center of rotation [x, y, z]
     * @param {String} crankPlane - Plane of rotation ('xy', 'yz', or 'xz')
     * @returns {ExtendedKalmanFilter} Configured EKF instance
     */
    static createCrankEKF(crankRadius, crankCenter = [0, 0, 0], crankPlane = 'yz') {
      return new ExtendedKalmanFilter({
        stateSize: 9,
        measurementSize: 3,
        useGravityModel: true,
        useCrankModel: true,
        crankRadius: crankRadius,
        crankCenter: crankCenter,
        crankPlane: crankPlane,
        directionChangeThreshold: 5, // More sensitive for crank mechanism
        initialState: [
          ...crankCenter, // Initial position at crank center
          0, 0, 0,        // Initial velocity
          0, 0, 0         // Initial acceleration
        ]
      });
    }
  }
  
  /**
   * Integration with AR marker tracking system - replace updateScenes function
   * @param {Array} markers - Detected markers
   * @param {ExtendedKalmanFilter} ekf - EKF instance
   */
  function updateScenesWithEKF(markers, ekf) {
    if (markers.length > 0) {
      // Marker detected - update status
      var modelStatus = document.getElementById('modelStatus');
      modelStatus.textContent = "Model: Marker Detected";
      modelStatus.style.color = "#44FF44";
      
      var corners = markers[0].corners;
      var isTracked = markers[0].id === -1;
      
      // Copy corners and convert to camera coordinate system
      var cameraCorners = [];
      for (var i = 0; i < corners.length; i++) {
        cameraCorners.push({
          x: corners[i].x - (canvas.width / 2),
          y: (canvas.height / 2) - corners[i].y
        });
      }
      
      // Estimate pose
      var pose = posit.pose(cameraCorners);
      
      // Get raw position from pose
      var rawPosition = pose.bestTranslation;
      
      // Update EKF with the measurement
      ekf.correct(rawPosition);
      
      // Predict next state
      ekf.predict();
      
      // Get position and rotation
      var ekfPosition = ekf.getPosition();
      var rotationAngles = extractEulerAngles(pose.bestRotation);
      
      // Check if we're in a rapid vertical movement situation
      if (ekf.isRapidVerticalMovementDetected()) {
        modelStatus.textContent = "Model: Rapid Movement";
        modelStatus.style.color = "#FFAA00";
      }
      
      // Update model with EKF-filtered position and raw rotation
      updateObjectWithEuler(model, rotationAngles, ekfPosition);
      
      // Make model visible if it's not already
      if (!model.visible && modelVisible) {
        model.visible = true;
      }
      
      lastMarkerDetected = true;
      step += 0.025;
      model.rotation.z -= step;
    } else {
      // No marker detected - update status
      var modelStatus = document.getElementById('modelStatus');
      modelStatus.textContent = "Model: No Marker Found";
      modelStatus.style.color = "#FF4444";
      
      // Continue predicting position for a short time when marker is lost
      if (lastMarkerDetected) {
        ekf.predict();
        
        // Use predicted position for a few frames after losing marker
        var predictedPosition = ekf.getPosition();
        
        // Only keep updating if the prediction seems reasonable (not diverging)
        if (predictedPosition[1] > -100 && predictedPosition[1] < 100) {
          updateObjectWithEuler(model, [0, 0, 0], predictedPosition);
        } else {
          // Reset EKF if prediction is diverging
          ekf.reset();
        }
      }
      
      // Hide model if needed but respect the modelVisible toggle
      if (model.visible && !modelVisible) {
        model.visible = false;
      }
      
      lastMarkerDetected = false;
    }
  }
  
  /**
   * Initialize EKF system and integrate with AR tracking
   */
  function initEKF() {
    // Create EKF instance
    const ekf = new ExtendedKalmanFilter({
      stateSize: 9,
      measurementSize: 3,
      useGravityModel: true,
      directionChangeThreshold: 10,
      debug: false
    });
    
    // Replace tick function
    const originalTick = window.tick;
    
    window.tick = function() {
      requestAnimationFrame(window.tick);
      
      if (video.readyState === video.HAVE_ENOUGH_DATA) {
        snapshot();
  
        // Process image and detect markers
        var originalImageData = context.getImageData(0, 0, canvas.width, canvas.height);
        var processedImageData = originalImageData;
        
        if (useAdaptiveFilter) {
          processedImageData = adaptiveFilter.process(originalImageData);
        }
        
        if (useBilateralFilter) {
          processedImageData = bilateralFilter.process(processedImageData);
        }
        
        var detectedMarkers = [];
        if (useMultiScale) {
          detectedMarkers = multiScaleDetector.detect(processedImageData);
        } else {
          detectedMarkers = detector.detect(processedImageData);
        }
        
        var markers = detectedMarkers;
        if (useOpticalFlow) {
          markers = opticalFlowTracker.track(originalImageData, detectedMarkers);
        }
        
        // Draw marker outlines
        drawCorners(markers);
        
        // Update scenes using EKF
        updateScenesWithEKF(markers, ekf);
        
        // Render views
        render();
        
        // Optional: Draw EKF debug visualization
        if (ekf.debug) {
          drawEKFDebug(ekf);
        }
      }
    };
    
    console.log("Extended Kalman Filter initialized for enhanced tracking");
    
    return ekf;
  }
  
  /**
   * Draw debug visualization for EKF
   * @param {ExtendedKalmanFilter} ekf - EKF instance
   */
  function drawEKFDebug(ekf) {
    const debugInfo = ekf.getDebugInfo();
    const ctx = context;
    const width = canvas.width;
    const height = canvas.height;
    
    // Draw state information
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '12px monospace';
    
    if (debugInfo.correctedState) {
      const pos = debugInfo.correctedState.slice(0, 3);
      const vel = debugInfo.correctedState.slice(3, 6);
      const acc = debugInfo.correctedState.slice(6, 9);
      
      ctx.fillText(`Position: ${pos.map(v => v.toFixed(2)).join(', ')}`, 10, height - 70);
      ctx.fillText(`Velocity: ${vel.map(v => v.toFixed(2)).join(', ')}`, 10, height - 55);
      ctx.fillText(`Accel: ${acc.map(v => v.toFixed(2)).join(', ')}`, 10, height - 40);
    }
    
    // Draw special indicator when vertical movement is detected
    if (ekf.isRapidVerticalMovementDetected()) {
      ctx.fillStyle = '#FFAA00';
      ctx.fillText('RAPID VERTICAL MOVEMENT DETECTED', 10, height - 20);
      
      // Draw direction indicator
      const centerX = width / 2;
      const centerY = height / 2;
      
      ctx.strokeStyle = '#FFAA00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX, centerY + 50);
      ctx.stroke();
      
      // Arrow head
      ctx.beginPath();
      ctx.moveTo(centerX - 10, centerY + 40);
      ctx.lineTo(centerX, centerY + 50);
      ctx.lineTo(centerX + 10, centerY + 40);
      ctx.stroke();
    }
  }
  
  // Export the EKF class for use in other modules
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
      ExtendedKalmanFilter,
      updateScenesWithEKF,
      initEKF
    };
  } else {
    // Browser global
    window.ExtendedKalmanFilter = ExtendedKalmanFilter;
    window.updateScenesWithEKF = updateScenesWithEKF;
    window.initEKF = initEKF;
  }