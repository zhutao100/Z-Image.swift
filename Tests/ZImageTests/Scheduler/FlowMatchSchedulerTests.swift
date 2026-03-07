import MLX
import XCTest

@testable import ZImage

final class FlowMatchSchedulerTests: XCTestCase {

  // MARK: - Initialization Tests

  func testSchedulerInitializationWithDefaults() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.timesteps.dim(0), 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)  // steps + 1 (includes final 0)
  }

  func testSchedulerInitializationWithCustomSteps() {
    let config = Self.makeConfig()

    for steps in [4, 9, 20, 50] {
      let scheduler = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config)
      XCTAssertEqual(scheduler.numInferenceSteps, steps)
      XCTAssertEqual(scheduler.timesteps.dim(0), steps)
      XCTAssertEqual(scheduler.sigmas.dim(0), steps + 1)
    }
  }

  // MARK: - Timestep Generation Tests

  func testTimestepGeneration() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)
    let timesteps = scheduler.timesteps.asArray(Float.self)

    // First timestep should be close to 1000 (max)
    XCTAssertGreaterThan(timesteps[0], 900)

    // Last timestep should be close to 0 (min, but not exactly 0)
    XCTAssertLessThan(timesteps.last!, 200)

    // Timesteps should be monotonically decreasing
    for i in 1..<timesteps.count {
      XCTAssertLessThan(timesteps[i], timesteps[i - 1], "Timesteps should decrease monotonically")
    }
  }

  // MARK: - Sigma Calculation Tests

  func testSigmaValues() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    // First sigma should be close to 1.0 (max)
    XCTAssertGreaterThan(sigmas[0], 0.9)

    // Last sigma should be exactly 0.0
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-6)

    // Sigmas should be monotonically decreasing
    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(sigmas[i], sigmas[i - 1], "Sigmas should decrease monotonically")
    }

    // All sigmas should be in [0, 1] range
    for sigma in sigmas {
      XCTAssertGreaterThanOrEqual(sigma, 0.0)
      XCTAssertLessThanOrEqual(sigma, 1.0)
    }
  }

  // MARK: - Dynamic Shifting Tests

  func testDynamicShiftingEnabled() {
    let config = Self.makeConfig(
      useDynamicShifting: true,
      baseShift: 0.5,
      maxShift: 1.15,
      baseImageSeqLen: 256,
      maxImageSeqLen: 4096
    )

    // Calculate mu for a typical image sequence length
    let imageSeqLen = 1024  // 32x32 latent
    let mu = calculateTestMu(
      imageSeqLen: imageSeqLen,
      baseSeqLen: 256,
      maxSeqLen: 4096,
      baseShift: 0.5,
      maxShift: 1.15
    )

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config, mu: mu)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    // With dynamic shifting, sigmas should still be valid
    for sigma in sigmas {
      XCTAssertGreaterThanOrEqual(sigma, 0.0)
      XCTAssertLessThanOrEqual(sigma, 1.0)
    }
  }

  func testDynamicShiftingDisabled() {
    let config = Self.makeConfig(shift: 3.0)  // Non-unity shift

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    // Shifted sigmas should be different from unshifted
    // The shift formula: sigma_shifted = shift * sigma / (1 + (shift - 1) * sigma)
    XCTAssertGreaterThan(sigmas[0], 0.5)  // Should still be reasonably high
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-6)
  }

  // MARK: - Scheduler Step Tests

  func testSchedulerStep() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)

    // Create sample tensors using explicit Float arrays to avoid float64
    // Then convert to bfloat16 to match model inference
    let sampleValues: [Float] = [1.0, 2.0, 3.0, 4.0]
    let modelOutputValues: [Float] = [0.1, 0.2, 0.3, 0.4]
    let sample = MLXArray(sampleValues, [1, 1, 2, 2]).asType(.bfloat16)
    let modelOutput = MLXArray(modelOutputValues, [1, 1, 2, 2]).asType(.bfloat16)

    // Perform a step
    let result = scheduler.step(modelOutput: modelOutput, timestepIndex: 0, sample: sample)

    // Result should have same shape as input
    XCTAssertEqual(result.shape, sample.shape)

    // Result should be different from input (unless dt is 0)
    // Convert to float32 before extracting values
    let resultF32 = result.asType(.float32)
    let sampleF32 = sample.asType(.float32)
    MLX.eval(resultF32, sampleF32)
    let resultData = resultF32.asArray(Float.self)
    let sampleData = sampleF32.asArray(Float.self)

    // The step formula is: sample + modelOutput * dt
    // Since dt = sigma[i+1] - sigma[i] (which is negative for decreasing sigmas)
    // The result should be different from the sample
    var allSame = true
    for i in 0..<resultData.count {
      if abs(resultData[i] - sampleData[i]) > 1e-6 {
        allSame = false
        break
      }
    }
    XCTAssertFalse(allSame, "Scheduler step should modify the sample")
  }

  func testSchedulerStepAllTimesteps() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config)
    // Use explicit Float array to avoid float64
    let sampleValues: [Float] = [1.0, 2.0, 3.0, 4.0]
    let modelOutputValues: [Float] = [0.1, 0.2, 0.3, 0.4]
    var sample = MLXArray(sampleValues, [1, 1, 2, 2]).asType(.bfloat16)

    // Run through all timesteps
    for i in 0..<scheduler.numInferenceSteps {
      let modelOutput = MLXArray(modelOutputValues, [1, 1, 2, 2]).asType(.bfloat16)
      sample = scheduler.step(modelOutput: modelOutput, timestepIndex: i, sample: sample)
      // Convert to float32 before eval
      let sampleF32 = sample.asType(.float32)
      MLX.eval(sampleF32)
    }

    // Final sample should have same shape
    XCTAssertEqual(sample.shape, [1, 1, 2, 2])
  }

  // MARK: - Edge Cases

  func testSingleStep() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 1, config: config)
    XCTAssertEqual(scheduler.numInferenceSteps, 1)
    XCTAssertEqual(scheduler.timesteps.dim(0), 1)
    XCTAssertEqual(scheduler.sigmas.dim(0), 2)
  }

  func testLargeStepCount() {
    let config = Self.makeConfig()

    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: 100, config: config)
    XCTAssertEqual(scheduler.numInferenceSteps, 100)
    XCTAssertEqual(scheduler.timesteps.dim(0), 100)

    let sigmas = scheduler.sigmas.asArray(Float.self)
    // More steps should give finer granularity
    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(sigmas[i], sigmas[i - 1])
    }
  }

  // MARK: - Helper Functions

  private func calculateTestMu(
    imageSeqLen: Int,
    baseSeqLen: Int,
    maxSeqLen: Int,
    baseShift: Float,
    maxShift: Float
  ) -> Float {
    let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
    let b = baseShift - m * Float(baseSeqLen)
    return Float(imageSeqLen) * m + b
  }
}

// Helper to create test config JSON and decode
extension FlowMatchSchedulerTests {
  static func makeConfig(
    numTrainTimesteps: Int = 1000,
    shift: Float = 1.0,
    useDynamicShifting: Bool = false,
    baseShift: Float? = nil,
    maxShift: Float? = nil,
    baseImageSeqLen: Int? = nil,
    maxImageSeqLen: Int? = nil
  ) -> ZImageSchedulerConfig {
    var json: [String: Any] = [
      "num_train_timesteps": numTrainTimesteps,
      "shift": shift,
      "use_dynamic_shifting": useDynamicShifting,
    ]
    if let baseShift = baseShift { json["base_shift"] = baseShift }
    if let maxShift = maxShift { json["max_shift"] = maxShift }
    if let baseImageSeqLen = baseImageSeqLen { json["base_image_seq_len"] = baseImageSeqLen }
    if let maxImageSeqLen = maxImageSeqLen { json["max_image_seq_len"] = maxImageSeqLen }

    let data = try! JSONSerialization.data(withJSONObject: json)
    return try! JSONDecoder().decode(ZImageSchedulerConfig.self, from: data)
  }
}
