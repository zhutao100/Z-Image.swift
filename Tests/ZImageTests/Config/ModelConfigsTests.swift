import XCTest

@testable import ZImage

final class ModelConfigsTests: XCTestCase {

  // MARK: - Transformer Config Tests

  func testTransformerConfigDecoding() throws {
    let json = """
      {
        "in_channels": 16,
        "dim": 3840,
        "n_layers": 30,
        "n_refiner_layers": 2,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-6,
        "qk_norm": true,
        "cap_feat_dim": 2560,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [32, 48, 48],
        "axes_lens": [1, 128, 128]
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageTransformerConfig.self, from: data)

    XCTAssertEqual(config.inChannels, 16)
    XCTAssertEqual(config.dim, 3840)
    XCTAssertEqual(config.nLayers, 30)
    XCTAssertEqual(config.nRefinerLayers, 2)
    XCTAssertEqual(config.nHeads, 30)
    XCTAssertEqual(config.nKVHeads, 30)
    XCTAssertEqual(config.normEps, 1e-6, accuracy: 1e-10)
    XCTAssertTrue(config.qkNorm)
    XCTAssertEqual(config.capFeatDim, 2560)
    XCTAssertEqual(config.ropeTheta, 256.0)
    XCTAssertEqual(config.tScale, 1000.0)
    XCTAssertEqual(config.axesDims, [32, 48, 48])
    XCTAssertEqual(config.axesLens, [1, 128, 128])
  }

  // MARK: - VAE Config Tests

  func testVAEConfigDecoding() throws {
    let json = """
      {
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 1024,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "mid_block_add_attention": true,
        "use_post_quant_conv": false,
        "use_quant_conv": false
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageVAEConfig.self, from: data)

    XCTAssertEqual(config.blockOutChannels, [128, 256, 512, 512])
    XCTAssertEqual(config.latentChannels, 16)
    XCTAssertEqual(config.scalingFactor, 0.3611, accuracy: 1e-4)
    XCTAssertEqual(config.shiftFactor, 0.1159, accuracy: 1e-4)
    XCTAssertEqual(config.sampleSize, 1024)
    XCTAssertEqual(config.inChannels, 3)
    XCTAssertEqual(config.outChannels, 3)
    XCTAssertEqual(config.layersPerBlock, 2)
    XCTAssertEqual(config.normNumGroups, 32)
    XCTAssertTrue(config.midBlockAddAttention)
    XCTAssertEqual(config.usePostQuantConv, false)
    XCTAssertEqual(config.useQuantConv, false)
  }

  func testVAEConfigScaleFactor() throws {
    let json = """
      {
        "block_out_channels": [128, 256, 512, 512],
        "latent_channels": 16,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 1024,
        "in_channels": 3,
        "out_channels": 3,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "mid_block_add_attention": true
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageVAEConfig.self, from: data)

    // 4 blocks = 2^3 = 8 scale factor
    XCTAssertEqual(config.vaeScaleFactor, 8)
    XCTAssertEqual(config.latentDivisor, 8)
  }

  func testVAEConfigScaleFactorVariousBlockCounts() {
    // Test with different block counts
    struct TestCase {
      let blockOutChannels: [Int]
      let expectedScale: Int
    }

    let testCases = [
      TestCase(blockOutChannels: [128], expectedScale: 1),
      TestCase(blockOutChannels: [128, 256], expectedScale: 2),
      TestCase(blockOutChannels: [128, 256, 512], expectedScale: 4),
      TestCase(blockOutChannels: [128, 256, 512, 512], expectedScale: 8),
      TestCase(blockOutChannels: [128, 256, 512, 512, 512], expectedScale: 16),
    ]

    for testCase in testCases {
      let json = """
        {
          "block_out_channels": \(testCase.blockOutChannels),
          "latent_channels": 16,
          "scaling_factor": 0.3611,
          "shift_factor": 0.1159,
          "sample_size": 1024,
          "in_channels": 3,
          "out_channels": 3,
          "layers_per_block": 2,
          "norm_num_groups": 32,
          "mid_block_add_attention": true
        }
        """

      do {
        let data = json.data(using: .utf8)!
        let config = try JSONDecoder().decode(ZImageVAEConfig.self, from: data)
        XCTAssertEqual(config.vaeScaleFactor, testCase.expectedScale, "Failed for blocks: \(testCase.blockOutChannels)")
      } catch {
        XCTFail("Failed to decode config: \(error)")
      }
    }
  }

  // MARK: - Scheduler Config Tests

  func testSchedulerConfigDecoding() throws {
    let json = """
      {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": true,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageSchedulerConfig.self, from: data)

    XCTAssertEqual(config.numTrainTimesteps, 1000)
    XCTAssertEqual(config.shift, 1.0)
    XCTAssertTrue(config.useDynamicShifting)
    XCTAssertEqual(config.baseShift, 0.5)
    XCTAssertEqual(config.maxShift, 1.15)
    XCTAssertEqual(config.baseImageSeqLen, 256)
    XCTAssertEqual(config.maxImageSeqLen, 4096)
  }

  func testSchedulerConfigMinimal() throws {
    let json = """
      {
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "use_dynamic_shifting": false
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageSchedulerConfig.self, from: data)

    XCTAssertEqual(config.numTrainTimesteps, 1000)
    XCTAssertEqual(config.shift, 3.0)
    XCTAssertFalse(config.useDynamicShifting)
    XCTAssertNil(config.baseShift)
    XCTAssertNil(config.maxShift)
    XCTAssertNil(config.baseImageSeqLen)
    XCTAssertNil(config.maxImageSeqLen)
  }

  // MARK: - Text Encoder Config Tests

  func testTextEncoderConfigDecoding() throws {
    let json = """
      {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000.0,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "head_dim": 128
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageTextEncoderConfig.self, from: data)

    XCTAssertEqual(config.hiddenSize, 2560)
    XCTAssertEqual(config.numHiddenLayers, 36)
    XCTAssertEqual(config.numAttentionHeads, 32)
    XCTAssertEqual(config.numKeyValueHeads, 8)
    XCTAssertEqual(config.intermediateSize, 9728)
    XCTAssertEqual(config.maxPositionEmbeddings, 40960)
    XCTAssertEqual(config.ropeTheta, 1000000.0)
    XCTAssertEqual(config.vocabSize, 151936)
    XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-10)
    XCTAssertEqual(config.headDim, 128)
  }

  // MARK: - Invalid Config Tests

  func testInvalidTransformerConfig() {
    let json = """
      {
        "invalid_field": 123
      }
      """

    let data = json.data(using: .utf8)!
    XCTAssertThrowsError(try JSONDecoder().decode(ZImageTransformerConfig.self, from: data))
  }

  func testInvalidVAEConfig() {
    let json = """
      {
        "block_out_channels": "not_an_array"
      }
      """

    let data = json.data(using: .utf8)!
    XCTAssertThrowsError(try JSONDecoder().decode(ZImageVAEConfig.self, from: data))
  }

  func testMalformedJSON() {
    let json = """
      { invalid json }
      """

    let data = json.data(using: .utf8)!
    XCTAssertThrowsError(try JSONDecoder().decode(ZImageSchedulerConfig.self, from: data))
  }

  // MARK: - Numeric Precision Tests

  func testFloatPrecision() throws {
    let json = """
      {
        "num_train_timesteps": 1000,
        "shift": 1.0000001,
        "use_dynamic_shifting": false
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageSchedulerConfig.self, from: data)

    // Should preserve precision
    XCTAssertEqual(config.shift, 1.0000001, accuracy: 1e-7)
  }

  func testLargeIntegerValues() throws {
    let json = """
      {
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "max_position_embeddings": 131072,
        "rope_theta": 1000000.0,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "head_dim": 128
      }
      """

    let data = json.data(using: .utf8)!
    let config = try JSONDecoder().decode(ZImageTextEncoderConfig.self, from: data)

    XCTAssertEqual(config.maxPositionEmbeddings, 131072)
    XCTAssertEqual(config.vocabSize, 151936)
  }
}
