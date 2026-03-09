import MLX
import XCTest

@testable import ZImage

final class LoRAApplicatorTests: XCTestCase {

  private func makeTinyTransformer() throws -> ZImageTransformer2DModel {
    let configJSON = """
      {
        "in_channels": 4,
        "dim": 4,
        "n_layers": 1,
        "n_refiner_layers": 0,
        "n_heads": 1,
        "n_kv_heads": 1,
        "norm_eps": 1e-6,
        "qk_norm": false,
        "cap_feat_dim": 4,
        "rope_theta": 10000.0,
        "t_scale": 1.0,
        "axes_dims": [2, 1, 1],
        "axes_lens": [1, 1, 1]
      }
      """
    let config = try JSONDecoder().decode(ZImageTransformerConfig.self, from: Data(configJSON.utf8))
    return ZImageTransformer2DModel(configuration: config)
  }

  func testFullMatrixLoKrIgnoresRawAlpha() throws {
    let transformer = try makeTinyTransformer()
    let targetKey = "layers.0.attention.to_q"

    let w1 = MLXArray([Float(1.0), 2.0, 3.0, 4.0], [2, 2]).asType(.float32)
    let w2 = MLXArray([Float(0.5), 1.0, 1.5, 2.0], [2, 2]).asType(.float32)
    let adapter = LoRAWeights(
      weights: [:],
      lokrWeights: [targetKey: LoKrWeights(w1: w1, w2: w2, alpha: 9_999_220_736)],
      rank: 16
    )

    let beforeWeight = transformer.layers[0].attention.toQ.weight.asType(.float32)
    let before = MLXArray(beforeWeight.asArray(Float.self), beforeWeight.shape)
    LoRAApplicator.applyLoKr(to: transformer, loraWeights: adapter, scale: 1.0)
    let after = transformer.layers[0].attention.toQ.weight.asType(.float32)

    let delta = after - before
    let expected = kron(w1, w2).asType(.float32)
    let maxDiff = abs(delta - expected).max().item(Float.self)

    XCTAssertLessThanOrEqual(maxDiff, 1e-5)
  }

  func testValidateTensorStabilityRejectsNonFiniteValues() {
    let unstable = MLXArray([Float.infinity, 0.0], [2]).asType(.float32)

    XCTAssertThrowsError(
      try PipelineUtilities.validateTensorStability(unstable, name: "unstable tensor")
    ) { error in
      XCTAssertTrue(error is PipelineUtilities.StabilityError)
    }
  }

  func testValidateDisplayImageRangeRejectsFullyClippedImage() {
    let image = MLXArray([Float(-3.0), -2.5, -1.25], [1, 1, 1, 3]).asType(.float32)

    XCTAssertThrowsError(
      try PipelineUtilities.validateDisplayImageRange(image, name: "decoded image")
    ) { error in
      XCTAssertTrue(error is PipelineUtilities.StabilityError)
    }
  }
}
