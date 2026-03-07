import XCTest

@testable import ZImage

final class ControlNetSharingTests: XCTestCase {
  func testControlnetSharesBaseTransformerModules() throws {
    let transformerConfig = try makeTransformerConfig()
    let transformer = ZImageTransformer2DModel(configuration: transformerConfig)
    let controlnet = ZImageControlNetModel(
      configuration: makeControlnetConfig(from: transformerConfig),
      sharedTransformer: transformer
    )

    XCTAssertTrue(controlnet.tEmbedder === transformer.tEmbedder)
    XCTAssertTrue(controlnet.capEmbedNorm === transformer.capEmbedNorm)
    XCTAssertTrue(controlnet.capEmbedLinear === transformer.capEmbedLinear)
    XCTAssertTrue(controlnet.noiseRefiner[0] === transformer.noiseRefiner[0])
    XCTAssertTrue(controlnet.contextRefiner[0] === transformer.contextRefiner[0])
    XCTAssertTrue(controlnet.allXEmbedder["2-1"] === transformer.allXEmbedder["2-1"])
  }

  private func makeTransformerConfig() throws -> ZImageTransformerConfig {
    let data = Data(
      """
      {
        "in_channels": 16,
        "dim": 64,
        "n_layers": 2,
        "n_refiner_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 4,
        "norm_eps": 0.00001,
        "qk_norm": true,
        "cap_feat_dim": 32,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [8, 8, 8],
        "axes_lens": [64, 64, 64]
      }
      """.utf8
    )
    return try JSONDecoder().decode(ZImageTransformerConfig.self, from: data)
  }

  private func makeControlnetConfig(from transformerConfig: ZImageTransformerConfig) -> ZImageControlNetConfig {
    ZImageControlNetConfig(
      inChannels: transformerConfig.inChannels,
      dim: transformerConfig.dim,
      nLayers: transformerConfig.nLayers,
      nRefinerLayers: transformerConfig.nRefinerLayers,
      nHeads: transformerConfig.nHeads,
      nKVHeads: transformerConfig.nKVHeads,
      normEps: transformerConfig.normEps,
      qkNorm: transformerConfig.qkNorm,
      capFeatDim: transformerConfig.capFeatDim,
      ropeTheta: transformerConfig.ropeTheta,
      tScale: transformerConfig.tScale,
      axesDims: transformerConfig.axesDims,
      axesLens: transformerConfig.axesLens,
      controlLayersPlaces: [0, 1],
      controlRefinerLayersPlaces: [0],
      controlInDim: 33,
      addControlNoiseRefiner: true
    )
  }
}
