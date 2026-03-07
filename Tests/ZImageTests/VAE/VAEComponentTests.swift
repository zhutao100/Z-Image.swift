import Logging
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class VAEComponentTests: XCTestCase {
  private let logger = Logger(label: "test.vae-components")

  private func makeTestConfig() -> VAEConfig {
    VAEConfig(
      inChannels: 3,
      outChannels: 3,
      latentChannels: 4,
      scalingFactor: 0.3611,
      shiftFactor: 0.1159,
      blockOutChannels: [32, 64],
      layersPerBlock: 1,
      normNumGroups: 8,
      sampleSize: 16,
      midBlockAddAttention: true
    )
  }

  private func weights(from module: Module) -> [String: MLXArray] {
    Dictionary(
      uniqueKeysWithValues: module.parameters().flattened().map { key, tensor in
        let serializedTensor =
          if tensor.ndim == 4 {
            tensor.transposed(0, 3, 1, 2)
          } else {
            tensor
          }
        return (key, serializedTensor)
      })
  }

  func testEncoderOnlyMatchesFullAutoencoderEncode() {
    let config = makeTestConfig()
    let full = AutoencoderKL(configuration: config)
    let encoderOnly = AutoencoderEncoderOnly(configuration: config)

    ZImageWeightsMapping.applyVAE(weights: weights(from: full), to: encoderOnly, logger: logger)

    let image = MLXRandom.normal([1, config.inChannels, config.sampleSize, config.sampleSize])
    let fullEncoded = full.encode(image)
    let encoderOnlyEncoded = encoderOnly.encode(image)
    MLX.eval(fullEncoded, encoderOnlyEncoded)

    XCTAssertEqual(fullEncoded.shape, encoderOnlyEncoded.shape)

    let fullValues = fullEncoded.asArray(Float.self)
    let encoderValues = encoderOnlyEncoded.asArray(Float.self)
    XCTAssertEqual(fullValues.count, encoderValues.count)

    let maxDifference = zip(fullValues, encoderValues).reduce(Float.zero) { current, pair in
      max(current, abs(pair.0 - pair.1))
    }
    XCTAssertLessThan(maxDifference, 1e-6)
  }

  func testDecoderOnlyMatchesFullAutoencoderDecode() {
    let config = makeTestConfig()
    let full = AutoencoderKL(configuration: config)
    let decoderOnly = AutoencoderDecoderOnly(configuration: config)

    ZImageWeightsMapping.applyVAE(weights: weights(from: full), to: decoderOnly, logger: logger)

    let latentSize = config.sampleSize / config.latentDivisor
    let latents = MLXRandom.normal([1, config.latentChannels, latentSize, latentSize])
    let (fullDecoded, _) = full.decode(latents, return_dict: false)
    let (decoderOnlyDecoded, _) = decoderOnly.decode(latents, return_dict: false)
    MLX.eval(fullDecoded, decoderOnlyDecoded)

    XCTAssertEqual(fullDecoded.shape, decoderOnlyDecoded.shape)

    let fullValues = fullDecoded.asArray(Float.self)
    let decoderValues = decoderOnlyDecoded.asArray(Float.self)
    XCTAssertEqual(fullValues.count, decoderValues.count)

    let maxDifference = zip(fullValues, decoderValues).reduce(Float.zero) { current, pair in
      max(current, abs(pair.0 - pair.1))
    }
    XCTAssertLessThan(maxDifference, 1e-6)
  }
}
