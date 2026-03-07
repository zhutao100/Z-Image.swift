import Foundation
import MLX
import MLXNN

public protocol VAEImageEncoding {
  var dtype: DType { get }
  func encode(_ images: MLXArray) -> MLXArray
}

extension AutoencoderKL: VAEImageEncoding {}

/// Encoder-only variant of the VAE for control/inpaint conditioning.
/// It builds only the encoder subgraph and exposes the same `encode` API.
public final class AutoencoderEncoderOnly: Module, VAEImageEncoding {
  public let configuration: VAEConfig
  @ModuleInfo(key: "encoder") private var encoder: VAEEncoder

  public init(configuration: VAEConfig) {
    self.configuration = configuration
    self._encoder.wrappedValue = VAEEncoder(config: configuration)
    super.init()
  }

  public var dtype: DType {
    encoder.convIn.weight.dtype
  }

  public func encode(_ images: MLXArray) -> MLXArray {
    var hidden = images.transposed(0, 2, 3, 1)
    hidden = encoder(hidden)
    hidden = hidden.transposed(0, 3, 1, 2)
    return hidden
  }
}
