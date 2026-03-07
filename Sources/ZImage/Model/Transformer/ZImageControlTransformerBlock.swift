import Foundation
import MLX
import MLXNN

/// Control transformer block that generates hints for ControlNet
/// This block processes control signals and outputs accumulated hints
public final class ZImageControlTransformerBlock: Module {
  let dim: Int
  let nHeads: Int
  let nKvHeads: Int
  let blockId: Int

  @ModuleInfo(key: "before_proj") var beforeProj: Linear?
  @ModuleInfo(key: "after_proj") var afterProj: Linear

  @ModuleInfo(key: "attention") var attention: ZImageSelfAttention
  @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]?
  @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
  @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
  @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
  @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
  @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward

  public init(
    blockId: Int,
    dim: Int,
    nHeads: Int,
    nKvHeads: Int,
    normEps: Float,
    qkNorm: Bool
  ) {
    self.blockId = blockId
    self.dim = dim
    self.nHeads = nHeads
    self.nKvHeads = nKvHeads

    // before_proj only exists at blockId == 0
    if blockId == 0 {
      self._beforeProj.wrappedValue = Linear(dim, dim, bias: true)
    }
    self._afterProj.wrappedValue = Linear(dim, dim, bias: true)

    self._attention.wrappedValue = ZImageSelfAttention(dim: dim, heads: nHeads, normEps: normEps, qkNorm: qkNorm)
    self._adaLN.wrappedValue = [Linear(min(dim, 256), 4 * dim, bias: true)]
    self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)

    let hiddenDim = Int(Float(dim) / 3.0 * 8.0)
    self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: hiddenDim)

    super.init()
  }

  /// Zero-initialize the projection layers (called after weight loading)
  public func zeroInitializeProjections() {
    if let beforeProj = beforeProj {
      beforeProj.weight._updateInternal(MLX.zeros(like: beforeProj.weight))
      if let bias = beforeProj.bias {
        bias._updateInternal(MLX.zeros(like: bias))
      }
    }
    afterProj.weight._updateInternal(MLX.zeros(like: afterProj.weight))
    if let bias = afterProj.bias {
      bias._updateInternal(MLX.zeros(like: bias))
    }
  }

  private func transformerForward(
    _ x: MLXArray,
    attnMask: MLXArray?,
    freqsCis: MLXArray?,
    adalnInput: MLXArray?
  ) -> MLXArray {
    var out = x

    if let c = adalnInput, let adaLNModule = adaLN {
      let mod = adaLNModule[0](c)
      let chunkSize = dim

      let attnScale = (1 + mod[0..., 0..<chunkSize])[.ellipsis, .newAxis, 0...]
      let attnGate = MLX.tanh(mod[0..., chunkSize..<(2 * chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpScale = (1 + mod[0..., (2 * chunkSize)..<(3 * chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpGate = MLX.tanh(mod[0..., (3 * chunkSize)..<(4 * chunkSize)])[.ellipsis, .newAxis, 0...]

      let attnOut = attention(attentionNorm1(out) * attnScale, attnMask: attnMask, freqsCis: freqsCis)
      out = out + attnGate * attentionNorm2(attnOut)

      let ffnOut = feedForward(ffnNorm1(out) * mlpScale)
      out = out + mlpGate * ffnNorm2(ffnOut)
    } else {
      let attnOut = attention(attentionNorm1(out), attnMask: attnMask, freqsCis: freqsCis)
      out = out + attentionNorm2(attnOut)

      let ffnOut = feedForward(ffnNorm1(out))
      out = out + ffnNorm2(ffnOut)
    }

    return out
  }

  /// Forward pass for control block
  /// - Parameters:
  ///   - c: Control context (stacked hints if blockId > 0, or initial control if blockId == 0)
  ///   - x: Main transformer hidden states (used only at blockId == 0)
  ///   - attnMask: Attention mask
  ///   - freqsCis: Rotary position embeddings
  ///   - adalnInput: Timestep embedding for adaptive layer norm
  /// - Returns: Stacked hints array [num_hints, batch, seq, dim]
  func callAsFunction(
    _ c: MLXArray,
    x: MLXArray,
    attnMask: MLXArray?,
    freqsCis: MLXArray?,
    adalnInput: MLXArray?
  ) -> MLXArray {
    var control: MLXArray
    var allC: [MLXArray]

    if blockId == 0 {
      // First block: add projected control to main hidden states
      guard let beforeProj = beforeProj else {
        fatalError("beforeProj must be initialized for blockId == 0")
      }
      control = beforeProj(c) + x
      allC = []
    } else {
      // Subsequent blocks: unbind stacked hints, take last as control
      let numHints = c.dim(0)
      allC = (0..<(numHints - 1)).map { c[$0] }
      control = c[numHints - 1]
    }

    // Run through transformer block
    control = transformerForward(control, attnMask: attnMask, freqsCis: freqsCis, adalnInput: adalnInput)

    // Project for skip connection
    let cSkip = afterProj(control)

    // Accumulate hints
    allC.append(cSkip)
    allC.append(control)

    return MLX.stacked(allC, axis: 0)
  }
}

/// Base transformer block that can apply control hints
/// Used as the main layers in the control transformer model
public final class BaseZImageTransformerBlock: Module {
  let dim: Int
  let nHeads: Int
  let nKvHeads: Int
  let modulation: Bool
  let blockId: Int?  // nil means no control at this layer

  @ModuleInfo(key: "attention") var attention: ZImageSelfAttention
  @ModuleInfo(key: "adaLN_modulation") var adaLN: [Linear]?
  @ModuleInfo(key: "attention_norm1") var attentionNorm1: RMSNorm
  @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
  @ModuleInfo(key: "attention_norm2") var attentionNorm2: RMSNorm
  @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
  @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward

  public init(
    layerId: Int,
    dim: Int,
    nHeads: Int,
    nKvHeads: Int,
    normEps: Float,
    qkNorm: Bool,
    modulation: Bool,
    blockId: Int?
  ) {
    self.dim = dim
    self.nHeads = nHeads
    self.nKvHeads = nKvHeads
    self.modulation = modulation
    self.blockId = blockId

    self._attention.wrappedValue = ZImageSelfAttention(dim: dim, heads: nHeads, normEps: normEps, qkNorm: qkNorm)
    if modulation {
      self._adaLN.wrappedValue = [Linear(min(dim, 256), 4 * dim, bias: true)]
    }
    self._attentionNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._attentionNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
    self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)

    let hiddenDim = Int(Float(dim) / 3.0 * 8.0)
    self._feedForward.wrappedValue = ZImageFeedForward(dim: dim, hiddenDim: hiddenDim)

    super.init()
  }

  /// Forward pass with optional hint application
  /// - Parameters:
  ///   - x: Input hidden states
  ///   - attnMask: Attention mask
  ///   - freqsCis: Rotary position embeddings
  ///   - adalnInput: Timestep embedding for adaptive layer norm
  ///   - hints: Optional array of control hints
  ///   - contextScale: Scale factor for hint contribution (default 1.0)
  /// - Returns: Output hidden states
  func callAsFunction(
    _ x: MLXArray,
    attnMask: MLXArray? = nil,
    freqsCis: MLXArray? = nil,
    adalnInput: MLXArray? = nil,
    hints: [MLXArray]? = nil,
    contextScale: Float = 1.0
  ) -> MLXArray {
    var out = x

    if modulation, let c = adalnInput, let adaLNModule = adaLN {
      let mod = adaLNModule[0](c)
      let chunkSize = dim

      let attnScale = (1 + mod[0..., 0..<chunkSize])[.ellipsis, .newAxis, 0...]
      let attnGate = MLX.tanh(mod[0..., chunkSize..<(2 * chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpScale = (1 + mod[0..., (2 * chunkSize)..<(3 * chunkSize)])[.ellipsis, .newAxis, 0...]
      let mlpGate = MLX.tanh(mod[0..., (3 * chunkSize)..<(4 * chunkSize)])[.ellipsis, .newAxis, 0...]

      let attnOut = attention(attentionNorm1(out) * attnScale, attnMask: attnMask, freqsCis: freqsCis)
      out = out + attnGate * attentionNorm2(attnOut)

      let ffnOut = feedForward(ffnNorm1(out) * mlpScale)
      out = out + mlpGate * ffnNorm2(ffnOut)
    } else {
      let attnOut = attention(attentionNorm1(out), attnMask: attnMask, freqsCis: freqsCis)
      out = out + attentionNorm2(attnOut)

      let ffnOut = feedForward(ffnNorm1(out))
      out = out + ffnNorm2(ffnOut)
    }

    // Apply control hint if this layer has one
    if let blockId = blockId, let hints = hints {
      out = out + hints[blockId] * contextScale
    }

    return out
  }
}
