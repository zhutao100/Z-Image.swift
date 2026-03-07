import Logging
import MLX
import MLXNN

public enum ZImageControlWeightsMapping {
  private static func canonicalizeControlnetQuantizedParameterKey(_ key: String) -> String {
    var paramKey = key
    paramKey = paramKey.replacingOccurrences(of: "feed_forward", with: "feedForward")
    paramKey = paramKey.replacingOccurrences(of: "to_q", with: "toQ")
    paramKey = paramKey.replacingOccurrences(of: "to_k", with: "toK")
    paramKey = paramKey.replacingOccurrences(of: "to_v", with: "toV")
    paramKey = paramKey.replacingOccurrences(of: "to_out", with: "toOut")
    paramKey = paramKey.replacingOccurrences(of: "before_proj", with: "beforeProj")
    paramKey = paramKey.replacingOccurrences(of: "after_proj", with: "afterProj")
    paramKey = paramKey.replacingOccurrences(of: "norm_q", with: "normQ")
    paramKey = paramKey.replacingOccurrences(of: "norm_k", with: "normK")
    paramKey = paramKey.replacingOccurrences(of: "attention_norm1", with: "attentionNorm1")
    paramKey = paramKey.replacingOccurrences(of: "attention_norm2", with: "attentionNorm2")
    paramKey = paramKey.replacingOccurrences(of: "ffn_norm1", with: "ffnNorm1")
    paramKey = paramKey.replacingOccurrences(of: "ffn_norm2", with: "ffnNorm2")
    paramKey = paramKey.replacingOccurrences(of: "adaLN_modulation", with: "adaLN")
    return paramKey
  }

  public static func applyControlnetWeights(
    weights: [String: MLXArray],
    to controlnet: ZImageControlNetModel,
    manifest: ZImageQuantizationManifest?,
    logger: Logger
  ) {
    let isQuantized = manifest != nil
    if let manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyControlnetQuantization(
        to: controlnet,
        manifest: manifest,
        availableKeys: availableKeys
      )
      logger.info("Applied quantization to controlnet (\(manifest.bits)-bit, group_size=\(manifest.groupSize))")
    }
    controlnet.loadControlXEmbedderWeights(from: weights)
    for (idx, block) in controlnet.controlNoiseRefiner.enumerated() {
      if isQuantized {
        let prefix = "controlNoiseRefiner.\(idx)"
        ZImageModuleWeightsApplier.applyToModule(
          block,
          weights: weights,
          prefix: prefix,
          logger: logger,
          tensorNameTransform: ZImageQuantizer.controlnetTensorName,
          parameterKeyTransform: canonicalizeControlnetQuantizedParameterKey
        )
      } else {
        let prefix = "control_noise_refiner.\(idx)"
        applyControlTransformerBlockWeights(weights: weights, prefix: prefix, to: block)
      }
    }
    for (idx, block) in controlnet.controlLayers.enumerated() {
      if isQuantized {
        let prefix = "controlLayers.\(idx)"
        ZImageModuleWeightsApplier.applyToModule(
          block,
          weights: weights,
          prefix: prefix,
          logger: logger,
          tensorNameTransform: ZImageQuantizer.controlnetTensorName,
          parameterKeyTransform: canonicalizeControlnetQuantizedParameterKey
        )
      } else {
        let prefix = "control_layers.\(idx)"
        applyControlTransformerBlockWeights(weights: weights, prefix: prefix, to: block)
      }
    }
    logger.info("Applied controlnet weights")
  }

  // swiftlint:disable:next cyclomatic_complexity
  private static func applyControlTransformerBlockWeights(
    weights: [String: MLXArray],
    prefix: String,
    to block: ZImageControlTransformerBlock
  ) {
    if let beforeProj = block.beforeProj {
      if let w = weights["\(prefix).before_proj.weight"] {
        beforeProj.weight._updateInternal(w)
      }
      if let b = weights["\(prefix).before_proj.bias"] {
        beforeProj.bias?._updateInternal(b)
      }
    }
    if let w = weights["\(prefix).after_proj.weight"] {
      block.afterProj.weight._updateInternal(w)
    }
    if let b = weights["\(prefix).after_proj.bias"] {
      block.afterProj.bias?._updateInternal(b)
    }
    if let w = weights["\(prefix).attention.to_q.weight"] {
      block.attention.toQ.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_k.weight"] {
      block.attention.toK.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_v.weight"] {
      block.attention.toV.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.to_out.0.weight"] {
      block.attention.toOut[0].weight._updateInternal(w)
    }
    if let b = weights["\(prefix).attention.to_out.0.bias"] {
      block.attention.toOut[0].bias?._updateInternal(b)
    }
    if let w = weights["\(prefix).attention.norm_q.weight"] {
      block.attention.normQ?.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention.norm_k.weight"] {
      block.attention.normK?.weight._updateInternal(w)
    }
    if let adaLN = block.adaLN, adaLN.count > 0 {
      if let w = weights["\(prefix).adaLN_modulation.0.weight"] {
        adaLN[0].weight._updateInternal(w)
      } else if let w = weights["\(prefix).adaLN_modulation.1.weight"] {
        adaLN[0].weight._updateInternal(w)
      }
      if let b = weights["\(prefix).adaLN_modulation.0.bias"] {
        adaLN[0].bias?._updateInternal(b)
      } else if let b = weights["\(prefix).adaLN_modulation.1.bias"] {
        adaLN[0].bias?._updateInternal(b)
      }
    }
    if let w = weights["\(prefix).attention_norm1.weight"] {
      block.attentionNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm1.weight"] {
      block.ffnNorm1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).attention_norm2.weight"] {
      block.attentionNorm2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).ffn_norm2.weight"] {
      block.ffnNorm2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w1.weight"] {
      block.feedForward.w1.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w2.weight"] {
      block.feedForward.w2.weight._updateInternal(w)
    }
    if let w = weights["\(prefix).feed_forward.w3.weight"] {
      block.feedForward.w3.weight._updateInternal(w)
    }
  }
}
