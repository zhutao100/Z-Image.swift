import MLX

@inline(__always)
func padSequenceByRepeatingLastToken(_ sequence: MLXArray, validLength: Int, padLength: Int) -> MLXArray {
  guard padLength > 0 else { return sequence }

  let batch = sequence.dim(0)
  let featureDim = sequence.dim(sequence.ndim - 1)
  let last = sequence[0..., validLength - 1, 0...].reshaped(batch, 1, featureDim)
  let pad = MLX.broadcast(last, to: [batch, padLength, featureDim])
  return MLX.concatenated([sequence, pad], axis: 1)
}
