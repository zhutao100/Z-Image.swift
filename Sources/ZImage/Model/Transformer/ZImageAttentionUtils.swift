import Foundation
import MLX

enum ZImageAttentionUtils {

  static func applyComplexRoPEBLHD(
    query: MLXArray,
    key: MLXArray,
    freqsCis: MLXArray
  ) -> (MLXArray, MLXArray) {
    let freqsCos = freqsCis[0..., 0..., 0][.newAxis, 0..., .newAxis, 0...]
    let freqsSin = freqsCis[0..., 0..., 1][.newAxis, 0..., .newAxis, 0...]

    return (
      applyRotary(query, freqsCos, freqsSin),
      applyRotary(key, freqsCos, freqsSin)
    )
  }

  @inline(__always)
  private static func applyRotary(
    _ x: MLXArray,
    _ freqsCos: MLXArray,
    _ freqsSin: MLXArray
  ) -> MLXArray {
    let originalDType = x.dtype
    let computeInput = x.asType(.float32)
    let cosInput = freqsCos.dtype == .float32 ? freqsCos : freqsCos.asType(.float32)
    let sinInput = freqsSin.dtype == .float32 ? freqsSin : freqsSin.asType(.float32)
    let shape = computeInput.shape
    let newShape = Array(shape.dropLast()) + [shape.last! / 2, 2]
    let xReshaped = computeInput.reshaped(newShape)

    let xReal = xReshaped[0..., 0..., 0..., 0..., 0]
    let xImag = xReshaped[0..., 0..., 0..., 0..., 1]

    let outReal = xReal * cosInput - xImag * sinInput
    let outImag = xReal * sinInput + xImag * cosInput

    let rotated = MLX.stacked([outReal, outImag], axis: -1).reshaped(shape)
    switch originalDType {
    case .float16, .bfloat16, .float32, .float64:
      return rotated.asType(originalDType)
    default:
      return rotated
    }
  }
}
