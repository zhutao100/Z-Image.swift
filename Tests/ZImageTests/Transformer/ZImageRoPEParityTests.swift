import MLX
import XCTest

@testable import ZImage

final class ZImageRoPEParityTests: MLXTestCase {
  func testRopeEmbedderMatchesDiffusersReferencePrecompute() {
    let embedder = ZImageRopeEmbedder(theta: 256.0, axesDims: [4, 6, 8], axesLens: [32, 32, 32])
    let ids = MLXArray(
      [
        Int32(1), 0, 0,
        3, 2, 1,
        7, 5, 4,
        9, 8, 6,
      ],
      [4, 3]
    )

    let actual = embedder(ids: ids)
    let expected = diffusersReferenceFreqs(theta: 256.0, axesDims: [4, 6, 8], ids: ids)

    assertClose(actual, expected, accuracy: 2e-6)
  }

  func testApplyComplexRoPEMatchesDiffusersFloat32PathForFloat32Inputs() {
    let freqsCis = diffusersReferenceFreqs(
      theta: 256.0,
      axesDims: [4, 4],
      ids: MLXArray(
        [
          Int32(1), 0,
          2, 1,
          4, 3,
        ],
        [3, 2]
      )
    )
    let query = MLXArray((0..<48).map { Float($0) / 10.0 }, [2, 3, 1, 8])
    let key = MLXArray((0..<48).map { Float($0) / 7.0 }, [2, 3, 1, 8])

    let (actualQuery, actualKey) = ZImageAttentionUtils.applyComplexRoPEBLHD(
      query: query,
      key: key,
      freqsCis: freqsCis
    )
    let expectedQuery = diffusersReferenceApplyRotary(query, freqsCis: freqsCis)
    let expectedKey = diffusersReferenceApplyRotary(key, freqsCis: freqsCis)

    XCTAssertEqual(actualQuery.dtype, .float32)
    XCTAssertEqual(actualKey.dtype, .float32)
    assertClose(actualQuery, expectedQuery, accuracy: 1e-6)
    assertClose(actualKey, expectedKey, accuracy: 1e-6)
  }

  func testApplyComplexRoPEReturnsOriginalBFloat16DTypeLikeDiffusers() {
    let freqsCis = diffusersReferenceFreqs(
      theta: 256.0,
      axesDims: [4, 4],
      ids: MLXArray(
        [
          Int32(1), 0,
          2, 1,
          4, 3,
        ],
        [3, 2]
      )
    )
    let query = MLXArray((0..<48).map { Float($0) / 10.0 }, [2, 3, 1, 8]).asType(.bfloat16)
    let key = MLXArray((0..<48).map { Float($0) / 7.0 }, [2, 3, 1, 8]).asType(.bfloat16)

    let (actualQuery, actualKey) = ZImageAttentionUtils.applyComplexRoPEBLHD(
      query: query,
      key: key,
      freqsCis: freqsCis
    )
    let expectedQuery = diffusersReferenceApplyRotary(query, freqsCis: freqsCis)
    let expectedKey = diffusersReferenceApplyRotary(key, freqsCis: freqsCis)

    XCTAssertEqual(actualQuery.dtype, .bfloat16)
    XCTAssertEqual(actualKey.dtype, .bfloat16)
    assertClose(actualQuery, expectedQuery, accuracy: 2e-2)
    assertClose(actualKey, expectedKey, accuracy: 2e-2)
  }

  private func diffusersReferenceFreqs(
    theta: Float,
    axesDims: [Int],
    ids: MLXArray
  ) -> MLXArray {
    let values = ids.asArray(Int32.self).map(Int.init)
    let rowCount = ids.dim(0)
    let axisCount = axesDims.count
    let totalHalfDim = axesDims.reduce(0) { $0 + $1 / 2 }
    var result: [Float] = []
    result.reserveCapacity(rowCount * totalHalfDim * 2)

    for row in 0..<rowCount {
      for axis in 0..<axisCount {
        let timestep = values[row * axisCount + axis]
        let dim = axesDims[axis]
        let halfDim = dim / 2

        for index in 0..<halfDim {
          let exponent = Double(index * 2) / Double(dim)
          let frequency = pow(Double(theta), -exponent)
          let angle = Float(Double(timestep) * frequency)
          result.append(Float(cos(Double(angle))))
          result.append(Float(sin(Double(angle))))
        }
      }
    }

    return MLXArray(result, [rowCount, totalHalfDim, 2])
  }

  private func diffusersReferenceApplyRotary(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
    let originalDType = x.dtype
    let compute = x.asType(.float32)
    let freqsCos = freqsCis[0..., 0..., 0][.newAxis, 0..., .newAxis, 0...].asType(.float32)
    let freqsSin = freqsCis[0..., 0..., 1][.newAxis, 0..., .newAxis, 0...].asType(.float32)
    let shape = compute.shape
    let newShape = Array(shape.dropLast()) + [shape.last! / 2, 2]
    let reshaped = compute.reshaped(newShape)

    let real = reshaped[0..., 0..., 0..., 0..., 0]
    let imag = reshaped[0..., 0..., 0..., 0..., 1]

    let outReal = real * freqsCos - imag * freqsSin
    let outImag = real * freqsSin + imag * freqsCos
    let rotated = MLX.stacked([outReal, outImag], axis: -1).reshaped(shape)

    switch originalDType {
    case .float16, .bfloat16, .float32, .float64:
      return rotated.asType(originalDType)
    default:
      return rotated
    }
  }

  private func assertClose(
    _ actual: MLXArray,
    _ expected: MLXArray,
    accuracy: Float,
    file: StaticString = #filePath,
    line: UInt = #line
  ) {
    XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)

    let actualValues = actual.asType(.float32).asArray(Float.self)
    let expectedValues = expected.asType(.float32).asArray(Float.self)
    XCTAssertEqual(actualValues.count, expectedValues.count, file: file, line: line)

    for index in actualValues.indices {
      XCTAssertEqual(actualValues[index], expectedValues[index], accuracy: accuracy, "index=\(index)", file: file, line: line)
    }
  }
}
