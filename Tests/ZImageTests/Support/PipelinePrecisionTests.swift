import MLX
import XCTest

@testable import ZImage

final class PipelinePrecisionTests: XCTestCase {
  func testRuntimeDTypeUsesFirstFloatingParameter() {
    let embedder = ZImageTimestepEmbedder(outSize: 16, midSize: 16)

    XCTAssertEqual(PipelineUtilities.runtimeDType(for: embedder), .float32)
  }

  func testCastModelInputToRuntimeDTypeIfNeededCastsToModuleDType() {
    let embedder = ZImageTimestepEmbedder(outSize: 16, midSize: 16)
    let input = MLXArray([Float(0), 1, 2, 3], [1, 4]).asType(.bfloat16)

    let typed = PipelineUtilities.castModelInputToRuntimeDTypeIfNeeded(input, module: embedder)

    XCTAssertEqual(typed.dtype, .float32)
  }

  func testCastModelInputToRuntimeDTypeIfNeededLeavesMatchingInputUntouched() {
    let embedder = ZImageTimestepEmbedder(outSize: 16, midSize: 16)
    let input = MLXArray([Float(0), 1, 2, 3], [1, 4])

    let typed = PipelineUtilities.castModelInputToRuntimeDTypeIfNeeded(input, module: embedder)

    XCTAssertEqual(typed.dtype, .float32)
  }
}
