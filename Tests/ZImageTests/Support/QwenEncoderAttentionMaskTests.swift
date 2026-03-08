import MLX
import XCTest

@testable import ZImage

final class QwenEncoderAttentionMaskTests: XCTestCase {
  private func makeEncoder() -> QwenEncoder {
    QwenEncoder(
      configuration: .init(
        vocabSize: 32,
        hiddenSize: 8,
        numHiddenLayers: 1,
        numAttentionHeads: 1,
        numKeyValueHeads: 1,
        intermediateSize: 16,
        maxPositionEmbeddings: 16,
        promptDropIndex: 0,
        headDim: 8
      )
    )
  }

  func testCreateAttentionMaskReturnsCausalModeWithoutPaddingMask() {
    let encoder = makeEncoder()
    let hiddenStates = MLX.zeros([1, 4, 8])

    let mask = encoder.createAttentionMask(h: hiddenStates, attentionMask: nil)

    guard case .causal = mask else {
      XCTFail("Expected causal mask when no prompt attention mask is provided")
      return
    }
  }

  func testCreateAttentionMaskCombinesCausalAndPaddingAsBoolMask() {
    let encoder = makeEncoder()
    let hiddenStates = MLX.zeros([2, 4, 8])
    let attentionMask = MLXArray(
      [Int32(1), 1, 0, 0,
       1, 0, 0, 0],
      [2, 4]
    )

    let maskMode = encoder.createAttentionMask(h: hiddenStates, attentionMask: attentionMask)

    guard case .array(let mask) = maskMode else {
      XCTFail("Expected explicit array mask when prompt attention mask is provided")
      return
    }

    XCTAssertEqual(mask.dtype, .bool)
    XCTAssertEqual(mask.shape, [2, 1, 4, 4])

    let actual = mask.asType(.int32).reshaped(-1).asArray(Int32.self)
    let expected: [Int32] = [
      1, 0, 0, 0,
      1, 1, 0, 0,
      1, 1, 0, 0,
      1, 1, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
      1, 0, 0, 0,
    ]

    XCTAssertEqual(actual, expected)
  }

  func testProcessTextEmbeddingsPadsBatchToLongestValidLength() {
    let hiddenStates = MLXArray(
      [
        Float(1), 10,
        2, 20,
        3, 30,
        4, 40,
        5, 50,
        6, 60,
        7, 70,
        8, 80,
      ],
      [2, 4, 2]
    )
    let attentionMask = MLXArray(
      [
        Int32(1), 1, 1, 0,
        1, 0, 0, 0,
      ],
      [2, 4]
    )

    let (embeddings, mask) = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      dropIndex: 0
    )

    XCTAssertEqual(embeddings.shape, [2, 3, 2])
    XCTAssertEqual(mask.shape, [2, 3])

    let maskValues = mask.asArray(Int32.self)
    XCTAssertEqual(maskValues, [1, 1, 1, 1, 0, 0])

    let embeddingValues = embeddings.asArray(Float.self)
    XCTAssertEqual(embeddingValues[0], 1.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[1], 10.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[4], 3.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[5], 30.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[6], 5.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[7], 50.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[10], 0.0, accuracy: 1e-6)
    XCTAssertEqual(embeddingValues[11], 0.0, accuracy: 1e-6)
  }
}
