import MLX
import XCTest

@testable import ZImage

final class SafeTensorsReaderTests: MLXTestCase {

  // MARK: - Error Type Tests

  func testFileTooSmallError() {
    let url = URL(fileURLWithPath: "/path/to/file.safetensors")
    let error = SafeTensorsReaderError.fileTooSmall(url)

    switch error {
    case .fileTooSmall(let errorURL):
      XCTAssertEqual(errorURL, url)
    default:
      XCTFail("Wrong error type")
    }
  }

  func testInvalidHeaderLengthError() {
    let url = URL(fileURLWithPath: "/path/to/file.safetensors")
    let error = SafeTensorsReaderError.invalidHeaderLength(url)

    switch error {
    case .invalidHeaderLength(let errorURL):
      XCTAssertEqual(errorURL, url)
    default:
      XCTFail("Wrong error type")
    }
  }

  func testMalformedHeaderError() {
    let url = URL(fileURLWithPath: "/path/to/file.safetensors")
    let error = SafeTensorsReaderError.malformedHeader(url)

    switch error {
    case .malformedHeader(let errorURL):
      XCTAssertEqual(errorURL, url)
    default:
      XCTFail("Wrong error type")
    }
  }

  func testTensorMetadataMissingError() {
    let error = SafeTensorsReaderError.tensorMetadataMissing("weight")

    switch error {
    case .tensorMetadataMissing(let name):
      XCTAssertEqual(name, "weight")
    default:
      XCTFail("Wrong error type")
    }
  }

  func testUnsupportedDTypeError() {
    let error = SafeTensorsReaderError.unsupportedDType("COMPLEX64")

    switch error {
    case .unsupportedDType(let dtype):
      XCTAssertEqual(dtype, "COMPLEX64")
    default:
      XCTFail("Wrong error type")
    }
  }

  func testInvalidOffsetsError() {
    let error = SafeTensorsReaderError.invalidOffsets(name: "tensor_name")

    switch error {
    case .invalidOffsets(let name):
      XCTAssertEqual(name, "tensor_name")
    default:
      XCTFail("Wrong error type")
    }
  }

  func testInvalidShapeError() {
    let error = SafeTensorsReaderError.invalidShape(name: "tensor_name")

    switch error {
    case .invalidShape(let name):
      XCTAssertEqual(name, "tensor_name")
    default:
      XCTFail("Wrong error type")
    }
  }

  func testTensorNotFoundError() {
    let error = SafeTensorsReaderError.tensorNotFound("missing_tensor")

    switch error {
    case .tensorNotFound(let name):
      XCTAssertEqual(name, "missing_tensor")
    default:
      XCTFail("Wrong error type")
    }
  }

  // MARK: - SafeTensorMetadata Tests

  func testSafeTensorMetadataElementCount() {
    let metadata = SafeTensorMetadata(
      name: "test",
      dtype: .float32,
      shape: [10, 20, 30],
      dataOffset: 0,
      byteCount: 10 * 20 * 30 * 4
    )

    XCTAssertEqual(metadata.elementCount, 6000)  // 10 * 20 * 30
  }

  func testSafeTensorMetadataEmptyShape() {
    let metadata = SafeTensorMetadata(
      name: "scalar",
      dtype: .float32,
      shape: [],
      dataOffset: 0,
      byteCount: 4
    )

    XCTAssertEqual(metadata.elementCount, 1)  // Empty product is 1
  }

  func testSafeTensorMetadata1D() {
    let metadata = SafeTensorMetadata(
      name: "vector",
      dtype: .float16,
      shape: [256],
      dataOffset: 100,
      byteCount: 256 * 2
    )

    XCTAssertEqual(metadata.name, "vector")
    XCTAssertEqual(metadata.dtype, .float16)
    XCTAssertEqual(metadata.shape, [256])
    XCTAssertEqual(metadata.dataOffset, 100)
    XCTAssertEqual(metadata.byteCount, 512)
    XCTAssertEqual(metadata.elementCount, 256)
  }

  // MARK: - DType Mapping Tests (Conceptual)

  func testExpectedByteSizes() {
    // These tests verify the expected byte sizes for different dtypes
    let dtypeSizes: [(DType, Int)] = [
      (.float32, 4),
      (.float16, 2),
      (.bfloat16, 2),
      (.float64, 8),
      (.int32, 4),
      (.int64, 8),
      (.int16, 2),
      (.int8, 1),
      (.uint32, 4),
      (.uint64, 8),
      (.uint16, 2),
      (.uint8, 1),
      (.bool, 1),
    ]

    for (dtype, expectedSize) in dtypeSizes {
      XCTAssertEqual(dtype.size, expectedSize, "Size mismatch for \(dtype)")
    }
  }

  // MARK: - File Handling Tests

  func testNonExistentFile() {
    let url = URL(fileURLWithPath: "/nonexistent/path/model.safetensors")
    XCTAssertThrowsError(try SafeTensorsReader(fileURL: url))
  }

  func testEmptyFile() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("empty.safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    // Create empty file
    FileManager.default.createFile(atPath: fileURL.path, contents: Data(), attributes: nil)

    XCTAssertThrowsError(try SafeTensorsReader(fileURL: fileURL)) { error in
      if case SafeTensorsReaderError.fileTooSmall = error {
        // Expected
      } else {
        XCTFail("Expected fileTooSmall error, got \(error)")
      }
    }
  }

  func testFileTooSmallForHeader() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("small.safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    // Create file with only 4 bytes (need at least 8 for header length)
    let smallData = Data([0, 0, 0, 0])
    try smallData.write(to: fileURL)

    XCTAssertThrowsError(try SafeTensorsReader(fileURL: fileURL)) { error in
      if case SafeTensorsReaderError.fileTooSmall = error {
        // Expected
      } else {
        XCTFail("Expected fileTooSmall error, got \(error)")
      }
    }
  }

  // MARK: - Integration Test with Real SafeTensors File

  func testCreateAndReadSafeTensors() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    // Create test tensors (use explicit Float arrays to avoid float64, then convert to bfloat16)
    let tensor1Values: [Float] = [1.0, 2.0, 3.0, 4.0]
    let tensor2Values: [Float] = [0.5, 1.5, 2.5]
    let tensor1 = MLXArray(tensor1Values, [2, 2]).asType(.bfloat16)
    let tensor2 = MLXArray(tensor2Values, [3]).asType(.bfloat16)

    let arrays: [String: MLXArray] = [
      "weight": tensor1,
      "bias": tensor2,
    ]

    // Save using MLX
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    // Read back using SafeTensorsReader
    let reader = try SafeTensorsReader(fileURL: fileURL)

    XCTAssertEqual(reader.tensorNames.count, 2)
    XCTAssertTrue(reader.contains("weight"))
    XCTAssertTrue(reader.contains("bias"))
    XCTAssertFalse(reader.contains("nonexistent"))

    // Check metadata
    let weightMeta = reader.metadata(for: "weight")
    XCTAssertNotNil(weightMeta)
    XCTAssertEqual(weightMeta?.shape, [2, 2])
    XCTAssertEqual(weightMeta?.elementCount, 4)

    let biasMeta = reader.metadata(for: "bias")
    XCTAssertNotNil(biasMeta)
    XCTAssertEqual(biasMeta?.shape, [3])
    XCTAssertEqual(biasMeta?.elementCount, 3)

    // Load tensors
    let loadedWeight = try reader.tensor(named: "weight")
    MLX.eval(loadedWeight)
    XCTAssertEqual(loadedWeight.shape, [2, 2])

    let loadedBias = try reader.tensor(named: "bias")
    MLX.eval(loadedBias)
    XCTAssertEqual(loadedBias.shape, [3])

    // Verify values
    let weightValues = loadedWeight.asArray(Float.self)
    XCTAssertEqual(weightValues[0], 1.0, accuracy: 1e-5)
    XCTAssertEqual(weightValues[1], 2.0, accuracy: 1e-5)

    let biasValues = loadedBias.asArray(Float.self)
    XCTAssertEqual(biasValues[0], 0.5, accuracy: 1e-5)
  }

  func testLoadAllTensors() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_all_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    // Use explicit Float arrays to avoid float64
    let w1: [Float] = [1.0, 2.0]
    let b1: [Float] = [0.1]
    let w2: [Float] = [3.0, 4.0, 5.0, 6.0]
    let arrays: [String: MLXArray] = [
      "layer1.weight": MLXArray(w1, [2]).asType(.bfloat16),
      "layer1.bias": MLXArray(b1, [1]).asType(.bfloat16),
      "layer2.weight": MLXArray(w2, [2, 2]).asType(.bfloat16),
    ]

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let loaded = try reader.loadAllTensors()

    XCTAssertEqual(loaded.count, 3)
    XCTAssertNotNil(loaded["layer1.weight"])
    XCTAssertNotNil(loaded["layer1.bias"])
    XCTAssertNotNil(loaded["layer2.weight"])
  }

  func testLoadAllTensorsWithDtypeConversion() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_dtype_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let tensorValues: [Float] = [1.0, 2.0, 3.0]
    let arrays: [String: MLXArray] = [
      "tensor": MLXArray(tensorValues, [3]).asType(.float32)
    ]

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let loaded = try reader.loadAllTensors(as: .bfloat16)

    let tensor = loaded["tensor"]!
    XCTAssertEqual(tensor.dtype, .bfloat16)
  }

  func testTensorNotFoundThrows() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_notfound_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let existsValues: [Float] = [1.0]
    let arrays: [String: MLXArray] = [
      "exists": MLXArray(existsValues, [1]).asType(.bfloat16)
    ]

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)

    XCTAssertThrowsError(try reader.tensor(named: "does_not_exist")) { error in
      if case SafeTensorsReaderError.tensorNotFound(let name) = error {
        XCTAssertEqual(name, "does_not_exist")
      } else {
        XCTFail("Expected tensorNotFound error")
      }
    }
  }

  func testAllMetadata() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_meta_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let aValues: [Float] = [1.0, 2.0]
    let bValues: [Float] = [3.0]
    let arrays: [String: MLXArray] = [
      "a": MLXArray(aValues, [2]).asType(.bfloat16),
      "b": MLXArray(bValues, [1]).asType(.bfloat16),
    ]

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let allMeta = reader.allMetadata()

    XCTAssertEqual(allMeta.count, 2)

    let names = Set(allMeta.map { $0.name })
    XCTAssertTrue(names.contains("a"))
    XCTAssertTrue(names.contains("b"))
  }

  func testTensorData() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("test_data_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let tensorValues: [Float] = [1.0, 2.0, 3.0, 4.0]
    let arrays: [String: MLXArray] = [
      "tensor": MLXArray(tensorValues, [4]).asType(.float32)
    ]

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let data = try reader.tensorData(named: "tensor")

    // 4 floats * 4 bytes = 16 bytes
    XCTAssertEqual(data.count, 16)
  }
}
