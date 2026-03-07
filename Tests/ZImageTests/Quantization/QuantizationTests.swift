import XCTest

@testable import ZImage

final class QuantizationTests: XCTestCase {

  // MARK: - Quantization Mode Tests

  func testQuantizationModeAffine() {
    let mode = ZImageQuantizationMode.affine
    XCTAssertEqual(mode.rawValue, "affine")
  }

  func testQuantizationModeMXFP4() {
    let mode = ZImageQuantizationMode.mxfp4
    XCTAssertEqual(mode.rawValue, "mxfp4")
  }

  // MARK: - Quantization Spec Tests

  func testQuantizationSpecDefaults() {
    let spec = ZImageQuantizationSpec()
    XCTAssertEqual(spec.groupSize, 32)
    XCTAssertEqual(spec.bits, 8)
    XCTAssertEqual(spec.mode, .affine)
  }

  func testQuantizationSpecCustom() {
    let spec = ZImageQuantizationSpec(groupSize: 64, bits: 4, mode: .mxfp4)
    XCTAssertEqual(spec.groupSize, 64)
    XCTAssertEqual(spec.bits, 4)
    XCTAssertEqual(spec.mode, .mxfp4)
  }

  func testQuantizationSpecDecoding() throws {
    let json = """
      {
        "group_size": 128,
        "bits": 4,
        "mode": "mxfp4"
      }
      """

    let data = json.data(using: .utf8)!
    let spec = try JSONDecoder().decode(ZImageQuantizationSpec.self, from: data)

    XCTAssertEqual(spec.groupSize, 128)
    XCTAssertEqual(spec.bits, 4)
    XCTAssertEqual(spec.mode, .mxfp4)
  }

  func testQuantizationSpecDecodingDefaults() throws {
    let json = "{}"

    let data = json.data(using: .utf8)!
    let spec = try JSONDecoder().decode(ZImageQuantizationSpec.self, from: data)

    XCTAssertEqual(spec.groupSize, 32)
    XCTAssertEqual(spec.bits, 8)
    XCTAssertEqual(spec.mode, .affine)
  }

  func testQuantizationSpecEncoding() throws {
    let spec = ZImageQuantizationSpec(groupSize: 64, bits: 8, mode: .affine)

    let encoder = JSONEncoder()
    let data = try encoder.encode(spec)
    let decoded = try JSONDecoder().decode(ZImageQuantizationSpec.self, from: data)

    XCTAssertEqual(decoded.groupSize, spec.groupSize)
    XCTAssertEqual(decoded.bits, spec.bits)
    XCTAssertEqual(decoded.mode, spec.mode)
  }

  // MARK: - Quantization Manifest Tests

  func testQuantizationManifestDecoding() throws {
    let json = """
      {
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "revision": "main",
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        "layers": [
          {
            "name": "transformer.layers.0.attn.q_proj",
            "shape": [3840, 3840],
            "in_dim": 3840,
            "out_dim": 3840,
            "file": "transformer/model.safetensors"
          }
        ]
      }
      """

    let data = json.data(using: .utf8)!
    let manifest = try JSONDecoder().decode(ZImageQuantizationManifest.self, from: data)

    XCTAssertEqual(manifest.modelId, "Tongyi-MAI/Z-Image-Turbo")
    XCTAssertEqual(manifest.revision, "main")
    XCTAssertEqual(manifest.groupSize, 32)
    XCTAssertEqual(manifest.bits, 8)
    XCTAssertEqual(manifest.mode, "affine")
    XCTAssertEqual(manifest.layers.count, 1)

    let layer = manifest.layers[0]
    XCTAssertEqual(layer.name, "transformer.layers.0.attn.q_proj")
    XCTAssertEqual(layer.shape, [3840, 3840])
    XCTAssertEqual(layer.inDim, 3840)
    XCTAssertEqual(layer.outDim, 3840)
    XCTAssertEqual(layer.file, "transformer/model.safetensors")
  }

  func testQuantizationManifestLayerWithOverrides() throws {
    let json = """
      {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        "layers": [
          {
            "name": "transformer.layers.0.attn.q_proj",
            "shape": [3840, 3840],
            "in_dim": 3840,
            "out_dim": 3840,
            "file": "transformer/model.safetensors",
            "quant_file": "transformer/model_quant.safetensors",
            "group_size": 64,
            "bits": 4,
            "mode": "mxfp4"
          }
        ]
      }
      """

    let data = json.data(using: .utf8)!
    let manifest = try JSONDecoder().decode(ZImageQuantizationManifest.self, from: data)

    let layer = manifest.layers[0]
    XCTAssertEqual(layer.quantFile, "transformer/model_quant.safetensors")
    XCTAssertEqual(layer.groupSize, 64)
    XCTAssertEqual(layer.bits, 4)
    XCTAssertEqual(layer.mode, "mxfp4")
  }

  func testQuantizationManifestMultipleLayers() throws {
    let json = """
      {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        "layers": [
          {
            "name": "transformer.layers.0.attn.q_proj",
            "shape": [3840, 3840],
            "in_dim": 3840,
            "out_dim": 3840,
            "file": "transformer/model-00001.safetensors"
          },
          {
            "name": "transformer.layers.0.attn.k_proj",
            "shape": [3840, 3840],
            "in_dim": 3840,
            "out_dim": 3840,
            "file": "transformer/model-00001.safetensors"
          },
          {
            "name": "transformer.layers.0.ff.linear1",
            "shape": [10240, 3840],
            "in_dim": 3840,
            "out_dim": 10240,
            "file": "transformer/model-00002.safetensors"
          }
        ]
      }
      """

    let data = json.data(using: .utf8)!
    let manifest = try JSONDecoder().decode(ZImageQuantizationManifest.self, from: data)

    XCTAssertEqual(manifest.layers.count, 3)
    XCTAssertEqual(manifest.layers[0].name, "transformer.layers.0.attn.q_proj")
    XCTAssertEqual(manifest.layers[1].name, "transformer.layers.0.attn.k_proj")
    XCTAssertEqual(manifest.layers[2].name, "transformer.layers.0.ff.linear1")
  }

  // MARK: - Quantization Error Tests

  func testInvalidGroupSizeError() {
    let error = ZImageQuantizationError.invalidGroupSize(16)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("16"))
    XCTAssertTrue(error.errorDescription!.contains("32, 64, 128"))
  }

  func testInvalidBitsError() {
    let error = ZImageQuantizationError.invalidBits(6)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("6"))
    XCTAssertTrue(error.errorDescription!.contains("4, 8"))
  }

  func testNoSafetensorsFoundError() {
    let url = URL(fileURLWithPath: "/some/path")
    let error = ZImageQuantizationError.noSafetensorsFound(url)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/some/path"))
  }

  func testQuantizationFailedError() {
    let error = ZImageQuantizationError.quantizationFailed("Memory error")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("Memory error"))
  }

  func testOutputDirectoryCreationFailedError() {
    let url = URL(fileURLWithPath: "/readonly/path")
    let error = ZImageQuantizationError.outputDirectoryCreationFailed(url)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/readonly/path"))
  }

  // MARK: - Quantizer Constants Tests

  func testSupportedGroupSizes() {
    XCTAssertTrue(ZImageQuantizer.supportedGroupSizes.contains(32))
    XCTAssertTrue(ZImageQuantizer.supportedGroupSizes.contains(64))
    XCTAssertTrue(ZImageQuantizer.supportedGroupSizes.contains(128))
    XCTAssertEqual(ZImageQuantizer.supportedGroupSizes.count, 3)
  }

  func testSupportedBits() {
    XCTAssertTrue(ZImageQuantizer.supportedBits.contains(4))
    XCTAssertTrue(ZImageQuantizer.supportedBits.contains(8))
    XCTAssertEqual(ZImageQuantizer.supportedBits.count, 2)
  }

  // MARK: - Has Quantization Tests

  func testHasQuantizationFalse() {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    XCTAssertFalse(ZImageQuantizer.hasQuantization(at: tempDir))
  }

  func testHasQuantizationTrue() throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let manifestURL = tempDir.appendingPathComponent("quantization.json")
    let manifest = """
      {
        "group_size": 32,
        "bits": 8,
        "mode": "affine",
        "layers": []
      }
      """
    try manifest.write(to: manifestURL, atomically: true, encoding: .utf8)

    XCTAssertTrue(ZImageQuantizer.hasQuantization(at: tempDir))
  }

  // MARK: - Tensor Name Transform Tests

  func testTransformerTensorNameTransform() {
    let path = "layers.0.attn.q_proj"
    let result = ZImageQuantizer.transformerTensorName(path)
    XCTAssertEqual(result, path)  // Should be unchanged
  }

  func testTextEncoderTensorNameTransformWithEncoder() {
    let path = "encoder.layers.0.self_attn.q_proj"
    let result = ZImageQuantizer.textEncoderTensorName(path)
    XCTAssertEqual(result, "model.layers.0.self_attn.q_proj")
  }

  func testTextEncoderTensorNameTransformWithoutEncoder() {
    let path = "layers.0.self_attn.q_proj"
    let result = ZImageQuantizer.textEncoderTensorName(path)
    XCTAssertEqual(result, path)  // Should be unchanged
  }

  // MARK: - Manifest Load from File Tests

  func testManifestLoadFromFile() throws {
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempDir) }

    let manifestURL = tempDir.appendingPathComponent("quantization.json")
    let manifestContent = """
      {
        "model_id": "test-model",
        "revision": "v1.0",
        "group_size": 64,
        "bits": 4,
        "mode": "mxfp4",
        "layers": []
      }
      """
    try manifestContent.write(to: manifestURL, atomically: true, encoding: .utf8)

    let manifest = try ZImageQuantizationManifest.load(from: manifestURL)
    XCTAssertEqual(manifest.modelId, "test-model")
    XCTAssertEqual(manifest.revision, "v1.0")
    XCTAssertEqual(manifest.groupSize, 64)
    XCTAssertEqual(manifest.bits, 4)
    XCTAssertEqual(manifest.mode, "mxfp4")
  }

  func testManifestLoadFromFileNotFound() {
    let nonexistentURL = URL(fileURLWithPath: "/nonexistent/quantization.json")
    XCTAssertThrowsError(try ZImageQuantizationManifest.load(from: nonexistentURL))
  }
}
