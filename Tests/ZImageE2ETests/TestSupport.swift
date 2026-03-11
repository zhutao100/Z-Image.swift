import Foundation
import XCTest

private let truthyEnvironmentValues: Set<String> = ["1", "true", "yes", "on"]

private struct SwiftPMPreparationError: LocalizedError {
  let message: String

  var errorDescription: String? { message }
}

private final class SwiftPMTestSupport: @unchecked Sendable {
  static let shared = SwiftPMTestSupport()

  let projectRoot: URL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

  private let lock = NSLock()
  private var preparedConfigurations: Set<String> = []
  private var builtProducts: Set<String> = []

  func ensureMLXMetalLibraryPrepared(configuration: String) throws {
    lock.lock()
    defer { lock.unlock() }

    if preparedConfigurations.contains(configuration) {
      return
    }

    try runProcess(
      executable: "/bin/bash",
      arguments: [
        projectRoot.appendingPathComponent("scripts/build_mlx_metallib.sh").path,
        "--configuration",
        configuration,
      ])
    preparedConfigurations.insert(configuration)
  }

  func ensureProductBuilt(named productName: String, configuration: String) throws {
    lock.lock()
    defer { lock.unlock() }

    let key = "\(configuration):\(productName)"
    if builtProducts.contains(key) {
      return
    }

    try runProcess(
      executable: "/usr/bin/env",
      arguments: [
        "swift",
        "build",
        "--configuration",
        configuration,
        "--product",
        productName,
      ])
    builtProducts.insert(key)
  }

  static func configuration(for executableURL: URL?) -> String {
    guard let path = executableURL?.path.lowercased() else {
      return "debug"
    }
    return path.contains("/release/") ? "release" : "debug"
  }

  private func runProcess(executable: String, arguments: [String]) throws {
    let process = Process()
    process.currentDirectoryURL = projectRoot
    process.executableURL = URL(fileURLWithPath: executable)
    process.arguments = arguments

    let stdout = Pipe()
    let stderr = Pipe()
    process.standardOutput = stdout
    process.standardError = stderr

    try process.run()
    process.waitUntilExit()

    guard process.terminationStatus == 0 else {
      let output =
        String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      let error =
        String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      let combined = [output, error].filter { !$0.isEmpty }.joined(separator: "\n")
      throw SwiftPMPreparationError(
        message:
          "Failed to prepare SwiftPM artifacts with `\(executable) \(arguments.joined(separator: " "))`"
            + (combined.isEmpty ? "" : "\n\(combined)")
      )
    }
  }
}

func requireE2ETestsEnabled() throws {
  let enabled =
    ProcessInfo.processInfo.environment["ZIMAGE_RUN_E2E_TESTS"]?
    .trimmingCharacters(in: .whitespacesAndNewlines)
    .lowercased()
  guard let enabled, truthyEnvironmentValues.contains(enabled) else {
    throw XCTSkip("Set ZIMAGE_RUN_E2E_TESTS=1 to enable CLI end-to-end tests.")
  }
}

func resolveSwiftPMExecutable(named executableName: String, for testCase: AnyClass) throws -> URL {
  guard let executableURL = Bundle(for: testCase).executableURL else {
    throw XCTSkip("Cannot determine the SwiftPM test executable location.")
  }

  let configuration = SwiftPMTestSupport.configuration(for: executableURL)
  let binaryDir = executableURL.deletingLastPathComponent()
  let productsDir =
    binaryDir
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
  let productURL = productsDir.appendingPathComponent(executableName)
  if !FileManager.default.fileExists(atPath: productURL.path) {
    try SwiftPMTestSupport.shared.ensureProductBuilt(named: executableName, configuration: configuration)
  }

  guard FileManager.default.fileExists(atPath: productURL.path) else {
    throw SwiftPMPreparationError(
      message: "\(executableName) was not produced at \(productURL.path) for SwiftPM \(configuration) tests."
    )
  }

  return productURL
}

func ensureMLXMetalLibraryAdjacent(to executableURL: URL) throws {
  let configuration = SwiftPMTestSupport.configuration(for: executableURL)
  let metalLibraryURL = executableURL.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
  if !FileManager.default.fileExists(atPath: metalLibraryURL.path) {
    try SwiftPMTestSupport.shared.ensureMLXMetalLibraryPrepared(configuration: configuration)
  }

  guard FileManager.default.fileExists(atPath: metalLibraryURL.path) else {
    throw SwiftPMPreparationError(
      message:
        "mlx.metallib was not produced at \(metalLibraryURL.path) for SwiftPM \(configuration) tests."
    )
  }
}
