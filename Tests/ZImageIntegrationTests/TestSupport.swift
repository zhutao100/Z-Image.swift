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
          "Failed to prepare SwiftPM MLX artifacts with `\(executable) \(arguments.joined(separator: " "))`"
            + (combined.isEmpty ? "" : "\n\(combined)")
      )
    }
  }
}

func integrationTestsEnabled() -> Bool {
  guard
    let enabled =
      ProcessInfo.processInfo.environment["ZIMAGE_RUN_INTEGRATION_TESTS"]?
      .trimmingCharacters(in: .whitespacesAndNewlines)
      .lowercased()
  else {
    return false
  }
  return truthyEnvironmentValues.contains(enabled)
}

func requireIntegrationTestsEnabled() throws {
  guard integrationTestsEnabled() else {
    throw XCTSkip("Set ZIMAGE_RUN_INTEGRATION_TESTS=1 to enable integration tests.")
  }
}

func ensureMLXMetalLibraryColocated(for testCase: AnyClass) throws {
  guard let executableURL = Bundle(for: testCase).executableURL else {
    throw XCTSkip("Cannot determine test executable location for colocating mlx.metallib.")
  }

  let configuration = SwiftPMTestSupport.configuration(for: executableURL)
  let binaryDir = executableURL.deletingLastPathComponent()
  let colocated = binaryDir.appendingPathComponent("mlx.metallib")
  if FileManager.default.fileExists(atPath: colocated.path) { return }

  // Expected layout for SwiftPM:
  //   <bin>/zimage.swiftPackageTests.xctest/Contents/MacOS/zimage.swiftPackageTests
  // and scripts/build_mlx_metallib.sh writes:
  //   <bin>/mlx.metallib
  let binRoot =
    binaryDir
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
  let built = binRoot.appendingPathComponent("mlx.metallib")
  if !FileManager.default.fileExists(atPath: built.path) {
    try SwiftPMTestSupport.shared.ensureMLXMetalLibraryPrepared(configuration: configuration)
  }

  guard FileManager.default.fileExists(atPath: built.path) else {
    throw SwiftPMPreparationError(
      message: "mlx.metallib was not produced at \(built.path) for SwiftPM \(configuration) tests."
    )
  }

  _ = try? FileManager.default.removeItem(at: colocated)
  try FileManager.default.copyItem(at: built, to: colocated)
}
