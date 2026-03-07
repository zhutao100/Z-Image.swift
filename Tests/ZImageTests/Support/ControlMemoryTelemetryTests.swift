import XCTest

@testable import ZImage

final class ControlMemoryTelemetryTests: XCTestCase {
  func testRenderPhaseLogIncludesAllCounters() {
    let line = ControlMemoryTelemetry.renderPhaseLog(
      phase: "control-context.before-build",
      residentBytes: 128 * 1024 * 1024,
      activeMemory: 64 * 1024 * 1024,
      cacheMemory: 32 * 1024 * 1024,
      peakMemory: 96 * 1024 * 1024
    )

    XCTAssertTrue(line.contains("[control-memory]"))
    XCTAssertTrue(line.contains("phase=control-context.before-build"))
    XCTAssertTrue(line.contains("resident=128.00MiB"))
    XCTAssertTrue(line.contains("active=64.00MiB"))
    XCTAssertTrue(line.contains("cache=32.00MiB"))
    XCTAssertTrue(line.contains("peak=96.00MiB"))
  }

  func testFormatUsesGiBForLargeValues() {
    XCTAssertEqual(
      ControlMemoryTelemetry.format(bytes: 3 * 1024 * 1024 * 1024),
      "3.00GiB"
    )
  }
}
