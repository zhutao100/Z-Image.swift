import Darwin
import Foundation
import Logging
import MLX

enum ControlMemoryTelemetry {
  static func logPhase(_ phase: String, logger: Logger) {
    logger.info("\(renderPhaseLog(phase: phase, residentBytes: residentMemoryBytes(), snapshot: Memory.snapshot()))")
  }

  static func renderPhaseLog(
    phase: String,
    residentBytes: UInt64?,
    snapshot: Memory.Snapshot
  ) -> String {
    renderPhaseLog(
      phase: phase,
      residentBytes: residentBytes,
      activeMemory: snapshot.activeMemory,
      cacheMemory: snapshot.cacheMemory,
      peakMemory: snapshot.peakMemory
    )
  }

  static func renderPhaseLog(
    phase: String,
    residentBytes: UInt64?,
    activeMemory: Int,
    cacheMemory: Int,
    peakMemory: Int
  ) -> String {
    let residentDescription =
      if let residentBytes {
        "\(format(bytes: residentBytes))(\(residentBytes))"
      } else {
        "unavailable"
      }

    return
      "[control-memory] phase=\(phase) resident=\(residentDescription) active=\(format(bytes: activeMemory))(\(activeMemory)) cache=\(format(bytes: cacheMemory))(\(cacheMemory)) peak=\(format(bytes: peakMemory))(\(peakMemory))"
  }

  static func residentMemoryBytes() -> UInt64? {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let status = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    guard status == KERN_SUCCESS else {
      return nil
    }
    return info.resident_size
  }

  static func format<T: BinaryInteger>(bytes: T) -> String {
    let value = Double(Int64(bytes))
    let gib = 1024.0 * 1024.0 * 1024.0
    let mib = 1024.0 * 1024.0
    let kib = 1024.0
    if value >= gib {
      return String(format: "%.2fGiB", value / gib)
    }
    if value >= mib {
      return String(format: "%.2fMiB", value / mib)
    }
    if value >= kib {
      return String(format: "%.2fKiB", value / kib)
    }
    return "\(bytes)B"
  }
}
