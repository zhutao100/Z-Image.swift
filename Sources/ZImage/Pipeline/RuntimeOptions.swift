import Foundation

public enum ModuleResidencyPolicy: String, Codable, Sendable {
  case oneShot = "one-shot"
  case warm
  case adaptive

  public var keepsHeavyModulesResident: Bool {
    self != .oneShot
  }
}

public struct ZImageRuntimeOptions: Sendable {
  public var residencyPolicy: ModuleResidencyPolicy

  public init(residencyPolicy: ModuleResidencyPolicy = .oneShot) {
    self.residencyPolicy = residencyPolicy
  }
}
