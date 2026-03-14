import XCTest

@testable import ZImage

final class HuggingFaceHubTests: XCTestCase {
  func testCacheDirectoryPrefersHFHubCacheAfterDefaultInitialization() async throws {
    _ = HuggingFaceHub.cacheDirectory()

    let tempCache = FileManager.default.temporaryDirectory.appendingPathComponent("hf-cache-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempCache, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempCache) }

    try await withEnvironment([
      "HF_HUB_CACHE": tempCache.path,
      "HF_HOME": nil,
    ]) {
      XCTAssertEqual(HuggingFaceHub.cacheDirectory().path, tempCache.path)
    }
  }

  func testCacheDirectoryUsesHFHomeHubSubdirectory() async throws {
    let tempHome = FileManager.default.temporaryDirectory.appendingPathComponent("hf-home-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempHome, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempHome) }

    try await withEnvironment([
      "HF_HUB_CACHE": nil,
      "HF_HOME": tempHome.path,
    ]) {
      let expected = tempHome.appendingPathComponent("hub")
      XCTAssertEqual(HuggingFaceHub.cacheDirectory().path, expected.path)
    }
  }

  func testEnsureSnapshotOfflineUsesCurrentHFHubCacheAfterDefaultInitialization() async throws {
    _ = HuggingFaceHub.cacheDirectory()

    let repoID = "example/offline-model"
    let revision = "main"
    let (tempCache, expectedSnapshot) = try makeTemporarySnapshotCache(repoID: repoID, revision: revision)
    defer { try? FileManager.default.removeItem(at: tempCache) }

    try await withEnvironment([
      "HF_HUB_CACHE": tempCache.path,
      "HF_HOME": nil,
    ]) {
      let snapshot = try await HuggingFaceHub.ensureSnapshot(
        repoId: repoID,
        revision: revision,
        offline: true
      )

      XCTAssertEqual(snapshot.standardizedFileURL.path, expectedSnapshot.standardizedFileURL.path)
    }
  }

  private func makeTemporarySnapshotCache(
    repoID: String,
    revision: String
  ) throws -> (cache: URL, snapshot: URL) {
    let fm = FileManager.default
    let tempCache = fm.temporaryDirectory.appendingPathComponent("hf-cache-\(UUID().uuidString)")
    let repoRoot = tempCache.appendingPathComponent("models--\(repoID.replacingOccurrences(of: "/", with: "--"))")
    let commit = "0123456789abcdef0123456789abcdef01234567"
    let snapshot = repoRoot.appendingPathComponent("snapshots").appendingPathComponent(commit)
    let refs = repoRoot.appendingPathComponent("refs")

    try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
    try fm.createDirectory(at: refs, withIntermediateDirectories: true)
    try Data(commit.utf8).write(to: refs.appendingPathComponent(revision))

    return (tempCache, snapshot)
  }

  private func withEnvironment<T>(
    _ variables: [String: String?],
    perform: () async throws -> T
  ) async throws -> T {
    let previousValues = Dictionary(
      uniqueKeysWithValues: variables.keys.map { name in
        (name, Self.environmentVariable(named: name))
      }
    )

    for (name, value) in variables {
      if let value {
        setenv(name, value, 1)
      } else {
        unsetenv(name)
      }
    }

    defer {
      for (name, previousValue) in previousValues {
        if let previousValue {
          setenv(name, previousValue, 1)
        } else {
          unsetenv(name)
        }
      }
    }

    return try await perform()
  }

  private static func environmentVariable(named name: String) -> String? {
    guard let value = getenv(name) else { return nil }
    return String(cString: value)
  }
}
