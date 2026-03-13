import Darwin
import Dispatch
import Foundation
import Logging
import ZImage
import ZImageCLICommon

private final class AsyncBox<T>: @unchecked Sendable {
  var value: T

  init(_ value: T) {
    self.value = value
  }
}

private enum WorkerKind: String, Sendable {
  case text
  case control
}

private struct WorkerProfile: Hashable, Sendable {
  var kind: WorkerKind
  var model: String?
  var weightsVariant: String?
  var controlnetWeights: String?
  var controlnetFile: String?
  var maxSequenceLength: Int
  var residencyPolicy: ModuleResidencyPolicy
  var forceTransformerOverrideOnly: Bool
}

private enum ResidentWorker: Sendable {
  case text(TextServeWorker)
  case control(ControlServeWorker)

  func unload() async {
    switch self {
    case .text(let worker):
      await worker.unload()
    case .control(let worker):
      await worker.unload()
    }
  }
}

private struct ResidentWorkerState: Sendable {
  var profile: WorkerProfile
  var worker: ResidentWorker
  var lastUsedAt: Date
}

private final class ServerControl: @unchecked Sendable {
  private let lock = NSLock()
  private var serverFD: Int32?

  init(serverFD: Int32) {
    self.serverFD = serverFD
  }

  func shutdown() {
    lock.lock()
    defer { lock.unlock() }

    guard let serverFD else { return }
    self.serverFD = nil
    close(serverFD)
  }
}

private enum QueueGate {
  case execute
  case cancelled
}

private struct QueuedSubmission {
  var jobID: String
  var eventSink: @Sendable (ServiceEventEnvelope) throws -> Void
  var continuation: CheckedContinuation<QueueGate, Never>
}

private enum CancelDisposition {
  case active
  case queued
  case notFound
}

public final class ServiceClient {
  private let socketPath: String
  private let decoder = JSONDecoder()
  private let encoder = JSONEncoder()

  public init(socketPath: String? = nil) {
    self.socketPath = ServiceSocketPath.resolve(socketPath)
  }

  public func submit(
    job: GenerationJobPayload,
    jobID: String = UUID().uuidString,
    eventHandler: (ServiceEventEnvelope) throws -> Void
  ) throws {
    let connection = try UnixDomainSocket.connect(path: socketPath)
    let request = ServiceRequestEnvelope(
      type: .submit,
      submission: ServiceSubmissionPayload(jobID: jobID, job: CodableGenerationJob.from(job))
    )
    try send(
      request,
      over: connection,
      terminalTypes: [.completed, .failed, .cancelled],
      eventHandler: eventHandler
    )
  }

  public func status() throws -> ServiceStatusSnapshot {
    let connection = try UnixDomainSocket.connect(path: socketPath)
    let request = ServiceRequestEnvelope(type: .status)
    var snapshot: ServiceStatusSnapshot?
    try send(request, over: connection, terminalTypes: [.status, .failed]) { event in
      switch event.type {
      case .status:
        snapshot = event.status
      case .failed:
        throw CLIError(message: event.message ?? "Status request failed")
      default:
        break
      }
    }

    guard let snapshot else {
      throw ServiceTransportError.invalidMessage("Service status response did not include a snapshot")
    }
    return snapshot
  }

  public func cancel(jobID: String) throws -> ServiceEventEnvelope {
    let connection = try UnixDomainSocket.connect(path: socketPath)
    let request = ServiceRequestEnvelope(
      type: .cancel,
      cancellation: ServiceCancellationPayload(jobID: jobID)
    )
    var terminalEvent: ServiceEventEnvelope?
    try send(request, over: connection, terminalTypes: [.cancelled, .failed]) { event in
      terminalEvent = event
      if event.type == .failed {
        throw CLIError(message: event.message ?? "Cancel request failed")
      }
    }

    guard let terminalEvent else {
      throw ServiceTransportError.invalidMessage("Service cancel response was incomplete")
    }
    return terminalEvent
  }

  public func shutdown() throws -> ServiceEventEnvelope {
    let connection = try UnixDomainSocket.connect(path: socketPath)
    let request = ServiceRequestEnvelope(type: .shutdown)
    var terminalEvent: ServiceEventEnvelope?
    try send(request, over: connection, terminalTypes: [.shutdownAcknowledged, .failed]) { event in
      terminalEvent = event
      if event.type == .failed {
        throw CLIError(message: event.message ?? "Shutdown request failed")
      }
    }

    guard let terminalEvent else {
      throw ServiceTransportError.invalidMessage("Service shutdown response was incomplete")
    }
    return terminalEvent
  }

  private func send(
    _ request: ServiceRequestEnvelope,
    over connection: SocketConnection,
    terminalTypes: Set<ServiceEventType>,
    eventHandler: (ServiceEventEnvelope) throws -> Void
  ) throws {
    let payload = try encoder.encode(request)
    guard let line = String(data: payload, encoding: .utf8) else {
      throw ServiceTransportError.invalidMessage("Failed to encode service request")
    }
    try connection.writeLine(line)

    while let responseLine = try connection.readLine() {
      let event = try decoder.decode(ServiceEventEnvelope.self, from: Data(responseLine.utf8))
      try eventHandler(event)
      if terminalTypes.contains(event.type) {
        return
      }
    }

    throw ServiceTransportError.invalidMessage("Service closed the connection before sending a terminal event")
  }
}

public final class StagingServiceDaemon {
  private let options: ServeOptions
  private let socketPath: String
  private let logger: Logger
  private let coordinator: SerialServiceCoordinator

  public init(options: ServeOptions, logger: Logger) {
    self.options = options
    self.socketPath = ServiceSocketPath.resolve(options.socketPath)
    self.logger = logger
    self.coordinator = SerialServiceCoordinator(options: options, logger: logger)
  }

  public func run() throws {
    let serverFD = try UnixDomainSocket.makeServerSocket(path: socketPath)
    let serverControl = ServerControl(serverFD: serverFD)
    let idleTimer = makeIdleTimer()
    defer {
      idleTimer.cancel()
      serverControl.shutdown()
      unlink(socketPath)
    }

    let coordinator = self.coordinator
    try waitForAsync {
      try await coordinator.prewarmIfNeeded()
    }

    logger.info(
      "Staging daemon listening on \(socketPath) with residency policy \(options.residencyPolicy.rawValue)")

    while true {
      let connection: SocketConnection
      do {
        connection = try UnixDomainSocket.acceptClient(serverFD: serverFD)
      } catch let error as ServiceTransportError {
        if case .acceptFailed(let message) = error, message.contains("Bad file descriptor") {
          return
        }
        throw error
      }

      let coordinator = self.coordinator
      let logger = self.logger
      let socketPath = self.socketPath
      DispatchQueue.global(qos: .userInitiated).async {
        do {
          try waitForAsync {
            try await Self.handleConnection(
              connection,
              socketPath: socketPath,
              coordinator: coordinator,
              logger: logger,
              shutdownHandler: {
                serverControl.shutdown()
              }
            )
          }
        } catch {
          logger.error("\(CLIErrors.describe(error))")
        }
      }
    }
  }

  private func makeIdleTimer() -> DispatchSourceTimer {
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
    timer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
    let coordinator = self.coordinator
    timer.setEventHandler {
      try? waitForAsync {
        await coordinator.evictIdleWorkerIfNeeded(now: Date())
      }
    }
    timer.resume()
    return timer
  }

  private static func handleConnection(
    _ connection: SocketConnection,
    socketPath: String,
    coordinator: SerialServiceCoordinator,
    logger: Logger,
    shutdownHandler: @escaping @Sendable () -> Void
  ) async throws {
    let decoder = JSONDecoder()
    let encoder = JSONEncoder()
    let sendEvent: @Sendable (ServiceEventEnvelope) throws -> Void = { event in
      let data = try encoder.encode(event)
      guard let line = String(data: data, encoding: .utf8) else {
        throw ServiceTransportError.invalidMessage("Failed to encode event payload")
      }
      try connection.writeLine(line)
    }

    guard let line = try connection.readLine() else {
      throw ServiceTransportError.invalidMessage("Missing request payload")
    }
    let request = try decoder.decode(ServiceRequestEnvelope.self, from: Data(line.utf8))
    switch request.type {
    case .submit:
      guard let submission = request.submission else {
        throw ServiceTransportError.invalidMessage("Missing submission payload")
      }
      try await coordinator.submit(submission, eventSink: sendEvent)
    case .status:
      let status = await coordinator.status(socketPath: socketPath)
      try sendEvent(.init(type: .status, status: status))
    case .cancel:
      guard let cancellation = request.cancellation else {
        throw ServiceTransportError.invalidMessage("Missing cancellation payload")
      }
      let disposition = await coordinator.cancel(jobID: cancellation.jobID)
      switch disposition {
      case .active:
        try sendEvent(
          .init(
            type: .cancelled,
            jobID: cancellation.jobID,
            message: "Cancellation requested for active job \(cancellation.jobID)"
          ))
      case .queued:
        try sendEvent(
          .init(
            type: .cancelled,
            jobID: cancellation.jobID,
            message: "Cancelled queued job \(cancellation.jobID)"
          ))
      case .notFound:
        try sendEvent(
          .init(
            type: .failed,
            jobID: cancellation.jobID,
            message: "No active or queued job matched \(cancellation.jobID)"
          ))
      }
    case .shutdown:
      if await coordinator.requestShutdown() {
        try sendEvent(
          .init(
            type: .shutdownAcknowledged,
            message: "Shutdown acknowledged for \(socketPath)"
          ))
        shutdownHandler()
      } else {
        try sendEvent(
          .init(
            type: .failed,
            message: "Cannot shutdown while a job is active or queued"
          ))
      }
    }
  }
}

private func waitForAsync(_ operation: @escaping @Sendable () async throws -> Void) throws {
  let semaphore = DispatchSemaphore(value: 0)
  let errorBox = AsyncBox<Error?>(nil)
  Task {
    do {
      try await operation()
    } catch {
      errorBox.value = error
    }
    semaphore.signal()
  }
  semaphore.wait()
  if let error = errorBox.value {
    throw error
  }
}

actor SerialServiceCoordinator {
  private static let adaptiveMinimumAvailableMemoryBytes: UInt64 = 8 * 1024 * 1024 * 1024

  private let options: ServeOptions
  private let logger: Logger
  private var isBusy = false
  private var isExecuting = false
  private var isShuttingDown = false
  private var activeJobID: String?
  private var activeJobTask: Task<URL, Error>?
  private var waiters: [QueuedSubmission] = []
  private var residentWorker: ResidentWorkerState?

  init(options: ServeOptions, logger: Logger) {
    self.options = options
    self.logger = logger
  }

  func prewarmIfNeeded() async throws {
    guard let profile = warmProfile() else { return }
    logger.info("Prewarming \(profile.kind.rawValue) worker for model \(profile.model ?? ZImageRepository.id)")
    let worker = try await makeWorker(for: profile)
    residentWorker = .init(profile: profile, worker: worker, lastUsedAt: Date())
  }

  func submit(
    _ submission: ServiceSubmissionPayload,
    eventSink: @escaping @Sendable (ServiceEventEnvelope) throws -> Void
  ) async throws {
    if isShuttingDown {
      try eventSink(
        .init(
          type: .failed,
          jobID: submission.jobID,
          message: "Service is shutting down and not accepting new jobs"
        ))
      return
    }

    let queuePosition = isBusy ? waiters.count + 1 : 0
    try eventSink(.init(type: .accepted, jobID: submission.jobID, queuePosition: queuePosition))

    if isBusy {
      let gate = await withCheckedContinuation { (continuation: CheckedContinuation<QueueGate, Never>) in
        waiters.append(.init(jobID: submission.jobID, eventSink: eventSink, continuation: continuation))
      }
      guard gate == .execute else { return }
    } else {
      isBusy = true
    }

    isExecuting = true
    activeJobID = submission.jobID
    defer {
      isExecuting = false
      activeJobID = nil
      activeJobTask = nil
      if waiters.isEmpty {
        isBusy = false
      } else {
        let next = waiters.removeFirst()
        next.continuation.resume(returning: .execute)
      }
    }

    let job = try submission.job.asPayload()
    let profile = profile(for: job)
    let worker = try await ensureWorker(for: profile)
    let task = Task<URL, Error> {
      switch (worker, job) {
      case (.text(let worker), .text(let options)):
        return try await worker.execute(options) { update in
          try? eventSink(.init(type: .progress, jobID: submission.jobID, progress: update))
        }
      case (.control(let worker), .control(let options)):
        return try await worker.execute(options) { update in
          try? eventSink(.init(type: .progress, jobID: submission.jobID, progress: update))
        }
      default:
        throw ServiceTransportError.invalidMessage("Worker profile does not match submitted job")
      }
    }
    activeJobTask = task

    do {
      let outputURL = try await task.value
      residentWorker?.lastUsedAt = Date()
      try eventSink(.init(type: .completed, jobID: submission.jobID, outputPath: outputURL.path))
    } catch is CancellationError {
      try eventSink(.init(type: .cancelled, jobID: submission.jobID, message: "Job cancelled"))
    } catch {
      try eventSink(.init(type: .failed, jobID: submission.jobID, message: CLIErrors.describe(error)))
    }
  }

  func status(socketPath: String) -> ServiceStatusSnapshot {
    ServiceStatusSnapshot(
      socketPath: socketPath,
      residencyPolicy: options.residencyPolicy,
      idleTimeoutSeconds: options.idleTimeoutSeconds,
      isExecuting: isExecuting,
      isShuttingDown: isShuttingDown,
      activeJobID: activeJobID,
      queuedJobIDs: waiters.map(\.jobID),
      residentWorker: residentWorker.map {
        ServiceWorkerSnapshot(
          kind: $0.profile.kind.rawValue,
          model: $0.profile.model,
          weightsVariant: $0.profile.weightsVariant,
          controlnetWeights: $0.profile.controlnetWeights,
          controlnetFile: $0.profile.controlnetFile,
          maxSequenceLength: $0.profile.maxSequenceLength,
          residencyPolicy: $0.profile.residencyPolicy,
          lastUsedAt: $0.lastUsedAt
        )
      }
    )
  }

  fileprivate func cancel(jobID: String) -> CancelDisposition {
    if activeJobID == jobID {
      activeJobTask?.cancel()
      return .active
    }

    guard let index = waiters.firstIndex(where: { $0.jobID == jobID }) else {
      return .notFound
    }

    let queuedSubmission = waiters.remove(at: index)
    do {
      try queuedSubmission.eventSink(.init(type: .cancelled, jobID: jobID, message: "Job cancelled while queued"))
    } catch {
      logger.warning("Failed to notify queued job \(jobID) about cancellation: \(CLIErrors.describe(error))")
    }
    queuedSubmission.continuation.resume(returning: .cancelled)
    if !isExecuting && waiters.isEmpty {
      isBusy = false
    }
    return .queued
  }

  func requestShutdown() -> Bool {
    guard !isExecuting, waiters.isEmpty else { return false }
    isShuttingDown = true
    return true
  }

  func evictIdleWorkerIfNeeded(now: Date) async {
    guard !isExecuting, let residentWorker else { return }
    guard residentWorker.profile.residencyPolicy.keepsHeavyModulesResident else { return }
    guard options.idleTimeoutSeconds > 0 else { return }

    let idleDuration = now.timeIntervalSince(residentWorker.lastUsedAt)
    guard idleDuration >= options.idleTimeoutSeconds else { return }

    logger.info(
      "Evicting idle \(residentWorker.profile.kind.rawValue) worker after \(Int(idleDuration.rounded()))s")
    await residentWorker.worker.unload()
    self.residentWorker = nil
  }

  private func ensureWorker(for profile: WorkerProfile) async throws -> ResidentWorker {
    if let residentWorker {
      if residentWorker.profile == profile {
        if shouldEvictAdaptiveWorker(profile: profile) {
          logger.info("Evicting adaptive worker before reuse due to low available memory")
          await residentWorker.worker.unload()
          self.residentWorker = nil
        } else {
          logger.info("Reusing resident \(profile.kind.rawValue) worker")
          self.residentWorker?.lastUsedAt = Date()
          return residentWorker.worker
        }
      } else {
        logger.info("Switching resident worker profile")
        await residentWorker.worker.unload()
        self.residentWorker = nil
      }
    }

    let worker = try await makeWorker(for: profile)
    residentWorker = .init(profile: profile, worker: worker, lastUsedAt: Date())
    return worker
  }

  private func makeWorker(for profile: WorkerProfile) async throws -> ResidentWorker {
    switch profile.kind {
    case .text:
      let worker = TextServeWorker(profile: profile, logger: logger)
      try await worker.prewarm()
      return .text(worker)
    case .control:
      let worker = ControlServeWorker(profile: profile, logger: logger)
      try await worker.prewarm()
      return .control(worker)
    }
  }

  private func shouldEvictAdaptiveWorker(profile: WorkerProfile) -> Bool {
    guard profile.residencyPolicy == .adaptive else { return false }
    let availableMemory = SystemMemory.availableBytes()
    guard availableMemory > 0 else { return false }
    return availableMemory < Self.adaptiveMinimumAvailableMemoryBytes
  }

  private func warmProfile() -> WorkerProfile? {
    if let warmControlnetWeights = options.warmControlnetWeights {
      return WorkerProfile(
        kind: .control,
        model: options.warmModel,
        weightsVariant: options.warmWeightsVariant,
        controlnetWeights: warmControlnetWeights,
        controlnetFile: options.warmControlnetFile,
        maxSequenceLength: options.warmMaxSequenceLength,
        residencyPolicy: options.residencyPolicy,
        forceTransformerOverrideOnly: false
      )
    }

    guard options.warmModel != nil else { return nil }
    return WorkerProfile(
      kind: .text,
      model: options.warmModel,
      weightsVariant: options.warmWeightsVariant,
      controlnetWeights: nil,
      controlnetFile: nil,
      maxSequenceLength: options.warmMaxSequenceLength,
      residencyPolicy: options.residencyPolicy,
      forceTransformerOverrideOnly: false
    )
  }

  private func profile(for job: GenerationJobPayload) -> WorkerProfile {
    switch job {
    case .text(let options):
      return WorkerProfile(
        kind: .text,
        model: options.model,
        weightsVariant: options.weightsVariant,
        controlnetWeights: nil,
        controlnetFile: nil,
        maxSequenceLength: options.maxSequenceLength ?? 512,
        residencyPolicy: self.options.residencyPolicy,
        forceTransformerOverrideOnly: options.forceTransformerOverrideOnly
      )
    case .control(let options):
      return WorkerProfile(
        kind: .control,
        model: options.model,
        weightsVariant: options.weightsVariant,
        controlnetWeights: options.controlnetWeights,
        controlnetFile: options.controlnetWeightsFile,
        maxSequenceLength: options.maxSequenceLength ?? 512,
        residencyPolicy: self.options.residencyPolicy,
        forceTransformerOverrideOnly: false
      )
    }
  }
}

private actor TextServeWorker {
  private let profile: WorkerProfile
  private let logger: Logger
  private let pipeline: ZImagePipeline

  init(profile: WorkerProfile, logger: Logger) {
    self.profile = profile
    self.logger = logger
    self.pipeline = ZImagePipeline(logger: logger)
  }

  func prewarm() async throws {
    let request = ZImageGenerationRequest(
      prompt: "",
      model: profile.model,
      weightsVariant: profile.weightsVariant,
      maxSequenceLength: profile.maxSequenceLength,
      forceTransformerOverrideOnly: profile.forceTransformerOverrideOnly,
      runtimeOptions: .init(residencyPolicy: profile.residencyPolicy)
    )
    try await pipeline.warm(request)
  }

  func execute(
    _ options: TextGenerationOptions,
    progressSink: @escaping @Sendable (JobProgressUpdate) -> Void
  ) async throws -> URL {
    let plan = CLICommandRunner.buildTextExecutionPlan(
      options,
      logger: logger,
      runtimeOptions: .init(residencyPolicy: profile.residencyPolicy)
    )
    CLICommandRunner.configureCacheLimit(options.cacheLimit, logger: logger)
    return try await pipeline.generate(
      plan.request,
      progressHandler: { progress in
        guard progress.stage == .denoising else { return }
        progressSink(
          JobProgressUpdate(
            stage: progress.stage.rawValue,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            fractionCompleted: progress.fractionCompleted
          ))
      })
  }

  func unload() {
    pipeline.unloadModel()
  }
}

private actor ControlServeWorker {
  private let profile: WorkerProfile
  private let logger: Logger
  private let pipeline: ZImageControlPipeline

  init(profile: WorkerProfile, logger: Logger) {
    self.profile = profile
    self.logger = logger
    self.pipeline = ZImageControlPipeline(logger: logger)
  }

  func prewarm() async throws {
    let request = ZImageControlGenerationRequest(
      prompt: "",
      model: profile.model,
      weightsVariant: profile.weightsVariant,
      controlnetWeights: profile.controlnetWeights,
      controlnetWeightsFile: profile.controlnetFile,
      maxSequenceLength: profile.maxSequenceLength,
      runtimeOptions: .init(residencyPolicy: profile.residencyPolicy)
    )
    try await pipeline.warm(request)
  }

  func execute(
    _ options: ControlGenerationOptions,
    progressSink: @escaping @Sendable (JobProgressUpdate) -> Void
  ) async throws -> URL {
    let plan = try CLICommandRunner.buildControlExecutionPlan(
      options,
      logger: logger,
      runtimeOptions: .init(residencyPolicy: profile.residencyPolicy)
    )
    CLICommandRunner.configureCacheLimit(options.cacheLimit, logger: logger)

    var request = plan.request
    request.progressCallback = { progress in
      if progress.stage == "Denoising" {
        progressSink(
          JobProgressUpdate(
            stage: progress.stage,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            fractionCompleted: progress.fractionCompleted
          ))
      } else if let enhancedPrompt = progress.enhancedPrompt {
        progressSink(
          JobProgressUpdate(
            stage: progress.stage,
            stepIndex: progress.stepIndex,
            totalSteps: progress.totalSteps,
            fractionCompleted: progress.fractionCompleted,
            enhancedPrompt: enhancedPrompt
          ))
      }
    }
    return try await pipeline.generate(request)
  }

  func unload() {
    pipeline.unloadModel()
  }
}

private enum SystemMemory {
  static func availableBytes() -> UInt64 {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    let pageSize = UInt64(sysconf(_SC_PAGESIZE))
    return UInt64(stats.free_count) * pageSize
  }
}
