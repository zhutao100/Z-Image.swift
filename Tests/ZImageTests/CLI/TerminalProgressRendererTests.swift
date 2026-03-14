import XCTest
@testable import ZImageCLICommon

final class TerminalProgressRendererTests: XCTestCase {
  func testDisplayProgressUsesProgressUpdateTotalSteps() {
    let display = DisplayProgress(
      update: JobProgressUpdate(
        stage: "Denoising",
        stepIndex: 0,
        totalSteps: 8,
        fractionCompleted: 0
      ))

    XCTAssertEqual(display?.total, 8)
    XCTAssertEqual(display?.completed, 0)
  }

  func testDisplayProgressClampsCompletedToProgressUpdateTotalSteps() {
    let display = DisplayProgress(
      update: JobProgressUpdate(
        stage: "Denoising",
        stepIndex: 12,
        totalSteps: 8,
        fractionCompleted: 1
      ))

    XCTAssertEqual(display?.total, 8)
    XCTAssertEqual(display?.completed, 8)
  }

  func testDisplayProgressRejectsNonPositiveTotals() {
    XCTAssertNil(
      DisplayProgress(
        update: JobProgressUpdate(
          stage: "Denoising",
          stepIndex: 0,
          totalSteps: 0,
          fractionCompleted: 0
        )))
  }
}
