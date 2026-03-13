import XCTest
@testable import ZImageCLICommon

final class MarkdownCommandExtractorTests: XCTestCase {
  func testExtractsServeInvocationFromMarkdownFence() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        # Prompt draft

        ```bash
        ZImageServe --prompt "a mountain lake" --model /tmp/model --output lake.png
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)
    XCTAssertEqual(submissions[0].jobID, "markdown-1")

    guard case .text(let options) = submissions[0].job else {
      return XCTFail("Expected text job")
    }
    XCTAssertEqual(options.prompt, "a mountain lake")
    XCTAssertEqual(options.model, "/tmp/model")
    XCTAssertEqual(options.outputPath, "lake.png")
  }

  func testExtractsCLIControlInvocationFromMarkdownFence() throws {
    let submissions = try MarkdownCommandExtractor.submissions(
      from:
        """
        ```zsh
        ZImageCLI control --prompt "a dancer" --control-image pose.png --controlnet-weights ./controlnet --output dancer.png
        ```
        """
    )

    XCTAssertEqual(submissions.count, 1)

    guard case .control(let options) = submissions[0].job else {
      return XCTFail("Expected control job")
    }
    XCTAssertEqual(options.prompt, "a dancer")
    XCTAssertEqual(options.controlImage, "pose.png")
    XCTAssertEqual(options.controlnetWeights, "./controlnet")
    XCTAssertEqual(options.outputPath, "dancer.png")
  }

  func testRejectsShellControlOperators() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake" | tee out.txt
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("Shell control operators"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testRejectsMultipleCommandsInSingleFence() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake"
          ZImageServe --prompt "a forest"
          ```
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("exactly one command"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }

  func testRejectsUnterminatedFence() throws {
    XCTAssertThrowsError(
      try MarkdownCommandExtractor.submissions(
        from:
          """
          ```bash
          ZImageServe --prompt "a lake"
          """
      )
    ) { error in
      guard let cliError = error as? CLIError else {
        return XCTFail("Expected CLIError")
      }
      XCTAssertTrue(cliError.message.contains("Unterminated fenced shell block"))
      XCTAssertEqual(cliError.usage, .markdown)
    }
  }
}
