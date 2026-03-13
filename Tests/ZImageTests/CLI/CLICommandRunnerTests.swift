import XCTest
@testable import ZImage
@testable import ZImageCLICommon

final class CLICommandRunnerTests: XCTestCase {
  func testDistillAdapterWarningUsesKnownRecipe() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .huggingFace(
        "alibaba-pai/Z-Image-Fun-Lora-Distill",
        filename: "Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
        scale: 1.0
      ),
      steps: nil,
      guidance: nil,
      loraScale: 1.0,
      preset: .zImage
    )

    XCTAssertNotNil(warning)
    XCTAssertTrue(warning?.contains("--steps 8 --guidance 1.0 --lora-scale 0.8") == true)
    XCTAssertTrue(warning?.contains("does not override") == true)
  }

  func testDistillAdapterWarningClearsWhenRecipeAlreadyMatches() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .local(
        "/tmp/Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
        scale: 0.8
      ),
      steps: 8,
      guidance: 1.0,
      loraScale: 0.8,
      preset: .zImage
    )

    XCTAssertNil(warning)
  }

  func testGenericLoRAWarningStillUsesModelDefaultsMessage() {
    let warning = CLICommandRunner.loraSamplingWarning(
      loraConfig: .huggingFace("ostris/z_image_turbo_childrens_drawings"),
      steps: nil,
      guidance: nil,
      loraScale: 1.0,
      preset: .zImageTurbo
    )

    XCTAssertNotNil(warning)
    XCTAssertTrue(warning?.contains("Using model defaults with LoRA") == true)
  }
}
