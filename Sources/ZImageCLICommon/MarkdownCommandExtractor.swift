import Foundation

public enum MarkdownCommandExtractor {
  private static let acceptedLanguages: Set<String> = ["bash", "sh", "zsh"]

  public static func submissions(fromPath path: String) throws -> [BatchSubmission] {
    let expandedPath = NSString(string: path).expandingTildeInPath
    let url = URL(fileURLWithPath: expandedPath)
    let markdown = try String(contentsOf: url, encoding: .utf8)
    return try submissions(from: markdown)
  }

  public static func submissions(from markdown: String) throws -> [BatchSubmission] {
    var submissions: [BatchSubmission] = []
    var currentLanguage: String?
    var currentLines: [String] = []

    for rawLine in markdown.split(separator: "\n", omittingEmptySubsequences: false) {
      let line = String(rawLine)
      let trimmed = line.trimmingCharacters(in: .whitespaces)

      if currentLanguage == nil {
        guard trimmed.hasPrefix("```") else { continue }
        let info = trimmed.dropFirst(3).trimmingCharacters(in: .whitespacesAndNewlines)
        let language = info.split(separator: " ").first.map(String.init)?.lowercased()
        if let language, acceptedLanguages.contains(language) {
          currentLanguage = language
          currentLines.removeAll(keepingCapacity: true)
        }
        continue
      }

      if trimmed.hasPrefix("```") {
        let block = currentLines.joined(separator: "\n")
        if block.contains("ZImageCLI") || block.contains("ZImageServe") {
          let tokens = try ShellWordsLexer.lexSingleCommand(block)
          if !tokens.isEmpty, GenerationJobInvocationParser.supportsProgram(tokens[0]) {
            submissions.append(
              BatchSubmission(
                jobID: "markdown-\(submissions.count + 1)",
                job: try GenerationJobInvocationParser.parse(tokens: tokens, usage: .markdown)
              ))
          }
        }
        currentLanguage = nil
        currentLines.removeAll(keepingCapacity: true)
        continue
      }

      currentLines.append(line)
    }

    if currentLanguage != nil {
      throw CLIError(message: "Unterminated fenced shell block in markdown file", usage: .markdown)
    }

    guard !submissions.isEmpty else {
      throw CLIError(
        message: "No fenced ZImageCLI or ZImageServe generation commands were found in the markdown file",
        usage: .markdown
      )
    }
    return submissions
  }
}
