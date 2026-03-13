import Foundation

public enum ShellWordsLexer {
  private enum QuoteMode {
    case none
    case single
    case double
  }

  public static func lexSingleCommand(_ script: String, usage: CLIUsageTopic = .markdown) throws -> [String] {
    var tokens: [String] = []
    var current = ""
    var mode: QuoteMode = .none
    var index = script.startIndex
    var pendingCommandBreak = false

    func finishToken() {
      guard !current.isEmpty else { return }
      tokens.append(current)
      current.removeAll(keepingCapacity: true)
    }

    while index < script.endIndex {
      let character = script[index]

      switch mode {
      case .single:
        if character == "'" {
          mode = .none
        } else {
          current.append(character)
        }
        index = script.index(after: index)

      case .double:
        if character == "\"" {
          mode = .none
          index = script.index(after: index)
          continue
        }
        if character == "\\" {
          let nextIndex = script.index(after: index)
          guard nextIndex < script.endIndex else {
            throw CLIError(message: "Unterminated escape sequence in markdown command", usage: usage)
          }
          let escaped = script[nextIndex]
          if escaped == "\n" {
            index = script.index(after: nextIndex)
            continue
          }
          current.append(escaped)
          index = script.index(after: nextIndex)
          continue
        }
        if character == "`" {
          throw CLIError(message: "Command substitution is not allowed in markdown commands", usage: usage)
        }
        current.append(character)
        index = script.index(after: index)

      case .none:
        if character == "\\" {
          let nextIndex = script.index(after: index)
          guard nextIndex < script.endIndex else {
            throw CLIError(message: "Unterminated escape sequence in markdown command", usage: usage)
          }
          let escaped = script[nextIndex]
          if escaped == "\n" {
            index = script.index(after: nextIndex)
            continue
          }
          current.append(escaped)
          index = script.index(after: nextIndex)
          continue
        }

        if character == "'" {
          mode = .single
          pendingCommandBreak = false
          index = script.index(after: index)
          continue
        }
        if character == "\"" {
          mode = .double
          pendingCommandBreak = false
          index = script.index(after: index)
          continue
        }
        if character == "`" {
          throw CLIError(message: "Command substitution is not allowed in markdown commands", usage: usage)
        }
        if character == "$" {
          throw CLIError(message: "Shell expansion is not supported in markdown commands", usage: usage)
        }
        if character == "|" || character == ";" || character == ">" || character == "<"
          || character == "&" || character == "(" || character == ")"
        {
          throw CLIError(message: "Shell control operators are not allowed in markdown commands", usage: usage)
        }
        if character == "#" && current.isEmpty {
          while index < script.endIndex, script[index] != "\n" {
            index = script.index(after: index)
          }
          continue
        }
        if character.isWhitespace {
          finishToken()
          if character == "\n" || character == "\r" {
            if !tokens.isEmpty {
              pendingCommandBreak = true
            }
          }
          index = script.index(after: index)
          continue
        }
        if pendingCommandBreak {
          throw CLIError(message: "Each markdown fence must contain exactly one command", usage: usage)
        }
        current.append(character)
        index = script.index(after: index)
      }
    }

    guard mode == .none else {
      throw CLIError(message: "Unterminated quote in markdown command", usage: usage)
    }

    finishToken()
    return tokens
  }
}
