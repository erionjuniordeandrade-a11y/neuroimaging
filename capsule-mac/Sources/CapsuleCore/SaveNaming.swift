import Foundation

public enum SaveNaming {
    private static let capsuleSuffix = ".capsule.html"

    public static func destination(
        source: URL,
        suggestedFilename: String,
        reserved: Set<URL> = [],
        fileManager: FileManager = .default
    ) -> URL {
        let directory = source.deletingLastPathComponent().standardizedFileURL
        let safeName = sanitized(suggestedFilename)
        var candidate = directory.appendingPathComponent(safeName).standardizedFileURL
        var collision = 2

        while fileManager.fileExists(atPath: candidate.path)
                || reserved.contains(candidate.standardizedFileURL) {
            candidate = directory.appendingPathComponent(
                collisionName(for: safeName, number: collision)
            ).standardizedFileURL
            collision += 1
        }

        return candidate
    }

    private static func sanitized(_ suggestion: String) -> String {
        let withoutSeparators = suggestion
            .replacingOccurrences(of: "/", with: "")
            .replacingOccurrences(of: "..", with: "")
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return withoutSeparators.isEmpty || withoutSeparators == "."
            ? "download"
            : withoutSeparators
    }

    private static func collisionName(for filename: String, number: Int) -> String {
        if filename.lowercased().hasSuffix(capsuleSuffix) {
            let stem = String(filename.dropLast(capsuleSuffix.count))
            return "\(stem) (\(number))\(capsuleSuffix)"
        }

        if let dot = filename.lastIndex(of: "."), dot != filename.startIndex {
            let stem = String(filename[..<dot])
            let suffix = String(filename[dot...])
            return "\(stem) (\(number))\(suffix)"
        }
        return "\(filename) (\(number))"
    }
}
