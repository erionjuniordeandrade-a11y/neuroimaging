import Foundation

public struct FamilyKey: Hashable, Sendable {
    public let directoryPath: String
    public let stem: String

    public init(directoryPath: String, stem: String) {
        self.directoryPath = directoryPath
        self.stem = stem
    }
}

public struct CapsuleFamily: Identifiable, Equatable, Sendable {
    public let key: FamilyKey
    public let newest: CapsuleInfo
    public let older: [CapsuleInfo]

    public var id: FamilyKey { key }

    public init(key: FamilyKey, newest: CapsuleInfo, older: [CapsuleInfo]) {
        self.key = key
        self.newest = newest
        self.older = older
    }
}

public struct ThreeVersions: Equatable, Sendable {
    public let ct: CapsuleInfo?
    public let mri: CapsuleInfo?
    public let mriTracts: CapsuleInfo?

    public subscript(kind: CapsuleKind) -> CapsuleInfo? {
        switch kind {
        case .ct: ct
        case .mri: mri
        case .mriTracts: mriTracts
        case .other: nil
        }
    }
}

public enum LibraryScope: Hashable, Sendable {
    case threeVersions
    case kind(CapsuleKind)
    case all
    case drafts
}

public enum CapsuleLibrary {
    private static let capsuleSuffix = ".capsule.html"

    public static func kind(of manifest: CapsuleManifest) -> CapsuleKind {
        if manifest.tractCount > 0 { return .mriTracts }
        if manifest.volumes.contains(where: { $0.kind.caseInsensitiveCompare("CT") == .orderedSame }) {
            return .ct
        }
        if manifest.volumes.contains(where: { $0.kind.caseInsensitiveCompare("MR") == .orderedSame }) {
            return .mri
        }
        return .other
    }

    public static func familyKey(for capsule: CapsuleInfo) -> FamilyKey {
        FamilyKey(
            directoryPath: capsule.url.deletingLastPathComponent().standardizedFileURL.path,
            stem: familyStem(capsule.filename)
        )
    }

    public static func familyStem(_ filename: String) -> String {
        var stem = filename
        if stem.lowercased().hasSuffix(capsuleSuffix) {
            stem = String(stem.dropLast(capsuleSuffix.count))
        }

        while let separator = stem.lastIndex(of: ".") {
            let segment = String(stem[stem.index(after: separator)...])
            guard isRemovableTrailingSegment(segment) else { break }
            stem = String(stem[..<separator])
        }
        return stem
    }

    public static func isDraft(filename: String) -> Bool {
        filename.localizedCaseInsensitiveContains("-draft")
    }

    public static func newest(in capsules: [CapsuleInfo]) -> CapsuleInfo? {
        capsules.sorted(by: isNewer).first
    }

    public static func threeVersions(in capsules: [CapsuleInfo]) -> ThreeVersions {
        let eligible = capsules.filter { $0.manifest != nil && !$0.isDraft }
        return ThreeVersions(
            ct: newest(in: eligible.filter { $0.kind == .ct }),
            mri: newest(in: eligible.filter { $0.kind == .mri }),
            mriTracts: newest(in: eligible.filter { $0.kind == .mriTracts })
        )
    }

    public static func families(
        in capsules: [CapsuleInfo],
        scope: LibraryScope,
        searchText: String = ""
    ) -> [CapsuleFamily] {
        let scoped: [CapsuleInfo]
        switch scope {
        case .threeVersions:
            let selected = threeVersions(in: capsules)
            scoped = CapsuleKind.threeVersionOrder.compactMap { selected[$0] }
        case let .kind(kind):
            scoped = capsules.filter { $0.kind == kind }
        case .all:
            scoped = capsules
        case .drafts:
            scoped = capsules.filter(\.isDraft)
        }

        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return Dictionary(grouping: scoped, by: familyKey(for:))
            .compactMap { key, members in
                let ordered = members.sorted(by: isNewer)
                guard let first = ordered.first else { return nil }
                let family = CapsuleFamily(key: key, newest: first, older: Array(ordered.dropFirst()))
                guard query.isEmpty || familyContains(family, query: query) else { return nil }
                return family
            }
            .sorted(by: { isNewer($0.newest, $1.newest) })
    }

    private static func familyContains(_ family: CapsuleFamily, query: String) -> Bool {
        ([family.newest] + family.older).contains { capsule in
            capsule.caseLabel.localizedStandardContains(query)
                || capsule.filename.localizedStandardContains(query)
        }
    }

    private static func isNewer(_ lhs: CapsuleInfo, _ rhs: CapsuleInfo) -> Bool {
        let leftDate = lhs.modificationDate ?? .distantPast
        let rightDate = rhs.modificationDate ?? .distantPast
        if leftDate != rightDate { return leftDate > rightDate }

        let leftVersion = lhs.version ?? Int.min
        let rightVersion = rhs.version ?? Int.min
        if leftVersion != rightVersion { return leftVersion > rightVersion }
        return lhs.id < rhs.id
    }

    private static func isRemovableTrailingSegment(_ segment: String) -> Bool {
        if segment.count == 8 && segment.allSatisfy(\.isNumber) { return true }

        if segment.first == "v" {
            let digits = segment.dropFirst()
            return !digits.isEmpty && digits.allSatisfy(\.isNumber)
        }

        guard segment.first == "r" else { return false }
        let suffix = segment.dropFirst()
        let digits = suffix.prefix(while: \.isNumber)
        guard !digits.isEmpty else { return false }
        return suffix.dropFirst(digits.count).allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }
    }
}
