import Foundation

public enum CapsuleKind: String, CaseIterable, Hashable, Sendable {
    case ct
    case mri
    case mriTracts
    case other

    public static let threeVersionOrder: [CapsuleKind] = [.ct, .mri, .mriTracts]

    public var displayName: String {
        switch self {
        case .ct: "TC / angio-TC"
        case .mri: "RM"
        case .mriTracts: "RM + tratos"
        case .other: "Outro"
        }
    }
}

public struct CapsuleRoot: Hashable, Sendable, Identifiable {
    public let url: URL
    public let label: String

    public var id: String { url.standardizedFileURL.path }

    public init(url: URL, label: String) {
        self.url = url.standardizedFileURL
        self.label = label
    }
}

public struct CapsuleVolume: Equatable, Sendable {
    public let kind: String
    public let label: String?

    public init(kind: String, label: String?) {
        self.kind = kind
        self.label = label
    }
}

public struct CapsuleManifest: Equatable, Sendable {
    public let version: Int
    public let caseLabel: String
    public let studyYear: String?
    public let createdUTC: String?
    public let volumes: [CapsuleVolume]
    public let tractCount: Int
    public let maskCount: Int
    public let tourStepCount: Int

    public init(
        version: Int,
        caseLabel: String,
        studyYear: String? = nil,
        createdUTC: String? = nil,
        volumes: [CapsuleVolume] = [],
        tractCount: Int = 0,
        maskCount: Int = 0,
        tourStepCount: Int = 0
    ) {
        self.version = version
        self.caseLabel = caseLabel
        self.studyYear = studyYear
        self.createdUTC = createdUTC
        self.volumes = volumes
        self.tractCount = tractCount
        self.maskCount = maskCount
        self.tourStepCount = tourStepCount
    }
}

public enum ManifestIssue: String, Error, Equatable, Sendable {
    case fileUnavailable
    case manifestNotFound
    case manifestTooLarge
    case invalidJSON
    case invalidShape

    public var message: String {
        switch self {
        case .fileUnavailable: "Não foi possível ler o arquivo."
        case .manifestNotFound: "Manifesto não encontrado."
        case .manifestTooLarge: "Manifesto não terminou dentro de 1 MiB."
        case .invalidJSON: "JSON do manifesto inválido."
        case .invalidShape: "Campos obrigatórios do manifesto ausentes."
        }
    }
}

public enum CapsuleContents: Equatable, Sendable {
    case readable(CapsuleManifest)
    case unreadable(ManifestIssue)
}

public struct CapsuleInfo: Identifiable, Equatable, Sendable {
    public let url: URL
    public let root: CapsuleRoot
    public let modificationDate: Date?
    public let sizeBytes: Int64?
    public let contents: CapsuleContents

    public var id: String { url.standardizedFileURL.path }
    public var filename: String { url.lastPathComponent }
    public var manifest: CapsuleManifest? {
        guard case let .readable(manifest) = contents else { return nil }
        return manifest
    }
    public var version: Int? { manifest?.version }
    public var kind: CapsuleKind {
        guard let manifest else { return .other }
        return CapsuleLibrary.kind(of: manifest)
    }
    public var isDraft: Bool {
        filename.localizedCaseInsensitiveContains("-draft")
    }
    public var caseLabel: String {
        if let manifest { return manifest.caseLabel }
        let suffix = ".capsule.html"
        if filename.lowercased().hasSuffix(suffix) {
            return String(filename.dropLast(suffix.count))
        }
        return filename
    }

    public init(
        url: URL,
        root: CapsuleRoot,
        modificationDate: Date?,
        sizeBytes: Int64?,
        contents: CapsuleContents
    ) {
        self.url = url.standardizedFileURL
        self.root = root
        self.modificationDate = modificationDate
        self.sizeBytes = sizeBytes
        self.contents = contents
    }
}

public enum RootScanStatus: Equatable, Sendable {
    case scanned
    case missing
    case unavailable
}

public struct RootScanReport: Equatable, Sendable, Identifiable {
    public let root: CapsuleRoot
    public let status: RootScanStatus
    public let capsuleCount: Int

    public var id: String { root.id }

    public init(root: CapsuleRoot, status: RootScanStatus, capsuleCount: Int) {
        self.root = root
        self.status = status
        self.capsuleCount = capsuleCount
    }
}

public struct ScanResult: Equatable, Sendable {
    public let capsules: [CapsuleInfo]
    public let roots: [RootScanReport]

    public init(capsules: [CapsuleInfo], roots: [RootScanReport]) {
        self.capsules = capsules
        self.roots = roots
    }
}
