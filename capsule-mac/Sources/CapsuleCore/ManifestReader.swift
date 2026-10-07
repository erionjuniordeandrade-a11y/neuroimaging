import Foundation

public enum ManifestReader {
    public static let initialByteLimit = 64 * 1024
    public static let maximumByteLimit = 1024 * 1024

    private static let openingTag = Data(
        #"<script id="capsule-manifest" type="application/json">"#.utf8
    )
    private static let closingTag = Data("</script>".utf8)

    public static func read(from url: URL) throws -> CapsuleManifest {
        do {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }

            let first = try handle.read(upToCount: initialByteLimit) ?? Data()
            switch extractJSON(from: first) {
            case let .found(json):
                return try decode(json)
            case .missing, .incomplete:
                let remainingLimit = maximumByteLimit - first.count
                let remainder = try handle.read(upToCount: remainingLimit) ?? Data()
                let prefix = first + remainder
                switch extractJSON(from: prefix) {
                case let .found(json):
                    return try decode(json)
                case .missing:
                    throw ManifestIssue.manifestNotFound
                case .incomplete:
                    throw ManifestIssue.manifestTooLarge
                }
            }
        } catch let issue as ManifestIssue {
            throw issue
        } catch {
            throw ManifestIssue.fileUnavailable
        }
    }

    public static func parse(prefix: Data) throws -> CapsuleManifest {
        switch extractJSON(from: prefix) {
        case let .found(json):
            return try decode(json)
        case .missing:
            throw ManifestIssue.manifestNotFound
        case .incomplete:
            throw ManifestIssue.manifestTooLarge
        }
    }

    private enum Extraction {
        case found(Data)
        case missing
        case incomplete
    }

    private static func extractJSON(from data: Data) -> Extraction {
        guard let opening = data.range(of: openingTag) else { return .missing }
        let bodyStart = opening.upperBound
        guard let closing = data[bodyStart..<data.endIndex].range(of: closingTag) else {
            return .incomplete
        }
        return .found(data.subdata(in: bodyStart..<closing.lowerBound))
    }

    private static func decode(_ data: Data) throws -> CapsuleManifest {
        do {
            _ = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw ManifestIssue.invalidJSON
        }

        do {
            let wire = try JSONDecoder().decode(ManifestPayload.self, from: data)
            guard !wire.caseInfo.label.isEmpty else { throw ManifestIssue.invalidShape }

            let volumes = (wire.volumes ?? []).compactMap { value -> CapsuleVolume? in
                guard case let .object(fields) = value,
                      let kind = fields["kind"]?.stringValue else { return nil }
                return CapsuleVolume(kind: kind, label: fields["label"]?.stringValue)
            }

            return CapsuleManifest(
                version: wire.version,
                caseLabel: wire.caseInfo.label,
                studyYear: wire.caseInfo.studyYear?.scalarString,
                createdUTC: wire.createdUTC?.stringValue,
                volumes: volumes,
                tractCount: wire.tracts?.arrayCount ?? 0,
                maskCount: wire.masks?.arrayCount ?? 0,
                tourStepCount: wire.tour?.tourStepCount ?? 0
            )
        } catch let issue as ManifestIssue {
            throw issue
        } catch {
            throw ManifestIssue.invalidShape
        }
    }
}

private struct ManifestPayload: Decodable {
    let version: Int
    let caseInfo: CasePayload
    let createdUTC: ManifestJSONValue?
    let volumes: [ManifestJSONValue]?
    let tracts: ManifestJSONValue?
    let masks: ManifestJSONValue?
    let tour: ManifestJSONValue?

    enum CodingKeys: String, CodingKey {
        case version
        case caseInfo = "case"
        case createdUTC = "created_utc"
        case volumes
        case tracts
        case masks
        case tour
    }
}

private struct CasePayload: Decodable {
    let label: String
    let studyYear: ManifestJSONValue?

    enum CodingKeys: String, CodingKey {
        case label
        case studyYear = "study_year"
    }
}

private indirect enum ManifestJSONValue: Decodable {
    case object([String: ManifestJSONValue])
    case array([ManifestJSONValue])
    case string(String)
    case number(String)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        if let values = try? [String: ManifestJSONValue](from: decoder) {
            self = .object(values)
            return
        }
        if let values = try? [ManifestJSONValue](from: decoder) {
            self = .array(values)
            return
        }

        let value = try decoder.singleValueContainer()
        if value.decodeNil() {
            self = .null
        } else if let string = try? value.decode(String.self) {
            self = .string(string)
        } else if let bool = try? value.decode(Bool.self) {
            self = .bool(bool)
        } else if let integer = try? value.decode(Int64.self) {
            self = .number(String(integer))
        } else if let double = try? value.decode(Double.self) {
            self = .number(String(double))
        } else {
            throw DecodingError.dataCorruptedError(
                in: value,
                debugDescription: "Unsupported manifest value"
            )
        }
    }

    var stringValue: String? {
        guard case let .string(value) = self else { return nil }
        return value
    }

    var scalarString: String? {
        switch self {
        case let .string(value), let .number(value): value
        case .object, .array, .bool, .null: nil
        }
    }

    var arrayCount: Int? {
        guard case let .array(values) = self else { return nil }
        return values.count
    }

    var tourStepCount: Int {
        switch self {
        case let .array(values): return values.count
        case let .object(values):
            guard let steps = values["steps"] else { return 0 }
            if let count = steps.arrayCount { return count }
            if case let .object(stepMap) = steps { return stepMap.count }
            return 0
        case .string, .number, .bool, .null:
            return 0
        }
    }
}
