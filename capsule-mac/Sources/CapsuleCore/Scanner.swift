import Foundation

public enum CapsuleScanner {
    public static let maximumDepth = 4
    private static let skippedDirectoryNames: Set<String> = [
        "archive", ".git", "node_modules", ".venv", "__pycache__"
    ]
    private static let capsuleSuffix = ".capsule.html"

    public static func scan(
        roots: [CapsuleRoot],
        isCancelled: @Sendable () -> Bool = { false }
    ) throws -> ScanResult {
        let fileManager = FileManager.default
        var capsules: [CapsuleInfo] = []
        var reports: [RootScanReport] = []
        var seenFiles: Set<String> = []

        for root in roots {
            try checkCancellation(isCancelled)
            let startIndex = capsules.count

            guard fileManager.fileExists(atPath: root.url.path) else {
                reports.append(RootScanReport(root: root, status: .missing, capsuleCount: 0))
                continue
            }

            do {
                let rootValues = try root.url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                guard rootValues.isSymbolicLink != true, rootValues.isDirectory == true else {
                    reports.append(RootScanReport(root: root, status: .unavailable, capsuleCount: 0))
                    continue
                }
            } catch {
                reports.append(RootScanReport(root: root, status: .unavailable, capsuleCount: 0))
                continue
            }

            var status: RootScanStatus = .scanned
            var pendingDirectories: [(url: URL, depth: Int)] = [(root.url, 0)]

            while let current = pendingDirectories.popLast() {
                try checkCancellation(isCancelled)
                let children: [URL]
                do {
                    children = try fileManager.contentsOfDirectory(
                        at: current.url,
                        includingPropertiesForKeys: [
                            .isDirectoryKey,
                            .isSymbolicLinkKey,
                            .contentModificationDateKey,
                            .fileSizeKey
                        ],
                        options: []
                    ).sorted { $0.lastPathComponent < $1.lastPathComponent }
                } catch {
                    status = .unavailable
                    continue
                }

                for child in children {
                    try checkCancellation(isCancelled)
                    let values: URLResourceValues
                    do {
                        values = try child.resourceValues(forKeys: [
                            .isDirectoryKey,
                            .isSymbolicLinkKey,
                            .contentModificationDateKey,
                            .fileSizeKey
                        ])
                    } catch {
                        if child.lastPathComponent.lowercased().hasSuffix(capsuleSuffix),
                           seenFiles.insert(child.standardizedFileURL.path).inserted {
                            capsules.append(makeUnreadableInfo(
                                url: child,
                                root: root,
                                issue: .fileUnavailable,
                                modificationDate: nil,
                                sizeBytes: nil
                            ))
                        }
                        continue
                    }

                    guard values.isSymbolicLink != true else { continue }
                    if values.isDirectory == true {
                        guard !skippedDirectoryNames.contains(child.lastPathComponent),
                              current.depth < maximumDepth else { continue }
                        pendingDirectories.append((child, current.depth + 1))
                        continue
                    }

                    guard child.lastPathComponent.lowercased().hasSuffix(capsuleSuffix),
                          seenFiles.insert(child.standardizedFileURL.path).inserted else { continue }

                    let modificationDate = values.contentModificationDate
                    let sizeBytes = values.fileSize.map(Int64.init)
                    let contents: CapsuleContents
                    do {
                        contents = .readable(try ManifestReader.read(from: child))
                    } catch let issue as ManifestIssue {
                        contents = .unreadable(issue)
                    } catch {
                        contents = .unreadable(.fileUnavailable)
                    }
                    capsules.append(CapsuleInfo(
                        url: child,
                        root: root,
                        modificationDate: modificationDate,
                        sizeBytes: sizeBytes,
                        contents: contents
                    ))
                }
            }

            reports.append(RootScanReport(
                root: root,
                status: status,
                capsuleCount: capsules.count - startIndex
            ))
        }

        return ScanResult(capsules: capsules, roots: reports)
    }

    private static func checkCancellation(_ isCancelled: @Sendable () -> Bool) throws {
        if isCancelled() { throw CancellationError() }
    }

    private static func makeUnreadableInfo(
        url: URL,
        root: CapsuleRoot,
        issue: ManifestIssue,
        modificationDate: Date?,
        sizeBytes: Int64?
    ) -> CapsuleInfo {
        CapsuleInfo(
            url: url,
            root: root,
            modificationDate: modificationDate,
            sizeBytes: sizeBytes,
            contents: .unreadable(issue)
        )
    }
}
