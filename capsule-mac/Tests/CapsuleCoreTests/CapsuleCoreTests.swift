import CapsuleCore
import Foundation
import XCTest

final class CapsuleCoreTests: XCTestCase {
    func testManifestReaderHandlesSmallAndLargeOffsetsAndObjectTour() throws {
        let directory = try temporaryDirectory()
        let small = directory.appendingPathComponent("small.capsule.html")
        let large = directory.appendingPathComponent("large.capsule.html")
        try (Data(repeating: 0x61, count: 128) + manifestHTML(tour: ["steps": [["id": 1], ["id": 2], ["id": 3]]]))
            .write(to: small)
        try (Data(repeating: 0x61, count: 80 * 1024) + manifestHTML(tour: ["steps": [["id": 1], ["id": 2]]]))
            .write(to: large)

        let smallManifest = try ManifestReader.read(from: small)
        let largeManifest = try ManifestReader.read(from: large)
        XCTAssertEqual(smallManifest.caseLabel, "synthetic")
        XCTAssertEqual(smallManifest.studyYear, "2024")
        XCTAssertEqual(smallManifest.tourStepCount, 3)
        XCTAssertEqual(largeManifest.tourStepCount, 2)
    }

    func testManifestReaderStopsAtOneMiBWhenClosingTagIsMissing() throws {
        let file = try temporaryDirectory().appendingPathComponent("incomplete.capsule.html")
        let opening = Data(#"<script id="capsule-manifest" type="application/json">{}"#.utf8)
        let bytes = opening + Data(repeating: 0x61, count: ManifestReader.maximumByteLimit)
        try bytes.write(to: file)

        XCTAssertThrowsError(try ManifestReader.read(from: file)) { error in
            XCTAssertEqual(error as? ManifestIssue, .manifestTooLarge)
        }
    }

    func testMissingManifestIsUnreadable() throws {
        let file = try temporaryDirectory().appendingPathComponent("not-a-capsule.capsule.html")
        try Data("<html></html>".utf8).write(to: file)

        XCTAssertThrowsError(try ManifestReader.read(from: file)) { error in
            XCTAssertEqual(error as? ManifestIssue, .manifestNotFound)
        }
    }

    func testKindClassificationUsesTractsThenCTThenMRThenOther() {
        XCTAssertEqual(CapsuleLibrary.kind(of: manifest(tractCount: 1, volumeKinds: ["CT"])), .mriTracts)
        XCTAssertEqual(CapsuleLibrary.kind(of: manifest(volumeKinds: ["CT", "MR"])), .ct)
        XCTAssertEqual(CapsuleLibrary.kind(of: manifest(volumeKinds: ["MR"])), .mri)
        XCTAssertEqual(CapsuleLibrary.kind(of: manifest()), .other)
    }

    func testFamilyStemRemovesRepeatedTrailingSegmentsFromKnownShapes() {
        XCTAssertEqual(
            CapsuleLibrary.familyStem("vs-seg-crop05.v2.r11.20260930.capsule.html"),
            "vs-seg-crop05"
        )
        XCTAssertEqual(
            CapsuleLibrary.familyStem("leipzig-t1-tracts.v2.r12-tour-draft.20260930.capsule.html"),
            "leipzig-t1-tracts"
        )
        XCTAssertEqual(CapsuleLibrary.familyStem("public-ct.capsule.html"), "public-ct")
        XCTAssertEqual(CapsuleLibrary.familyStem("cta.capsule.html"), "cta")
    }

    func testFamilyKeySeparatesContainingDirectories() throws {
        let base = try temporaryDirectory()
        let firstDirectory = base.appendingPathComponent("one", isDirectory: true)
        let secondDirectory = base.appendingPathComponent("two", isDirectory: true)
        try FileManager.default.createDirectory(at: firstDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: secondDirectory, withIntermediateDirectories: true)
        let root = CapsuleRoot(url: base, label: "synthetic")
        let first = capsuleInfo(
            name: "same.v2.capsule.html",
            directory: firstDirectory,
            root: root,
            manifest: manifest()
        )
        let second = capsuleInfo(
            name: "same.v3.capsule.html",
            directory: secondDirectory,
            root: root,
            manifest: manifest(version: 3)
        )

        XCTAssertNotEqual(
            CapsuleLibrary.familyKey(for: first),
            CapsuleLibrary.familyKey(for: second)
        )
    }

    func testDraftDetectionIgnoresCase() throws {
        let directory = try temporaryDirectory()
        let root = CapsuleRoot(url: directory, label: "synthetic")
        let draft = capsuleInfo(
            name: "case-DRAFT.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest()
        )
        XCTAssertTrue(draft.isDraft)
        XCTAssertTrue(CapsuleLibrary.isDraft(filename: draft.filename))
    }

    func testNewestUsesModificationDateThenManifestVersion() throws {
        let directory = try temporaryDirectory()
        let root = CapsuleRoot(url: directory, label: "synthetic")
        let date = Date(timeIntervalSince1970: 1_800_000_000)
        let older = capsuleInfo(
            name: "case.v4.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest(version: 4),
            modificationDate: date
        )
        let newerVersion = capsuleInfo(
            name: "case.v5.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest(version: 5),
            modificationDate: date
        )
        let latestDate = capsuleInfo(
            name: "case.v1.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest(version: 1),
            modificationDate: date.addingTimeInterval(1)
        )

        XCTAssertEqual(CapsuleLibrary.newest(in: [older, newerVersion])?.version, 5)
        XCTAssertEqual(CapsuleLibrary.newest(in: [newerVersion, latestDate])?.version, 1)
    }

    func testThreeVersionsExcludesDraftsAndLeavesMissingKindNil() throws {
        let directory = try temporaryDirectory()
        let root = CapsuleRoot(url: directory, label: "synthetic")
        let date = Date(timeIntervalSince1970: 1_800_000_000)
        let ct = capsuleInfo(
            name: "ct.v1.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest(version: 1, volumeKinds: ["CT"]),
            modificationDate: date
        )
        let draftMRI = capsuleInfo(
            name: "mri-draft.v9.capsule.html",
            directory: directory,
            root: root,
            manifest: manifest(version: 9, volumeKinds: ["MR"]),
            modificationDate: date.addingTimeInterval(1)
        )

        let selection = CapsuleLibrary.threeVersions(in: [ct, draftMRI])
        XCTAssertEqual(selection.ct?.id, ct.id)
        XCTAssertNil(selection.mri)
        XCTAssertNil(selection.mriTracts)
    }

    func testSaveNamingAddsTwoAndThreeAndStripsTraversal() throws {
        let directory = try temporaryDirectory()
        let source = directory.appendingPathComponent("source.capsule.html")
        try Data().write(to: directory.appendingPathComponent("copy.capsule.html"))
        try Data().write(to: directory.appendingPathComponent("copy (2).capsule.html"))

        let collision = SaveNaming.destination(source: source, suggestedFilename: "copy.capsule.html")
        XCTAssertEqual(collision.lastPathComponent, "copy (3).capsule.html")
        XCTAssertEqual(collision.deletingLastPathComponent().standardizedFileURL, directory.standardizedFileURL)

        let traversing = SaveNaming.destination(source: source, suggestedFilename: "../../outside.capsule.html")
        XCTAssertEqual(traversing.lastPathComponent, "outside.capsule.html")
        XCTAssertEqual(traversing.deletingLastPathComponent().standardizedFileURL, directory.standardizedFileURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: traversing.path))
    }

    func testScannerSkipsArchiveAndSymlinksAndHonorsDepthAndMissingRoots() throws {
        let base = try temporaryDirectory()
        let rootURL = base.appendingPathComponent("root", isDirectory: true)
        let root = CapsuleRoot(url: rootURL, label: "synthetic root")
        try FileManager.default.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try writeCapsule("root.capsule.html", in: rootURL)
        try writeCapsule("skip.capsule.html", in: rootURL.appendingPathComponent("archive", isDirectory: true))
        try writeCapsule("skip.capsule.html", in: rootURL.appendingPathComponent(".git", isDirectory: true))

        let depthOne = rootURL.appendingPathComponent("one", isDirectory: true)
        try writeCapsule("depth-one.capsule.html", in: depthOne)
        var depthFour = depthOne
        for name in ["two", "three", "four"] {
            depthFour.appendPathComponent(name, isDirectory: true)
        }
        try writeCapsule("depth-four.capsule.html", in: depthFour)
        let depthFive = depthFour.appendingPathComponent("five", isDirectory: true)
        try writeCapsule("depth-five.capsule.html", in: depthFive)

        let target = base.appendingPathComponent("linked", isDirectory: true)
        try writeCapsule("linked.capsule.html", in: target)
        let link = rootURL.appendingPathComponent("linked-dir", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: target)

        let missing = CapsuleRoot(url: base.appendingPathComponent("missing"), label: "missing")
        let result = try CapsuleScanner.scan(roots: [root, missing])
        let names = Set(result.capsules.map(\.filename))

        XCTAssertEqual(names, ["root.capsule.html", "depth-one.capsule.html", "depth-four.capsule.html"])
        XCTAssertEqual(result.roots.first?.status, .scanned)
        XCTAssertEqual(result.roots.first?.capsuleCount, 3)
        XCTAssertEqual(result.roots.last?.status, .missing)
    }

    func testScannerListsUnreadableManifestWithReason() throws {
        let rootURL = try temporaryDirectory()
        let root = CapsuleRoot(url: rootURL, label: "synthetic")
        try Data("<html></html>".utf8).write(to: rootURL.appendingPathComponent("bad.capsule.html"))

        let result = try CapsuleScanner.scan(roots: [root])
        XCTAssertEqual(result.capsules.count, 1)
        XCTAssertEqual(result.capsules.first?.contents, .unreadable(.manifestNotFound))
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("CapsuleCoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func manifestHTML(tour: Any = [Any]()) -> Data {
        let object: [String: Any] = [
            "version": 2,
            "case": ["label": "synthetic", "study_year": 2024],
            "created_utc": "2026-10-01T00:00:00Z",
            "volumes": [["kind": "MR", "label": "volume"]],
            "tracts": NSNull(),
            "masks": [["id": "mask"]],
            "tour": tour
        ]
        let json = try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return Data(#"<script id="capsule-manifest" type="application/json">"#.utf8)
            + json
            + Data("</script>".utf8)
    }

    private func writeCapsule(_ name: String, in directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try manifestHTML().write(to: directory.appendingPathComponent(name))
    }

    private func manifest(
        version: Int = 1,
        tractCount: Int = 0,
        volumeKinds: [String] = []
    ) -> CapsuleManifest {
        CapsuleManifest(
            version: version,
            caseLabel: "synthetic",
            volumes: volumeKinds.map { CapsuleVolume(kind: $0, label: nil) },
            tractCount: tractCount
        )
    }

    private func capsuleInfo(
        name: String,
        directory: URL,
        root: CapsuleRoot,
        manifest: CapsuleManifest,
        modificationDate: Date? = nil
    ) -> CapsuleInfo {
        CapsuleInfo(
            url: directory.appendingPathComponent(name),
            root: root,
            modificationDate: modificationDate,
            sizeBytes: 1,
            contents: .readable(manifest)
        )
    }
}
