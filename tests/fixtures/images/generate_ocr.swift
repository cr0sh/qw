// Uses only macOS AppKit. Run: swift tests/fixtures/images/generate_ocr.swift
import AppKit
import Foundation

let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json"))) as! [String: Any]
for fixture in manifest["fixtures"] as! [[String: Any]] where fixture["kind"] as? String == "generated_ocr" {
    let dimensions = fixture["dimensions"] as! [Int]
    let width = dimensions[0], height = dimensions[1]
    let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: width * 4, bitsPerPixel: 32)!
    let context = NSGraphicsContext(bitmapImageRep: bitmap)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: width, height: height).fill()
    let size = fixture["font_size"] as! CGFloat
    let font = NSFont(name: "Menlo-Regular", size: size)!
    let attributes: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.black]
    let lines = (fixture["ocr_ground_truth"] as! String).components(separatedBy: "\n")
    for (index, line) in lines.enumerated() {
        (line as NSString).draw(at: NSPoint(x: 48, y: CGFloat(height) - 64 - size - CGFloat(index) * (size + 16)), withAttributes: attributes)
    }
    NSGraphicsContext.restoreGraphicsState()
    let png = bitmap.representation(using: .png, properties: [:])!
    let output = directory.appendingPathComponent(fixture["path"] as! String)
    try png.write(to: output)
    print(output.path)
}
