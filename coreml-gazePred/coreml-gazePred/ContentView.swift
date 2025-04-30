//
//  ContentView.swift
//  coreml-gazePred
//
//  Created by Pantelis Mitropoulos on 17/9/24.
//

import SwiftUI
import RealityKit
import RealityKitContent
import CoreML
import Vision
import UIKit
import Accelerate

struct ContentView: View {
    @State private var infoText: String = "Hello!"
    @State private var saliencyModel: saliency?
    @State private var tsModel: time_series?
    @State private var image: UIImage?

    var body: some View {
        let originalImg = UIImage(named: "Video_8_User_3_Task_1_15.jpg")
        VStack {
            if let image = image, let originalImg = originalImg {
                ZStack {
                    Image(uiImage: originalImg)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 384, height: 288)
                        .clipped()
                        .padding()
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 384, height: 288)
                        .padding()
                        .opacity(0.9)
                }
            } else if let originalImg = originalImg {
                Image(uiImage: originalImg)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 384, height: 288)
                    .clipped()
                    .padding()
            }

            Text(infoText)
            Button("Gaze Prediction") {
                infoText = "Wait for Gaze Prediction"
                Task {
                    await gazePredict()
                }
            }
        }
        .padding()
        .onAppear(perform: loadModels)
    }
    
    enum MLMultiArrayError: Error {
        case creationFailed
    }
    
    func gazePredict() async {
        // INPUT
        let startTime: UInt64 = DispatchTime.now().uptimeNanoseconds
        let hmd_horz: [Float] = [27.92, 27.92, 27.87, 27.87, 27.84, 27.87, 28.03, 27.99, 28.02, 28.16, 28.15, 28.11, 27.99, 27.64, 27.38]
        let hmd_vert: [Float] = [-22.84, -22.84, -22.87, -22.82, -22.83, -22.88, -22.85, -22.83, -22.86, -22.80, -22.73, -22.74, -22.55, -22.42, -22.37]
        if let saliencyMaps = await saliencyMap() {
            let prediction = await tsMixer(combinedArray: saliencyMaps, hmd_horz: hmd_horz, hmd_vert: hmd_vert)
            if let prediction = prediction {
                let shape = prediction.shape

                for i in 0..<shape[1].intValue {
                    for j in 0..<shape[2].intValue {
                        let value = j == 0 ? prediction[[0, i as NSNumber, j as NSNumber]].floatValue + 27.38 : prediction[[0, i as NSNumber, j as NSNumber]].floatValue - 22.37
                        print(i, j, value)
                    }
                }
                print(prediction)
            }
        } else {
            return
        }
        let endTime = DispatchTime.now().uptimeNanoseconds - startTime
        print("Total time")
        print(endTime / 1000000)
        self.infoText = "Finished! Check console for output"
    }
    
    func saliencyMap() async -> MLMultiArray? {
        let imageNames = [
            "Video_8_User_3_Task_1_1.jpg", "Video_8_User_3_Task_1_2.jpg",
            "Video_8_User_3_Task_1_3.jpg", "Video_8_User_3_Task_1_4.jpg",
            "Video_8_User_3_Task_1_5.jpg", "Video_8_User_3_Task_1_6.jpg",
            "Video_8_User_3_Task_1_7.jpg", "Video_8_User_3_Task_1_8.jpg",
            "Video_8_User_3_Task_1_9.jpg", "Video_8_User_3_Task_1_10.jpg",
            "Video_8_User_3_Task_1_11.jpg", "Video_8_User_3_Task_1_12.jpg",
            "Video_8_User_3_Task_1_13.jpg", "Video_8_User_3_Task_1_14.jpg",
            "Video_8_User_3_Task_1_15.jpg"
        ]
        var output = [MLMultiArray]()
        var processTime: UInt64 = 0
        var modelTime: UInt64 = 0

        for imageName in imageNames {
            let startTime: UInt64 = DispatchTime.now().uptimeNanoseconds
            if let image = UIImage(named: imageName),
               let arrayImg = preprocess(image: image) {
                processTime += DispatchTime.now().uptimeNanoseconds - startTime
                do {
                    let startTime = DispatchTime.now().uptimeNanoseconds
                    guard let modelOutput = try saliencyModel?.prediction(fov_rgb: arrayImg) else {
                        print("Failed to retrieve saliency map")
                        return nil
                    }
                    output.append(modelOutput.saliecy_maps)
                    modelTime += DispatchTime.now().uptimeNanoseconds - startTime

                } catch {
                    print("Failed to create saliency map")
                }
            } else {
                print("Failed to load image")
            }
        }
        print("Created saliency maps")
        let startTimePost = DispatchTime.now().uptimeNanoseconds
        let combinedShape = [1, 15, 288, 384] as [NSNumber]
        guard let combinedArray = try? MLMultiArray(shape: combinedShape, dataType: .float32) else {
            print("Failed to create MLMultiArray")
            return nil
        }
        // Fill combinedArray with data from output
        for (index, saliencyMap) in output.enumerated() {
            // Use block copy for better performance
            let destPointer = UnsafeMutablePointer<Float>(OpaquePointer(combinedArray.dataPointer)).advanced(by: index * saliencyMap.count)
            let srcPointer = UnsafePointer<Float>(OpaquePointer(saliencyMap.dataPointer))
            destPointer.initialize(from: srcPointer, count: saliencyMap.count)
        }
        let postTime = DispatchTime.now().uptimeNanoseconds - startTimePost
        print("Saliency Times [preprocess, model, postprocess]")
        print(processTime / 1000000, modelTime / 1000000, postTime / 1000000)
        return combinedArray
    }

    func tsMixer(combinedArray: MLMultiArray, hmd_horz: [Float], hmd_vert: [Float]) async -> MLMultiArray? {

        // Create the MLMultiArray with shape [1, 15, 4]
        let shape = [1, 15, 4] as [NSNumber]
        guard var hmdTimeSeries = try? MLMultiArray(shape: shape, dataType: .float32) else {
            print("Failed to create MLMultiArray")
            return nil
        }

        let startTimePre: UInt64 = DispatchTime.now().uptimeNanoseconds
        do {
            hmdTimeSeries = try calculateHmd(angles_horz: hmd_horz, angles_vert: hmd_vert)
        } catch {
            print("Failed to create time series")
        }
        let preprocessTime = DispatchTime.now().uptimeNanoseconds - startTimePre

        let startTime: UInt64 = DispatchTime.now().uptimeNanoseconds
        var gazePrediction: MLMultiArray?
        do {
            guard let modelOutput = try tsModel?.prediction(processed_saliency_maps: combinedArray, hmd_time_series: hmdTimeSeries) else {
                print("Failed to run tsmixer")
                return nil
            }
            gazePrediction = modelOutput.gaze_predictions
        } catch {
            print("Failed to run tsmixer")
        }
        let modelTime = DispatchTime.now().uptimeNanoseconds - startTime
        
        print("TSMixer Times [preprocess, model]")
        print(preprocessTime / 1000000, modelTime / 1000000)
        return gazePrediction
    }
    
    func angularVelocity(angles_horz: [Float], angles_vert: [Float], sampleRate: Float=30, velRange: Float=700) -> (horz: [Float], vert: [Float]) {
        let count = angles_horz.count
        let countMinusOne = count - 1
        
        // Preallocate arrays with exact final size
        var vel_horz = [Float](repeating: 0.0, count: count)
        var vel_vert = [Float](repeating: 0.0, count: count)
        
        // Calculate velocities for all indices but the first
        for i in 0..<countMinusOne {
            // Calculate both horizontal and vertical in the same loop iteration
            let horzValue = (angles_horz[i+1] - angles_horz[i]) * sampleRate
            let vertValue = (angles_vert[i+1] - angles_vert[i]) * sampleRate
            
            // Use min/max for clamping - more efficient than conditionals
            vel_horz[i+1] = min(max(horzValue, -velRange), velRange)
            vel_vert[i+1] = min(max(vertValue, -velRange), velRange)
        }
        
        // Set first element to match second element (duplicating the first calculated velocity)
        vel_horz[0] = vel_horz[1]
        vel_vert[0] = vel_vert[1]
        
        return (horz: vel_horz, vert: vel_vert)
    }
    
    func anglesDiff(angles_horz: [Float], angles_vert: [Float]) -> (horz: [Float], vert: [Float]) {
        let count = angles_horz.count
        let countMinusOne = count - 1
        
        // Preallocate arrays with exact final size
        var diff_horz = [Float](repeating: 0.0, count: count)
        var diff_vert = [Float](repeating: 0.0, count: count)
        
        // Combine the horizontal and vertical calculations in a single loop
        for i in 0..<countMinusOne {
            let horzDiff = angles_horz[i+1] - angles_horz[i]
            let vertDiff = angles_vert[i+1] - angles_vert[i]
            
            diff_horz[i+1] = horzDiff
            diff_vert[i+1] = vertDiff
        }
        
        // Set first element to match second element (duplicating the first calculated diff)
        diff_horz[0] = diff_horz[1]
        diff_vert[0] = diff_vert[1]
        
        return (horz: diff_horz, vert: diff_vert)
    }
    
    func calculateHmd(angles_horz: [Float], angles_vert: [Float]) throws -> MLMultiArray {
        let velocity = angularVelocity(angles_horz: angles_horz, angles_vert: angles_vert)
        let diffs = anglesDiff(angles_horz: angles_horz, angles_vert: angles_vert)

        let shape = [1, 15, 4] as [NSNumber]
        guard let hmdTimeSeries = try? MLMultiArray(shape: shape, dataType: .float32) else {
            throw MLMultiArrayError.creationFailed
        }
        
        // Get direct pointer access to the MLMultiArray's memory for faster writes
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(hmdTimeSeries.dataPointer))
        
        // Pre-calculate strides for faster indexing
        let timeSeriesStride = 4 // 4 values per time step
        
        for i in 0..<15 {
            let baseIndex = i * timeSeriesStride
            pointer[baseIndex] = velocity.horz[i]
            pointer[baseIndex + 1] = velocity.vert[i]
            pointer[baseIndex + 2] = diffs.horz[i]
            pointer[baseIndex + 3] = diffs.vert[i]
        }
        
        return hmdTimeSeries
    }
    
    func loadModels() {
        print("Loading models")
        do {
            saliencyModel = try saliency()
        } catch {
            print("Failed to load saliency map model: \(error.localizedDescription)")
        }
        do {
            tsModel = try time_series()
        } catch {
            print("Failed to load tsmixer model: \(error.localizedDescription)")
        }
    }

    func preprocess(image: UIImage) -> MLMultiArray? {
        guard let cgImage = image.cgImage else { return nil }

        let width = 384
        let height = 288
        
        let bytesPerRow = width * 4
        var pixelData = [UInt8](repeating: 0, count: height * bytesPerRow)
        
        guard let context = pixelData.withUnsafeMutableBytes({ ptr in
            CGContext(
                data: ptr.baseAddress,
                width: width, 
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        }) else { return nil }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        let shape = [1, 3, height, width] as [NSNumber]
        guard let array = try? MLMultiArray(shape: shape, dataType: .float32) else { return nil }
        let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(array.dataPointer))
        
        // Pre-calculate combined normalization constants
        let r_scale = Float32(1.0 / (255.0 * 0.229))
        let g_scale = Float32(1.0 / (255.0 * 0.224))
        let b_scale = Float32(1.0 / (255.0 * 0.225))
        let r_offset = Float32(0.485 / 0.229)
        let g_offset = Float32(0.456 / 0.224)
        let b_offset = Float32(0.406 / 0.225)
        
        let channelSize = height * width
        
        // Process all pixels with minimal operations
        for y in 0..<height {
            let rowStart = y * width
            let rowByteStart = y * bytesPerRow
            
            for x in 0..<width {
                let pixelIndex = rowStart + x
                let byteIndex = rowByteStart + (x * 4)
                
                // Combined scaling and offset in one operation
                pointer[pixelIndex] = Float32(pixelData[byteIndex]) * r_scale - r_offset
                pointer[pixelIndex + channelSize] = Float32(pixelData[byteIndex + 1]) * g_scale - g_offset
                pointer[pixelIndex + 2 * channelSize] = Float32(pixelData[byteIndex + 2]) * b_scale - b_offset
            }
        }

        return array
    }

}

#Preview(windowStyle: .automatic) {
    ContentView()
}
