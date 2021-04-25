import Foundation
import SwiftUI
import AppKit
import Metal
import MetalKit
import QuartzCore

func GetMetalDevice() -> MTLDevice? {
    MTLCreateSystemDefaultDevice()
}

private let rendererShader = """
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

typedef enum AAPLVertexInputIndex {
    AAPLVertexInputIndexVertices     = 0,
    AAPLVertexInputIndexViewportSize = 1,
} VertexInputIndex;

typedef struct {
    vector_float2 position;
    vector_float2 textureCoordinate;
} Vertex;

struct RasterizerData {
    float4 clipSpacePosition [[position]];
    float2 textureCoordinate;
};

vertex RasterizerData vertexShader(
    uint vertexID [[ vertex_id ]],
    constant Vertex *vertexArray [[ buffer(AAPLVertexInputIndexVertices) ]],
    constant vector_uint2 *viewportSizePointer [[ buffer(AAPLVertexInputIndexViewportSize) ]]
) {
    RasterizerData out;
    float2 pixelSpacePosition = vertexArray[vertexID].position.xy;
    float2 viewportSize = float2(*viewportSizePointer);
    out.clipSpacePosition.xy = pixelSpacePosition / (viewportSize / 2.0);
    out.clipSpacePosition.z = 0.0;
    out.clipSpacePosition.w = 1.0;
    out.textureCoordinate = vertexArray[vertexID].textureCoordinate;
    return out;
}

fragment float4 samplingShader(
    RasterizerData in [[stage_in]],
    texture2d<float> colorTexture [[ texture(0) ]]
) {
    constexpr sampler textureSampler (mag_filter::linear, min_filter::linear);
    const auto colorSample = colorTexture.sample (textureSampler, in.textureCoordinate);
    return colorSample;
}

kernel void clearTexture(texture2d<float, access::write> texture [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    texture.write(float4(0, 0, 0, 1), ushort2(gid));
}
"""

protocol MetalRenderer {
    func render(_ texture: MTLTexture, buffer: MTLCommandBuffer)
}

struct EmptyRenderer: MetalRenderer {
    func render(_ texture: MTLTexture, buffer: MTLCommandBuffer) {}
}


class MTKTrackingView: MTKView {
    var onMouseMove: ((CGPoint, CGVector) -> ())?
    var onMouseClick: ((CGPoint, Bool) -> ())?
    
    var trackingArea: NSTrackingArea?
    var leftMouseDown: Bool = false
    var rightMouseDown: Bool = false

    override func updateTrackingAreas() {
        if trackingArea != nil {
            self.removeTrackingArea(trackingArea!)
        }
        let options: NSTrackingArea.Options = [.mouseEnteredAndExited, .mouseMoved, .activeInKeyWindow]
        trackingArea = NSTrackingArea(rect: self.bounds, options: options, owner: self, userInfo: nil)
        self.addTrackingArea(trackingArea!)
    }
    
    override func mouseMoved(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        onMouseMove?(
            CGPoint(x: point.x, y: frame.height - point.y),
            CGVector(dx: event.deltaX, dy: event.deltaY)
        )
    }
    
    override func mouseUp(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        onMouseClick?(CGPoint(x: point.x, y: frame.height - point.y), true)
    }
    
    override func rightMouseUp(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        onMouseClick?(CGPoint(x: point.x, y: frame.height - point.y), false)
    }
    
    override func mouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        onMouseClick?(CGPoint(x: point.x, y: frame.height - point.y), true)
    }
    
    override func rightMouseDragged(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        onMouseClick?(CGPoint(x: point.x, y: frame.height - point.y), false)
    }
    
}


class MetalViewController: NSViewController, MTKViewDelegate {
    let device: MTLDevice
    private(set) var metalView: MTKTrackingView!
    fileprivate(set) var renderer: MetalRenderer
    private var commandQueue: MTLCommandQueue
    private var texture: MTLTexture?
    private var fragmentShader: MTLFunction?
    private var vertexShader: MTLFunction?
    private var renderPipeline: MTLRenderPipelineState?
    private var displayLink: CVDisplayLink?
    private var clearState: MTLComputePipelineState?
    
    
    var onMouseMove: ((CGPoint, CGVector) -> ())? {
        didSet {
            metalView?.onMouseMove = onMouseMove
        }
    }
    var onMouseClick: ((CGPoint, Bool) -> ())? {
        didSet {
            metalView?.onMouseClick = onMouseClick
        }
    }
    
    init(device: MTLDevice, renderer: MetalRenderer) {
        self.device = device
        self.renderer = renderer
        self.commandQueue = device.makeCommandQueue()!
        
        super.init()
    }
    
    required init?(coder: NSCoder) {
        guard let device = GetMetalDevice(), let queue = device.makeCommandQueue() else {
            return nil
        }
        self.device = device
        self.renderer = EmptyRenderer()
        self.commandQueue = queue
        super.init(coder: coder)
    }
    
    override init(nibName nibNameOrNil: NSNib.Name?, bundle nibBundleOrNil: Bundle?) {
        let device = GetMetalDevice()!
        let queue = device.makeCommandQueue()!
        self.device = device
        self.renderer = EmptyRenderer()
        self.commandQueue = queue
        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
    }
    
    private func startDisplayLink() {
        if self.displayLink == nil {
            CVDisplayLinkCreateWithActiveCGDisplays(&self.displayLink)
        }
        guard let displayLink = self.displayLink, !CVDisplayLinkIsRunning(displayLink) else {
            return
        }
        CVDisplayLinkSetOutputHandler(displayLink) { link, ts1, ts2, options, option2 in
            DispatchQueue.main.async {
                self.metalView.setNeedsDisplay(self.metalView.bounds)
            }
            return kCVReturnSuccess
        }
        CVDisplayLinkStart(displayLink)
    }
    
    private func stopDisplayLink() {
        guard let displayLink = self.displayLink, CVDisplayLinkIsRunning(displayLink) else {
            return
        }
        CVDisplayLinkStop(displayLink)
        self.displayLink = nil
    }
    
    override func loadView() {
        let view = MTKTrackingView(frame: .zero, device: self.device)
        self.view = view
        self.metalView = view
        view.onMouseMove = onMouseMove
        view.onMouseClick = onMouseClick
        view.delegate = self
        view.addTrackingRect(.zero, owner: self, userData: nil, assumeInside: false)
        
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.enableSetNeedsDisplay = true
        
        self.mtkView(view, drawableSizeWillChange: view.drawableSize)
        
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true
        
        guard let library = try? device.makeLibrary(source: rendererShader, options: compileOptions) else {
            return
        }
        fragmentShader = library.makeFunction(name: "samplingShader")
        vertexShader = library.makeFunction(name: "vertexShader")
        
        if let clearFunction = library.makeFunction(name: "clearTexture"),
           let state = try? device.makeComputePipelineState(function: clearFunction) {
            self.clearState = state
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "render pipeline"
        pipelineDescriptor.vertexFunction = vertexShader
        pipelineDescriptor.fragmentFunction = fragmentShader
        pipelineDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        self.renderPipeline = try? device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        guard size.width > 0, size.height > 0 else {
            self.texture = nil
            return
        }
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba32Float
        descriptor.width = Int(size.width)
        descriptor.height = Int(size.height)
        descriptor.textureType = .type2D
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        guard
            let texture = self.device.makeTexture(descriptor: descriptor),
            let buffer = commandQueue.makeCommandBuffer(),
            let encoder = buffer.makeComputeCommandEncoder(),
            let state = clearState
        else {
            self.texture = nil
            return
        }
        
        encoder.setComputePipelineState(state)
        encoder.setTexture(texture, index: 0)
        
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(
            width: (texture.width + tgSize.width - 1) / tgSize.width,
            height: (texture.height + tgSize.height - 1) / tgSize.height,
            depth: 1
        )
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        buffer.commit()
        buffer.waitUntilCompleted()
        
        self.texture = texture
        
        // view.addTrackingRect(view.bounds, owner: self, userData: nil, assumeInside: false)
    }
    
    func draw(in view: MTKView) {
        guard
            let descriptor = view.currentRenderPassDescriptor,
            let buffer = commandQueue.makeCommandBuffer(),
            let drawable = view.currentDrawable,
            let texture = self.texture
        else {
            return
        }
        
        self.renderer.render(texture, buffer: buffer)
        
        guard let renderCommandEncoder = buffer.makeRenderCommandEncoder(descriptor: descriptor), let renderPipeline = self.renderPipeline else {
            return
        }
        
        let viewportSize: [UInt32] = [UInt32(view.drawableSize.width), UInt32(view.drawableSize.height)]
        renderCommandEncoder.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(view.drawableSize.width), height: Double(view.drawableSize.height), znear: -1, zfar: 1))
        renderCommandEncoder.setRenderPipelineState(renderPipeline)
        
        let w = Float(viewportSize[0]) / 2
        let h = Float(viewportSize[1]) / 2
        
        let vertices: [Float] = [
             w, -h, 1, 1,
            -w, -h, 0, 1,
            -w,  h, 0, 0,
             w, -h, 1, 1,
            -w,  h, 0, 0,
             w,  h, 1, 0
        ]
        renderCommandEncoder.setVertexBytes(vertices, length: vertices.count * MemoryLayout<Float>.stride, index: 0)
        renderCommandEncoder.setVertexBytes(viewportSize, length: viewportSize.count * MemoryLayout<UInt32>.stride, index: 1)
        renderCommandEncoder.setFragmentTexture(texture, index: 0)
        renderCommandEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)
        renderCommandEncoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        startDisplayLink()
    }
    
    override func viewDidDisappear() {
        stopDisplayLink()
    }
}


struct MetalView: NSViewControllerRepresentable {
    let device: MTLDevice
    let renderer: MetalRenderer
    let onMouseMove: ((CGPoint, CGVector) -> ())?
    let onMouseClick: ((CGPoint, Bool) -> ())?
    
    init(renderer: MetalRenderer, device: MTLDevice, onMouseMove: ((CGPoint, CGVector) -> ())? = nil, onMouseClick: ((CGPoint, Bool) -> ())? = nil) {
        self.device = device
        self.renderer = renderer
        self.onMouseMove = onMouseMove
        self.onMouseClick = onMouseClick
    }
    
    func makeNSViewController(context: Context) -> MetalViewController {
        let vc = MetalViewController(device: device, renderer: self.renderer)
        vc.renderer = self.renderer
        vc.onMouseMove = onMouseMove
        vc.onMouseClick = onMouseClick
        return vc
    }
    
    func updateNSViewController(_ vc: MetalViewController, context: Context) {
        vc.renderer = self.renderer
        vc.onMouseMove = onMouseMove
        vc.onMouseClick = onMouseClick
    }
}

private let simulationShader = """
#include <metal_stdlib>
using namespace metal;

#define PI 3.141592653589

typedef struct {
    float2 pos;
    float dir;
    float pad;
    int4 species;
} agent_t;

typedef struct {
    float sensor_offset;
    int sensor_size;
    float sensor_angle_spacing;
    float turn_speed;
    float evaporation_speed;
    float move_speed;
    float trail_weight;
} config_t;


uint hash(uint seed) {
    seed ^= 2447636419u;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= 2654435769u;
    seed ^= seed >> 16;
    seed *= 2654435769u;
    return seed;
}

float sense(agent_t agent, float dir, int2 dim, texture2d<float, access::read> texture, const config_t config) {
    auto sensor_angle = agent.dir + dir;
    auto sensor_dir = float2(cos(sensor_angle), sin(sensor_angle));
    auto sensor_pos = agent.pos + sensor_dir * config.sensor_offset;
    
    float sum = 0;
    
    auto bound = config.sensor_size - 1;
    
    // sum = dot(texture.read(ushort2(sensor_pos)), float4(agent.species) * 2 - 1);
    
    for (int dy = -bound; dy <= bound; dy++) {
        for (int dx = -bound; dx <= bound; dx++) {
            int x = sensor_pos.x + dx;
            int y = sensor_pos.y + dy;
    
            if (x >= 0 && y >= 0 && x < dim.x && y < dim.y) {
                sum += dot(texture.read(ushort2(x, y)), float4(agent.species) * 2 - 1);
            }
        }
    }
    
    return sum;
}

float3 unitcircle_random(thread uint *seed) {
    auto argSeed = hash(*seed);
    auto absSeed = hash(argSeed);
    *seed = absSeed;
    
    auto arg = (float) argSeed / UINT_MAX * 2 * PI;
    auto absSqrt = (float) absSeed / UINT_MAX;
    auto absR = absSqrt * absSqrt;
    
    return float3(absR * cos(arg), absR * sin(arg), arg + PI);
}

kernel void initAgents(
    device agent_t *agents [[buffer(0)]],
    constant uint2 &dim [[buffer(1)]],
    constant uint &num_agents [[buffer(2)]],
    constant uint &num_species [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    auto mid = float2(dim) / 2;
    auto rad = float(min(dim.x, dim.y)) / 2;
    
    auto seed = gid;
    auto init = unitcircle_random(&seed);
    auto pos = float2(init.x, init.y) * rad + mid;
    
    agent_t agent = agents[gid];
    agent.pos = pos;
    agent.dir = init.z;
    
    if (num_species == 1) {
        agent.species = int4(0, 1, 1, 1);
    } else if (num_species == 2) {
        agent.species = int4(0, gid % 2, 1 - gid % 2, 1);
    } else if (num_species == 3) {
        agent.species = int4(gid % 3 == 2, gid % 3 == 1, gid % 3 == 0, 1);
    }
    
    agents[gid] = agent;
}

kernel void updateAgents(
    device agent_t *agents [[buffer(0)]],
    constant uint2 &dim [[buffer(1)]],
    constant uint &num_agents [[buffer(2)]],
    constant config_t &config [[buffer(3)]],
    constant float &time_delta [[buffer(4)]],
    texture2d<float, access::read> texture_read [[texture(0)]],
    texture2d<float, access::write> texture_write [[texture(1)]],
    uint gid [[thread_position_in_grid]]
) {
    auto idim = int2(dim);
    auto agent = agents[gid];
    auto rnd = hash(agent.pos.y * dim.x + agent.pos.x + hash(gid));
    auto dir_vec = float2(cos(agent.dir), sin(agent.dir));
    auto new_pos = agent.pos + config.move_speed * time_delta * dir_vec;
    
    if (new_pos.x < 0 || new_pos.y < 0 || new_pos.x >= dim.x || new_pos.y >= dim.y) {
        new_pos = clamp(new_pos, float2(0, 0), float2(dim) - 0.01);
        agent.dir = (float) rnd / UINT_MAX * 2 * PI;
    }
    agent.pos = new_pos;
    
    
    auto fwd_w = sense(agent, 0, idim, texture_read, config);
    auto left_w = sense(agent, config.sensor_angle_spacing, idim, texture_read, config);
    auto right_w = sense(agent, -config.sensor_angle_spacing, idim, texture_read, config);
    rnd = hash(rnd);
    
    auto rnd_steer_strength = (float) rnd / UINT_MAX;
    
    if (fwd_w >= left_w && fwd_w >= right_w) {
        // noop
    } else if (fwd_w < left_w && fwd_w < right_w) {
        agent.dir += (rnd_steer_strength - 0.5) * 2 * config.turn_speed * time_delta;
    } else if (right_w > left_w) {
        agent.dir -= rnd_steer_strength * config.turn_speed * time_delta;
    } else if (left_w > right_w) {
        agent.dir += rnd_steer_strength * config.turn_speed * time_delta;
    }
    
    agents[gid] = agent;
    texture_write.write(min(float4(agent.species) * config.trail_weight, 1), ushort2(new_pos));
}

kernel void updateTrails(
    texture2d<float, access::read> texture_read [[texture(0)]],
    texture2d<float, access::write> texture_write [[texture(1)]],
    constant uint2 &dim [[buffer(0)]],
    constant config_t &config [[buffer(1)]],
    constant float &time_delta [[buffer(2)]],
    device const float4 *sources [[buffer(3)]],
    constant uint &num_sources [[buffer(4)]],
    constant uint &num_species [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 sum = 0;
    const int2 idim = int2(dim);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int x = gid.x + dx;
            int y = gid.y + dy;
            
            if (x >= 0 && y >= 0 && x < idim.x && y < idim.y) {
                sum += texture_read.read(ushort2(x, y));
            }
        }
    }
    
    // texture_write.write(sum / 9, ushort2(gid));
    
    // auto color = vec<float, 4>((float) gid.x / (float) dim.x, (float) gid.y / (float) dim.y, 1, 1);
    auto current = sum / 9;
    current *= max(0.01, 1 - config.evaporation_speed);
    current.w = 1;
    
    auto source_radius = (float) min(dim.x, dim.y) * 0.1f;
    for (uint i = 0; i < num_sources; i++) {
        auto source = sources[i];
        auto dist = distance(source.xy, float2(gid)) / source_radius;
        if (dist <= 1) {
            if (source.z < 0) {
                current = min(max(dist - 0.2, 0.0f), current);
            } else if (num_species == 1) {
                current.yz = max(1 - dist, current.yz);
            } else {
                current.y = max(1 - dist, current.y);
            }
        }
    }
    
    texture_write.write(current, ushort2(gid));
}

kernel void updateSpecies(
    device agent_t *agents [[buffer(0)]],
    constant uint &num_agents [[buffer(1)]],
    constant uint &num_species [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (num_species == 1) {
        agents[gid].species = int4(0, 1, 1, 1);
    } else if (num_species == 2) {
        agents[gid].species = int4(0, gid % 2, 1 - gid % 2, 1);
    } else if (num_species == 3) {
        agents[gid].species = int4(gid % 3 == 2, gid % 3 == 1, gid % 3 == 0, 1);
    }
}

kernel void performInteractions(
    device agent_t *agents [[buffer(0)]],
    constant uint &num_agents [[buffer(1)]],
    device const float4 *interaction_points [[buffer(2)]],
    constant uint &num_interactions [[buffer(3)]],
    constant uint &seed [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    auto rnd = hash(hash(gid) ^ seed);
    
    agent_t agent = agents[gid];
    
    for (uint i = 0; i < num_interactions; i++) {
        rnd = hash(rnd);
        float4 interaction = interaction_points[i];
        float2 pos = interaction.xy;
        float2 dirVec = interaction.zw;
        float dir = atan2(dirVec.y, dirVec.x);
        if ((float) rnd / UINT_MAX <= 0.0001) {
            agent.pos = pos + unitcircle_random(&rnd).xy * 20 - 10;
            agent.dir = dir;
        }
    }
    
    agents[gid] = agent;
}

"""

struct SimulationConfig: Codable, Hashable {
    var sensorOffset: Float
    var sensorSize: Int32
    var sensorAngleSpacing: Float
    var turnSpeed: Float
    var evaporationSpeed: Float
    var moveSpeed: Float
    var trailWeight: Float
    var species: Int
    
    var floatSpecies: Float {
        get {Float(species)}
        set {species = Int(newValue)}
    }
    
    static var `default`: Self {
        .init(sensorOffset: 15, sensorSize: 1, sensorAngleSpacing: 0.2 * .pi, turnSpeed: 50, evaporationSpeed: 0.5, moveSpeed: 60, trailWeight: 1, species: 1)
    }
    
    static var byteCount: Int {
        return 7 * 4
    }
    
    func encoded() -> UnsafeRawPointer {
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: 7 * 4, alignment: MemoryLayout<Float>.alignment)
        
        ptr.assumingMemoryBound(to: Float.self)[0] = sensorOffset
        ptr.assumingMemoryBound(to: Int32.self)[1] = sensorSize
        ptr.assumingMemoryBound(to: Float.self)[2] = sensorAngleSpacing
        ptr.assumingMemoryBound(to: Float.self)[3] = turnSpeed
        ptr.assumingMemoryBound(to: Float.self)[4] = evaporationSpeed
        ptr.assumingMemoryBound(to: Float.self)[5] = moveSpeed
        ptr.assumingMemoryBound(to: Float.self)[6] = trailWeight
        
        return UnsafeRawPointer(ptr)
    }
}

struct Agent {
    let pos: (Float, Float)
    let dir: Float
    let species: (Int32, Int32, Int32, Int32)
    
    static var byteCount: Int { 8 * 4 }
}

func randomAgentInCircle(radius: Float, center: (Float, Float), species: (Int32, Int32, Int32)) -> Agent {
    let radSqrt = Float.random(in: 0 ... 1)
    let rad = radSqrt * radSqrt * radius
    let arg = Float.random(in: 0 ..< 2 * .pi)
    let pos = (cos(arg) * rad + center.0, sin(arg) * rad + center.1)
    let dir = arg + .pi
    return Agent(pos: pos, dir: dir, species: (species.0, species.1, species.2, 1))
}

class SimulationRenderer: MetalRenderer, ObservableObject {
    private let device: MTLDevice
    private let initState: MTLComputePipelineState
    private let agentsState: MTLComputePipelineState
    private let trailsState: MTLComputePipelineState
    private let speciesState: MTLComputePipelineState
    private let interactionsState: MTLComputePipelineState
    private var agents: MTLBuffer?
    private var previousTime: TimeInterval? = nil
    private var activeSpecies: Int = 1
    
    @Published private var __agentCount: Int
    var agentCount: Int {
        didSet {
            let roundedValue = agentCount % 256 == 0 ? agentCount : ((agentCount / 256 + 1) * 256)
            __agentCount = min(max(roundedValue, 1<<10), 1<<24)
            agents = nil
        }
    }
    @Published var configuration: SimulationConfig = .default {
        didSet {
            species = configuration.species
        }
    }
    @Published var species: Int = 1
    
    var mouseEvents: [(Float, Float, Float, Float)] = []
    var sources: [(Float, Float, Bool)] = []
    
    init?(device: MTLDevice, agentCount: Int) {
        let compileOptions = MTLCompileOptions()
        compileOptions.fastMathEnabled = true
        guard
            let library = try? device.makeLibrary(source: simulationShader, options: compileOptions),
            let initAgents = library.makeFunction(name: "initAgents"),
            let updateAgents = library.makeFunction(name: "updateAgents"),
            let updateTrails = library.makeFunction(name: "updateTrails"),
            let updateSpecies = library.makeFunction(name: "updateSpecies"),
            let performInteractions = library.makeFunction(name: "performInteractions"),
            let initState = try? device.makeComputePipelineState(function: initAgents),
            let agentsState = try? device.makeComputePipelineState(function: updateAgents),
            let trailsState = try? device.makeComputePipelineState(function: updateTrails),
            let speciesState = try? device.makeComputePipelineState(function: updateSpecies),
            let interactionsState = try? device.makeComputePipelineState(function: performInteractions)
        else {
            return nil
        }
        
        self.device = device
        self.initState = initState
        self.agentsState = agentsState
        self.trailsState = trailsState
        self.speciesState = speciesState
        self.interactionsState = interactionsState
        
        self.agentCount = agentCount
        let roundedAgentCount = agentCount % 256 == 0 ? agentCount : ((agentCount / 256 + 1) * 256)
        self.__agentCount = roundedAgentCount
    }
    
    private func initAgents(width: Int, height: Int, buffer: MTLCommandBuffer) {
        guard
            let agentBuffer = device.makeBuffer(length: Agent.byteCount * self.__agentCount, options: .storageModePrivate),
            let encoder = buffer.makeComputeCommandEncoder()
        else {
            return
        }
        
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        let tgCount = MTLSize(width: (__agentCount + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        
        encoder.setComputePipelineState(initState)
        encoder.setBuffer(agentBuffer, offset: 0, index: 0)
        encoder.setBytes([UInt32(width), UInt32(height)], length: 8, index: 1)
        encoder.setBytes([UInt32(__agentCount)], length: 4, index: 2)
        encoder.setBytes([UInt32(species)], length: 4, index: 3)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        
        self.agents = agentBuffer
    }
    
    func render(_ texture: MTLTexture, buffer: MTLCommandBuffer) {
        let time = CACurrentMediaTime()
        let delta: Double
        if let previousTime = self.previousTime {
            delta = min(time - previousTime, 1 / 20)
        } else {
            delta = 1.0 / 60.0
        }
        self.previousTime = time
        
        if agents == nil {
            initAgents(width: texture.width, height: texture.height, buffer: buffer)
        }
        updateSpeciesIfNecessary(buffer: buffer)
        updateInteractionsIfNecessary(buffer: buffer)
        updateAgents(texture: texture, buffer: buffer, delta: delta)
        updateTrails(texture: texture, buffer: buffer, delta: delta)
    }
    
    private func updateAgents(texture: MTLTexture, buffer: MTLCommandBuffer, delta: Double) {
        guard let encoder = buffer.makeComputeCommandEncoder() else {
            return
        }
        
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        let tgCount = MTLSize(width: (__agentCount + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        
        let configBytes = configuration.encoded()
        defer {
            configBytes.deallocate()
        }
        
        encoder.setComputePipelineState(self.agentsState)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(texture, index: 1)
        encoder.setBuffer(agents, offset: 0, index: 0)
        encoder.setBytes([UInt32(texture.width), UInt32(texture.height)], length: 8, index: 1)
        encoder.setBytes([UInt32(__agentCount)], length: 4, index: 2)
        encoder.setBytes(configBytes, length: SimulationConfig.byteCount, index: 3)
        encoder.setBytes([Float(delta)], length: 4, index: 4)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }
    
    private func updateTrails(texture: MTLTexture, buffer: MTLCommandBuffer, delta: Double) {
        guard let encoder = buffer.makeComputeCommandEncoder() else {
            return
        }
        let tgSize = MTLSize(width: 16, height: 16, depth: 1)
        let tgCount = MTLSize(
            width: (texture.width + tgSize.width - 1) / tgSize.width,
            height: (texture.height + tgSize.height - 1) / tgSize.height,
            depth: 1
        )
        
        let configBytes = configuration.encoded()
        defer {
            configBytes.deallocate()
        }
        
        encoder.setComputePipelineState(self.trailsState)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(texture, index: 1)
        encoder.setBytes([UInt32(texture.width), UInt32(texture.height)], length: 8, index: 0)
        encoder.setBytes(configBytes, length: SimulationConfig.byteCount, index: 1)
        encoder.setBytes([Float(delta)], length: 4, index: 2)
        
        let sourceData: [Float] = sources.suffix(256).flatMap {[$0 * 2, $1 * 2, $2 ? 1 : -1, 0]}
        
        encoder.setBytes(sourceData + (sources.isEmpty ? [Float(0), Float(0), Float(0), Float(0)] : []), length: max(sourceData.count, 4) * 4, index: 3)
        encoder.setBytes([UInt32(min(sources.count, 256))], length: 4, index: 4)
        encoder.setBytes([UInt32(species)], length: 4, index: 5)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }
    
    private func updateSpeciesIfNecessary(buffer: MTLCommandBuffer) {
        guard activeSpecies != species, let encoder = buffer.makeComputeCommandEncoder() else {
            return
        }
        activeSpecies = species
        
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        let tgCount = MTLSize(width: (__agentCount + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        
        encoder.setComputePipelineState(speciesState)
        encoder.setBuffer(agents, offset: 0, index: 0)
        encoder.setBytes([UInt32(__agentCount)], length: 4, index: 1)
        encoder.setBytes([UInt32(species)], length: 4, index: 2)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
    }
    
    private func updateInteractionsIfNecessary(buffer: MTLCommandBuffer) {
        guard mouseEvents.count > 1 else {
            return
        }
        guard let encoder = buffer.makeComputeCommandEncoder(), let agents = agents else {
            return
        }
        
        let interpolationFactor = 16
        
        let events = mouseEvents.map {[$0 * 2, $1 * 2, $2, $3]}
        var interpolatedEvents = [[Float]](repeating: [Float](repeating: 0, count: 4), count: mouseEvents.count * interpolationFactor)
        for i in interpolatedEvents.indices {
            for j in 0 ..< 4 {
                let scale = Float(i % interpolationFactor) / Float(interpolationFactor)
                let left = events[i / interpolationFactor][j] * scale
                let right = events[min(i / interpolationFactor + 1, events.count - 1)][j] * (1 - scale)
                interpolatedEvents[i][j] = left + right
            }
        }
        
        let tgSize = MTLSize(width: 256, height: 1, depth: 1)
        let tgCount = MTLSize(width: (__agentCount + tgSize.width - 1) / tgSize.width, height: 1, depth: 1)
        
        encoder.setComputePipelineState(interactionsState)
        encoder.setBuffer(agents, offset: 0, index: 0)
        encoder.setBytes([UInt32(self.__agentCount)], length: 4, index: 1)
        encoder.setBytes(Array(interpolatedEvents.joined()), length: interpolatedEvents.count * 16, index: 2)
        encoder.setBytes([UInt32(interpolatedEvents.count)], length: 4, index: 3)
        encoder.setBytes([UInt32.random(in: 0 ..< .max)], length: 4, index: 4)
        encoder.dispatchThreadgroups(tgCount, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()
        
        mouseEvents.removeSubrange(0 ..< mouseEvents.count - 1)
    }
}

struct PresetView: View {
    var image: String
    @Binding var isSelected: Bool
    
    var body: some View {
        ZStack(alignment: .center) {
            Image(nsImage: #imageLiteral(resourceName: image))
            .resizable()
            .aspectRatio(contentMode: .fill)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            
            if isSelected {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color.blue, style: .init(lineWidth: 3))
            }
        }
        .frame(height: 100, alignment: .center)
        .onTapGesture {
            isSelected = true
        }
    }
}

struct Preset: Hashable, Identifiable {
    var id: Self { self }
    var configuration: SimulationConfig
    var image: String
}

struct PresetListView: View {
    var presets: [Preset]
    @Binding var selectedPreset: Preset?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            VStack(alignment: .leading) {
                Text("Presets")
                .font(Font.title.bold())
                
                Text("Click a preset to apply")
                .font(Font.headline)
            }
            .padding(.bottom, -6)
            
            ForEach(presets) { preset in
                PresetView(image: preset.image, isSelected: Binding(get: {preset == selectedPreset}, set: { isSelected in
                    if isSelected {
                        selectedPreset = preset
                    }
                }))
            }
        }
        .padding(.bottom, 24)
    }
}

let presetJSON = [
    """
    {"sensorSize":1,"moveSpeed":66.327728271484375,"trailWeight":0.29464563727378845,"sensorOffset":100,"species":3,"evaporationSpeed":0.036113649606704712,"turnSpeed":69.388725280761719,"sensorAngleSpacing":0.9386482834815979}
    """,
    """
    {"sensorSize":1,"moveSpeed":36.770816802978516,"trailWeight":1,"sensorOffset":15,"species":1,"evaporationSpeed":0.18474458158016205,"turnSpeed":50,"sensorAngleSpacing":0.62831848859786987}
    """,
    """
    {"sensorSize":1,"moveSpeed":23.808805465698242,"trailWeight":1,"sensorOffset":9.6658048629760742,"species":2,"evaporationSpeed":0.15799397230148315,"turnSpeed":50,"sensorAngleSpacing":0.62831848859786987}
    """,
    """
    {"sensorSize":1,"moveSpeed":55.156581878662109,"trailWeight":1,"sensorOffset":6.8443918228149414,"species":3,"evaporationSpeed":0.1048186868429184,"turnSpeed":42.165946960449219,"sensorAngleSpacing":0.92847675085067749}
    """,
    """
    {"sensorSize":1,"moveSpeed":37.027008056640625,"trailWeight":1,"sensorOffset":6.8443918228149414,"species":1,"evaporationSpeed":0.1048186868429184,"turnSpeed":8.1130819320678711,"sensorAngleSpacing":0.92847675085067749}
    """,
    """
    {"sensorSize":1,"moveSpeed":100,"trailWeight":1,"sensorOffset":51.022769927978516,"species":2,"evaporationSpeed":0.1048186868429184,"turnSpeed":8.1130819320678711,"sensorAngleSpacing":0.12783576548099518}
    """,
    """
    {"sensorSize":1,"moveSpeed":100,"trailWeight":1,"sensorOffset":50.472572326660156,"species":1,"evaporationSpeed":0.02332134366035461,"turnSpeed":20.793535232543945,"sensorAngleSpacing":0.62831848859786987}
    """
]

struct ConfigPanel: View {
    @ObservedObject var renderer: SimulationRenderer
    var presets: [Preset] = presetJSON.enumerated().map { idx, json in
        Preset(configuration: try! JSONDecoder().decode(SimulationConfig.self, from: Data(json.utf8)), image: "preset\(idx + 1)")
    }
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading) {
                PresetListView(presets: presets, selectedPreset: Binding(
                    get: {presets.first(where: {$0.configuration == renderer.configuration})},
                    set: { preset in
                        if let preset = preset {
                            withAnimation {
                                renderer.configuration = preset.configuration
                            }
                        }
                    }
                ))
                
                Divider()
                
                Text("Configuration")
                .font(Font.title.bold())
                
                Group {
                    
                    CustomButton("Reset Agents", style: .destructive) {
                        renderer.agentCount = renderer.agentCount
                    }
                    
                    CustomButton("Reset Sources & Obstacles", style: .destructive) {
                        renderer.sources = []
                        renderer.mouseEvents = []
                    }
                    
                    Divider()
                    
                    Text("Number of Agents")
                    TextField("# Agents", value: $renderer.agentCount, formatter: NumberFormatter())
                    
                    Divider()
                }
                
                Group {
                    Text("Sensor Offset")
                    Slider(value: $renderer.configuration.sensorOffset, in: 2 ... 100)
                    
                    Divider()
                    
                    Text("Sensor Angle Spacing")
                    Slider(value: $renderer.configuration.sensorAngleSpacing, in: 0.05 ... Float.pi - 0.05)
                    
                    Divider()
                }
                
                Group {
                    Text("Turn Speed")
                    Slider(value: $renderer.configuration.turnSpeed, in: 0.1 ... 100)
                    
                    Divider()
                    
                    Text("Evaporation Speed")
                    Slider(value: $renderer.configuration.evaporationSpeed, in: 0.01 ... 0.9)
                    
                    Divider()
                    
                    Text("Move Speed")
                    Slider(value: $renderer.configuration.moveSpeed, in: 0.1 ... 100)
                    
                    Divider()
                }
                
                Group {
                    Text("Trail Weight")
                    Slider(value: $renderer.configuration.trailWeight, in: 0.01 ... 1)
                    
                    Divider()
                
                    Text("Species")
                    Slider(value: $renderer.configuration.floatSpecies, in: 1 ... 3)
                }
            }
            .padding()
        }
    }
}

struct SimulationView: View {
    let device: MTLDevice
    @ObservedObject var renderer: SimulationRenderer
    
    init(device: MTLDevice, renderer: SimulationRenderer) {
        self.device = device
        self.renderer = renderer
    }
    
    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            MetalView(
                renderer: renderer,
                device: device,
                onMouseMove: { point, dir in
                    self.renderer.mouseEvents.append((
                        Float(point.x),
                        Float(point.y),
                        Float(dir.dx),
                        Float(dir.dy)
                    ))
                },
                onMouseClick: { point, isSource in
                    self.renderer.sources.append((
                        Float(point.x),
                        Float(point.y),
                        isSource
                    ))
                }
            )
            
            ConfigPanel(renderer: renderer)
            .frame(width: 300)
        }
    }
}

public struct ContentView: View {
    public init() {}
    
    public var body: some View {
        Group {
            if let device = GetMetalDevice(), let renderer = SimulationRenderer(device: device, agentCount: 100_000) {
                ZStack(alignment: .center) {
                    SimulationView(device: device, renderer: renderer)
                    
                    TutorialView()
                }
            } else {
                VStack {
                    Spacer()
                    
                    HStack {
                        Spacer()
                        
                        Text("Metal not available.")
                        
                        Spacer()
                    }
                    
                    Spacer()
                }
            }
        }
    }
}

struct CustomButton: View {
    enum Style {
        case primary
        case secondary
        case destructive
    }
    
    var title: String
    var style: Style
    var action: () -> Void
    
    @Environment(\.colorScheme) var colorScheme: ColorScheme
    
    private var textColor: Color {
        switch (style, colorScheme) {
        case (.primary, _):
            return Color.white
        case (.secondary, .light):
            return .black
        case (.secondary, .dark):
            return .white
        case (.destructive, _):
            return .red
        @unknown default:
            return .white
        }
    }
    
    private var backgroundColor: Color {
        switch (style, colorScheme) {
        case (.primary, _):
            return .blue
        case (.secondary, .light), (.destructive, .light):
            return Color(white: 0.9)
        case (.secondary, .dark), (.destructive, .dark):
            return Color(white: 0.2)
        @unknown default:
            return .blue
        }
    }
    
    init(_ title: String, style: CustomButton.Style, action: @escaping () -> Void) {
        self.title = title
        self.style = style
        self.action = action
    }
    
    var body: some View {
        Button(action: self.action) {
            HStack {
                Spacer()
                
                Text(title)
                .foregroundColor(textColor)
                
                Spacer()
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(backgroundColor))
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct TutorialContainer<Content: View>: View {
    var content: Content
    @Binding var currentPage: Int
    let pageCount: Int
    
    @Environment(\.colorScheme) var colorScheme: ColorScheme
    
    init(pageCount: Int, currentPage: Binding<Int>, @ViewBuilder _ content: () -> Content) {
        self._currentPage = currentPage
        self.pageCount = pageCount
        self.content = content()
    }
    
    var body: some View {
        VStack(spacing: 0) {
            content
            .padding(32)
            
            Divider()
            
            HStack(spacing: 16) {
                if currentPage > 0 {
                    CustomButton("Previous", style: .secondary) {
                        withAnimation {
                            self.currentPage -= 1
                        }
                    }
                    .frame(width: 150)
                }
                
                Spacer()
                
                if currentPage + 1 < pageCount {
                    CustomButton("Skip Tutorial", style: .secondary) {
                        withAnimation {
                            self.currentPage = pageCount
                        }
                    }
                    .frame(width: 150)
                    
                    CustomButton("Next", style: .primary) {
                        withAnimation {
                            self.currentPage += 1
                        }
                    }
                    .frame(width: 150)
                } else {
                    CustomButton("Done", style: .primary) {
                        withAnimation {
                            self.currentPage = pageCount
                        }
                    }
                    .frame(width: 150)
                }
            }
            .padding(12)
        }
        .background(RoundedRectangle(cornerRadius: 20, style: .continuous).fill(colorScheme == .dark ? Color.black : Color.white))
        .frame(maxWidth: 600, maxHeight: 600)
    }
}

@ViewBuilder func TutorialPage1() -> some View {
    VStack(alignment: .leading, spacing: 16) {
        // Make view use whole available width
        HStack {
            Spacer()
            
            Text("Physarum")
            .font(Font.title.bold())
            
            Spacer()
        }
        
        Text("This playground simulates one or multiple species of Physarum polycephalum (commonly known as slime mold) using Metal compute shaders.")
        
        Text("Given two food sources, Physarum Polycephalum will find an approximation of the shortest path connecting the sources. With more than two food sources, the slime mold can solve transportation problems, which describe the allocation and distribution of resources.")
        
        Text("We can simulate its behavior using a swarm of agents. With the help of Metal, the number of agents can be in the millions on a discrete GPU.")
    }
}

@ViewBuilder func TutorialPage2() -> some View {
    VStack(alignment: .leading, spacing: 16) {
        // Make view use whole available width
        HStack {
            Spacer()
            
            Text("Agents")
            .font(Font.title.bold())
            
            Spacer()
        }
        
        HStack {
            Spacer()
            
            VStack {
                HStack {
                    Image(systemName: "viewfinder")
                    .resizable()
                    .frame(width: 20, height: 20, alignment: .center)
                    
                    Image(systemName: "viewfinder")
                    .resizable()
                    .frame(width: 20, height: 20, alignment: .center)
                    
                    Image(systemName: "viewfinder")
                    .resizable()
                    .frame(width: 20, height: 20, alignment: .center)
                }
                
                Image(systemName: "location.circle.fill")
                .resizable()
                .frame(width: 20, height: 20, alignment: .center)
            }
            
            Spacer()
        }
        
        Text("Each agent has three sensors, one in front, one to the side and two on its sides and it leaves behind a trail, when moving.")
        
        Text("With its sensors, it can pick up trails left behind by other agents and it will follow them.")
        
        Text("By changing the distance of the sensors from the agent or the angle between the sensors, we can change, how the agent behaves.")
        
        Text("Furthermore, we can change its behavior by controlling its movement and turning speed, as well as the strength of tails and the speed, at which they evaporate.")
    }
}

@ViewBuilder func TutorialPage3() -> some View {
    VStack(alignment: .leading, spacing: 16) {
        // Make view use whole available width
        HStack {
            Spacer()
            
            Text("Species")
            .font(Font.title.bold())
            
            Spacer()
        }
        
        Text("We can simulate multiple competing species of agents at the same time. Choose a preset with multiple species or change the number of species in the configuration panel.")
        
        Text("Agents will follow trails of their species and avoid trails of other species.")
    }
}

@ViewBuilder func TutorialPage4() -> some View {
    VStack(alignment: .leading, spacing: 16) {
        // Make view use whole available width
        HStack {
            Spacer()
            
            Text("Interactivity")
            .font(Font.title.bold())
            
            Spacer()
        }
        
        Text("Move the mouse over the simulation area to spawn new agents.")
        
        Text("Click and drag anywhere on the simulation area to add a new source of food. Agents of one species will attempt to build transportation networks between food sources.")
        Text("Right click and drag to add obstacles.")
        
        Text("Simulating transport networks between food sources works best with the last preset.")
    }
}

struct TutorialView: View {
    @State var currentPage: Int = 0
    var pageCount: Int = 4
    
    var isCompleted: Bool {
        currentPage >= pageCount
    }
    
    var body: some View {
        if isCompleted {
            EmptyView()
        } else {
            VStack {
                Spacer()
                
                HStack {
                    Spacer()
                    
                    TutorialContainer(pageCount: pageCount, currentPage: $currentPage) {
                        switch currentPage {
                        case 0:
                            TutorialPage1()
                        case 1:
                            TutorialPage2()
                        case 2:
                            TutorialPage3()
                        case 3:
                            TutorialPage4()
                        default:
                            EmptyView()
                        }
                    }
                    
                    Spacer()
                }
                
                Spacer()
            }
            .background(Color.black.opacity(0.3))
        }
    }
}
