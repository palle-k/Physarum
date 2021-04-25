/*:
 # Physarum
 
 This playground simulates one or multiple species of Physarum polycephalum (commonly known as slime mold) using a series of Metal compute shaders.
 
 ## Physarum Polycephalum
 
 Physarum Polycephalum is an acellular organism.
 This organism is interesting from a computer science standpoint, as its transport model can serve as an approximation of various computer science problems.
 
 Given two food sources, Physarum Polycephalum will find an approximation of the shortest path connecting the sources. With more than two food sources, the slime mold can solve transportation problems, which describe the allocation and distribution of resources.
 
 ## How to use this playground
 
 1. Run this Playground page
 2. If necessary, adapt the frame of the hosting view and run again.
 3. Use the Presets and Configuration panels to change the behavior of the simulation
 4. Move the mouse over the simulation area to spawn new agents
 5. Click and drag to add sources of food, right click and drag to add obstacles.
 
 ### Food Sources
 
 To add a food source, click anywhere on the simulation area. Food sources can be removed by clicking reset.
 Finding optimal paths between food sources works best with a single species and the last preset.
 Given some time, the simulation will find a set of paths that connects the food sources in an efficient way.
 I recommend optionally increasing the sensor offset and reducing the sensor angle spacing when trying this out. 
 
 ### Configuration
 
 **Number of Agents:**
 If the device running this playground does not have a discrete GPU, exceeding 100,000 agents can lead to slowdowns.
 
 **Agents**:
 Each agent has a position and is moving in a certain direction.
 The speed, at which the agent moves forward is controlled by the movement speed parameter.
 The agent has three sensors, which help it navigate; one facing forwards, the other ones facing to either side. If one sensor detects a stronger trail of same species agents than the others, the agent will move in that direction.
 The sensor offset controls the distance of the sensors from the agent.
 The sensor angle spacing controls, how far the sideways facing sensors are rotated away from the center sensor.
 The turn speed controls, how fast agents rotate towards stronger trails.
 
 **Trail Weight**:
 Controls, how strong the trail is, that agents leave behind.
 
 **Evaporation Speed**:
 Determines the time that it takes for trails to disappear from the map.
 
 **Species**:
 Optionally, up to three species can be simulated at the same time.
 Different species avoid each other. Avoidance of trails of other species is controlled by the sensor configuration.
 
 ## Technologies
 
 This playground uses MetalKit to render textures to the screen.
 Textures are rendered with Metal compute shaders.
 
 The preset and control panel has been built using SwiftUI.
 
 */

import PlaygroundSupport
import SwiftUI
import AppKit

let view = ContentView()

let hostingView = NSHostingView(rootView: view)
hostingView.frame = CGRect(x: 0, y: 0, width: 1400, height: 1100)

PlaygroundPage.current.liveView = hostingView

