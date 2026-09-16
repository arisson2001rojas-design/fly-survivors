# FlyTouhou architecture design

Date: 2026-09-16
Status: Approved design, not yet implemented
Target game: Touhou Koumakyou: Classic (`th06c.exe`)
Target brain host: Linux + AMD Radeon RX 9070
Target game host: Windows 10

## 1. Goal

FlyTouhou will connect the existing whole-brain Drosophila FlyWire v783 LIF simulation in `fly-survivors` to Touhou Koumakyou: Classic. Windows runs Touhou and extracts only sensory/game-state information needed to construct what the fly can see. Linux runs the visual drive, FlyWire brain simulation, motor readout, and session telemetry. Motor commands return to Windows over the LAN and are applied automatically to Touhou.

The system should behave as a closed-loop experiment rather than a conventional game bot. Windows may expose visible geometry and game lifecycle state, but it must not compute the dodge direction, collision prediction, optimal path, or future bullet impact. Those decisions must emerge from the sensory drive, connectome simulation, and motor readout.

The first version focuses on survival and reflexive movement. Shooting is held automatically. Focus and bomb are disabled initially and may be added later as additional motor outputs.

## 2. Fixed experimental assumptions

- Touhou is configured with 5 starting lives.
- Losing one life does not reset the brain.
- Using a Continue does not reset the brain.
- Up to 3 Continues may be used automatically.
- The brain is reset only after the run is definitively over because no Continues remain.
- A new run starts with a clean neuronal state.
- Historical telemetry is never deleted by a brain reset.
- History persists across program restarts and is grouped into sessions and runs.
- Extra lives / 1UPs are recorded explicitly rather than assuming a fixed total number of lives.
- The current LIF network has fixed connectivity, so improved scores or survival across runs are not by themselves evidence of synaptic learning. The history is intended to measure performance and later provide a baseline for any future plasticity experiment.

## 3. Verified game build

The selected executable is the Steam `th06c.exe` build with SHA-256:

```text
1E2F280EE8EDE3018AABBD72897A46684957EE18442A89C3E2D6194C06B628FD
```

This matches the publicly studied 64-bit Classic build. Existing reverse-engineering work establishes that this executable is PE32+/x86-64 and uses ASLR, so runtime addresses must be treated as:

```text
runtime_address = module_base + RVA
```

Known public findings useful to FlyTouhou include:

- menu cursor: `module + 0x00B5C168`
- internal score: `module + 0x003A3B4C`
- keyboard polling is obtained through dynamically resolved `GetKeyboardState`, so x64 input hooks must account for the `GetProcAddress` path
- this Classic build must not reuse original TH06 32-bit offsets
- stage, lives, graze, and the game-object pools required by FlyTouhou are not considered verified until they are independently mapped and validated on this exact SHA

The bridge must refuse to enable automatic memory-based control if the executable hash changes.

## 4. High-level architecture

```text
WINDOWS 10                                  LINUX / RX 9070

Touhou th06c.exe
     |
     | internal state @ game tick
     v
FlyTouhou Windows Agent
     |  player / visible bullets / lasers /
     |  enemies / lifecycle state
     |
     | geometry -> retinal stimulus
     +---------------- UDP ----------------> RetinaDrive / visual inputs
                                             |
                                             v
                                       FlyWire v783 LIF
                                             |
                                             v
                                       MotorReadout
                                             |
                                             v
                                       Locomotion / action
     <---------------- UDP -----------------+
     |
     v
keyboard/control injection

Reliable event channel <------------------> run lifecycle + telemetry

Linux telemetry ----> Windows local relay ----> browser WebGL 3D brain viewer
```

The brain simulation is authoritative for movement. The visualization and logging paths are observers and must never block simulation or game control.

## 5. Touhou integration strategy

### 5.1 Phase A: external probe

Before injecting a permanent bridge, a Windows probe will inspect the running `th06c.exe` process and help identify the required structures and RVAs on the verified SHA.

The probe must map and validate, in this order:

1. player position and player state
2. enemy bullet pool and bullet attributes
3. laser pool
4. enemies and bosses
5. remaining lives / extra lives
6. Continue state and count
7. stage and overall game state

Validation is behavioral: move the player, pause, spawn bullets, lose a life, enter a Continue screen, and compare candidate memory values against the visible game state. A value is not accepted only because it changes plausibly.

### 5.2 Phase B: injected x64 bridge

After the structures are verified, an x64 DLL bridge will run inside `th06c.exe` and read the relevant structures directly once per game update. This avoids repeated cross-process `ReadProcessMemory` calls in the final 60 Hz path.

The bridge produces a compact snapshot containing only current state, for example:

```text
frame_id
timestamp
player
visible_bullets[]
visible_lasers[]
visible_enemies[]
lives
continues_used
stage
game_state
```

The exact binary schema is defined during implementation after the object layouts are proven.

### 5.3 Input injection

The final Windows side must apply movement and menu actions automatically. For this build, the design should follow the already observed `GetKeyboardState`/`GetProcAddress` behavior rather than assuming the original TH06 DirectInput path.

Gameplay input and menu automation are separate concerns:

- gameplay: movement direction plus held shot
- lifecycle automation: Start, Continue, restart after final Game Over, and any required menu confirmation

The state machine suppresses normal movement commands while the game is not in active gameplay.

## 6. Game lifecycle state machine

The Windows Agent maintains an explicit lifecycle state. A conceptual flow is:

```text
STARTING
   -> PLAYING

PLAYING
   -> RESPAWNING       on life loss
   -> CONTINUE_SCREEN  after all current lives are exhausted and a Continue is offered
   -> STAGE_TRANSITION when appropriate
   -> FINAL_GAME_OVER  when no Continue remains

RESPAWNING
   -> PLAYING

CONTINUE_SCREEN
   -> automatically select Continue
   -> PLAYING

FINAL_GAME_OVER
   -> emit reliable run-end event
   -> request brain reset
   -> automatically start a new run
   -> PLAYING
```

Losing a life or using Continue must not reset the brain. The brain keeps evolving during death/reappearance animations and Continue transitions unless later measurement shows that Touhou fully freezes the simulation and an explicit pause is needed for synchronization.

Only `FINAL_GAME_OVER` starts a new neuronal trial.

## 7. Sensory representation

### 7.1 Do not stream screenshots as the primary sensory path

The final design does not send full RGB game frames over the LAN. The Windows side reads the game objects and directly constructs the sensory stimulus. Screenshot capture remains useful only for debugging, recordings, visual verification, and fallback state recognition.

### 7.2 Whole visible playfield

The fly receives the full visible Touhou playfield, not only a small crop around the player. Geometry is transformed into a player-centered, egocentric representation before retinal projection.

The sensory transform may use:

- current relative position
- current apparent size / footprint
- current visible luminance or contrast
- current motion implied by object displacement between game ticks

It must not provide:

- `will_hit` flags
- predicted future positions
- optimal dodge vectors
- nearest safe point
- pathfinding results
- information about off-screen objects that the fly could not see

### 7.3 Monochrome first version

The existing `VirtualEye`/`RetinaDrive` path is intensity/contrast based rather than RGB-color based. The first Touhou bridge therefore uses a monochrome retinal stimulus.

Color is a future experimental extension, not a requirement for the initial closed loop.

### 7.4 Retinal output

The goal is a compact per-column retinal vector rather than a pixel image. Static eye geometry is initialized once. On each Touhou tick, the Windows side projects visible objects onto the virtual eye and sends the current retinal stimulus.

The sensory bridge must preserve enough spatial and temporal structure for looming and motion-sensitive pathways to respond without deciding the correct action itself.

## 8. Timing and LAN transport

Touhou runs at approximately 60 Hz, so a game tick is about 16.7 ms. The design target is to keep the complete perception-to-action loop near one Touhou frame whenever practical.

### 8.1 Latest-only real-time channels

Latency matters more than reliable delivery for rapidly changing sensory and motor data. Therefore:

- Windows -> Linux retinal stimulus: UDP, latest-only
- Linux -> Windows motor command: UDP, latest-only
- Linux -> visualization relay: UDP/latest-only for live neural activity

Each real-time packet carries at least:

```text
protocol_version
message_type
sequence_number
timestamp
payload
```

Missing sequence numbers are recorded for diagnostics but never block waiting for retransmission. If a consumer falls behind, stale real-time packets are discarded.

### 8.2 Reliable event channel

A reliable TCP/WebSocket-style channel carries low-rate, non-discardable state such as:

- session start/end
- run start/end
- life lost
- Continue used
- 1UP gained
- stage transition
- final Game Over
- brain reset acknowledgement
- history metadata
- compatibility / hash failure

### 8.3 Browser limitation

The browser does not receive raw UDP directly. The Windows Agent acts as the local relay: it receives neural telemetry and exposes it to the browser through localhost WebSocket or an equivalent browser-friendly transport.

## 9. Motor behavior

The first version keeps the current conceptual motor path:

```text
visual drive -> FlyWire -> descending-neuron readout -> locomotion -> direction
```

Touhou output is digital keyboard movement. The first implementation should preserve the existing eight-direction concept and adapt key bindings to Touhou's actual controls.

Initial action policy:

- movement: brain-controlled
- shot: continuously held
- focus: disabled
- bomb: disabled
- menu / Continue / restart: Windows state machine, not brain-controlled

The automatic menu actions are infrastructure rather than gameplay intelligence and are excluded from the biological decision loop.

## 10. Brain reset semantics

A neuronal reset occurs only after final Game Over when no Continues remain.

Reset means the next run starts with fresh transient simulation state, including the relevant membrane potentials, conductances, spike-history state, sensory adaptation state, and motor smoothing state as defined by the current simulator reset APIs.

Persistent items that are not reset:

- connectome structure
- model parameters
- application configuration
- run/session history
- executable compatibility records

No synaptic-learning state exists in the current model. If plasticity is added later, its persistence policy must be specified separately rather than silently coupled to the existing reset.

## 11. Persistent history

History is one cumulative dataset across days, organized by sessions and runs.

Conceptual structure:

```text
history
  session
    run
      segment / Continue period
        life
        events
```

Each execution of FlyTouhou creates a new session. Each final Game Over ends one run. A run may contain the initial segment plus up to three Continue segments.

Record at minimum:

- session ID and wall-clock timestamps
- run number
- executable SHA-256
- model/config version
- stage reached
- run duration
- life start/end times
- deaths
- Continue events
- 1UP events
- score when available
- motor summary
- selected visual-neuron and descending-neuron activity
- packet loss and end-to-end timing statistics
- a short high-detail neural window around important events such as deaths

Do not continuously persist every value of all ~138k neurons at every LIF step. Live visualization can be high-dimensional, but long-term storage should be event-focused and summarized.

The history can be used to plot survival time, stage progress, lives consumed, Continue usage, and other performance trends. Such trends are descriptive performance measurements; with fixed synapses they are not proof that the brain learned.

## 12. 3D brain visualization

A browser-based WebGL viewer on Windows displays the FlyWire brain while Linux continues simulating independently.

### 12.1 Static data

Neuron anatomical coordinates (`pos_x`, `pos_y`, `pos_z`) and stable neuron identifiers are transferred/cached once rather than retransmitted every frame.

### 12.2 Live data

Linux aggregates recent spike activity into a visualization rate around 20-30 updates per second. The viewer receives only changing activity and selected summary rates, then illuminates the corresponding neurons in the 3D scene.

The viewer should support:

- rotate / pan / zoom
- live activity highlighting
- selected visual and descending-neuron rates
- current stage, life, Continue count, run and session
- pause of the visualization without pausing the simulation
- replay / rewind around recorded death events

The viewer is strictly observational. Closing it or slowing it down must not alter Touhou control or brain timing.

## 13. Failure handling and compatibility

### Game executable changed

Before enabling memory-based automation, Windows computes the SHA-256 of `th06c.exe`. If it differs from the verified build, memory hooks are disabled and the program reports that the build requires revalidation.

### Lost UDP packets

Record sequence gaps and continue with the newest packet. Never build a latency queue to recover obsolete sensory or motor frames.

### Linux brain unavailable

Windows releases all held gameplay keys and stops automatic control rather than continuing the last movement indefinitely.

### Windows Agent unavailable

Linux stops emitting meaningful game-control output but may keep the brain service alive for diagnostics.

### Viewer unavailable

No effect on simulation or gameplay.

### Invalid object data

The bridge validates pool pointers/counts/ranges before serializing. A failed validation disables the affected automatic path instead of trusting stale offsets.

## 14. Validation strategy

Implementation is staged so each boundary can be measured before the next one is enabled.

1. Read-only external probe identifies player state.
2. Identify and validate bullets and lasers.
3. Identify and validate enemies.
4. Identify lives, Continue state, stage, and final Game Over.
5. Show a diagnostic visualization of the geometry read from memory and compare it with Touhou.
6. Produce the monochrome retinal representation while Touhou remains human-controlled.
7. Measure Windows -> Linux LAN timing and packet loss.
8. Feed retina into the existing brain without returning controls.
9. Return motor output to Windows and control movement only.
10. Hold shot automatically.
11. Automate death/Continue/restart lifecycle.
12. Enable persistent history.
13. Enable live 3D brain visualization.
14. Run long sessions containing many full runs.

For every closed-loop run, record enough timing data to separate a biological/control failure from an infrastructure failure. At minimum measure:

- Touhou tick timestamp
- sensory snapshot timestamp
- packet receive timestamp
- brain simulation interval
- motor packet timestamp
- Windows action-application timestamp
- sequence gaps

Primary initial success criteria:

- no backlog of stale sensory/motor packets
- control remains synchronized with the active Touhou run
- automatic Continue/restart works without manual intervention
- life loss does not reset the brain
- final Game Over does reset the brain exactly once before the next run
- the retinal representation corresponds to visible game geometry
- the observer/3D viewer can be stopped with no effect on gameplay
- history survives process restarts and keeps runs separated correctly

A 60 Hz closed loop is the target, but final rate claims must come from measurements of the complete system rather than extrapolation from the isolated LIF benchmark.

## 15. Scope boundaries for the first implementation

Included:

- exact verified Steam Classic build
- x64 external probe then x64 bridge
- player/bullet/laser/enemy/lifecycle extraction
- full-playfield egocentric monochrome retinal drive
- LAN split between Windows and Linux
- brain-controlled movement
- automatic shot
- automatic Continue and restart
- persistent session/run history
- live 3D observer with death-event rewind

Explicitly deferred:

- New Classic (`th06nc.exe`) support
- original 2002 32-bit TH06 support
- color vision
- brain-controlled shot timing
- focus control
- bomb control
- synaptic plasticity / learning rules
- predictive dodge helpers
- pathfinding
- training or reward optimization

## 16. Public research references used for the integration design

- `vittorioromeo/th12_hfr`, `docs/games/TH06NC_DEVNOTES.md`: x64 build fingerprints, ASLR/RVA conventions, and reverse-engineering notes for the 2026 releases.
- `hakatashi/sattori`, `worker/docs/titles/th06c.md`: exact `th06c` build behavior, x64 recording path, `GetKeyboardState` resolution path, menu cursor and score RVAs, and currently unmapped state.
- Existing `fly-survivors` code: `eye.py`, `agent.py`, `motor.py`, `viz.py`, and whole-brain LIF implementation.

These references are starting evidence, not permission to reuse offsets from a different executable. Every gameplay structure used by the final bridge must be validated against the exact SHA listed above.
