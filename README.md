# fly-survivors

A whole-brain simulation of the fruit fly (*Drosophila melanogaster*), built from the
FlyWire connectome, that will play **Vampire Survivors**.

138,639 neurons and 15 million connections are simulated as leaky integrate-and-fire
units on the GPU. The goal: feed the game screen into the fly's eyes, read the
descending neurons that drive walking, and let the fly's own escape reflexes keep it
alive in a bullet-heaven game.

## Status

- [x] Connectome loader (FlyWire v783, via the files of Shiu et al. 2024)
- [x] LIF whole-brain simulator, all 15 M synapses, PyTorch + Triton, CUDA-graph captured
- [x] Validation against the original Brian2 model: sugar-sensing neurons -> proboscis
      motor neuron MN9 at 90 Hz (reference: 88 Hz), ~400 active neurons, same top neurons
- [x] Real time: 80 us per 0.1 ms step on an RTX 4080 SUPER (1.2x real time)
- [ ] Virtual eye: screen capture -> polar warp -> photoreceptor columns
- [ ] Motor readout: descending neurons -> virtual gamepad
- [ ] Vampire Survivors integration (dxcam + vgamepad), level-up menu handling
- [ ] Live visualization of the brain while it plays

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install triton-windows                # Triton on Windows (Linux: pip install triton)
pip install -e .[dev]
python scripts/download_data.py        # ~105 MB, once
python scripts/validate_proboscis.py   # sugar GRNs -> MN9
python scripts/benchmark.py            # speed on your GPU
pytest                                 # unit tests on a synthetic network (CPU)
```

## Model

Same equations and constants as Shiu et al. 2024 (Brian2 model), reimplemented with a
fixed 0.1 ms step and no host-side control flow so a whole step runs as one CUDA graph:

```
dv/dt = (v_rest - v + g) / tau_m         v_rest = v_reset = -52 mV, v_th = -45 mV, tau_m = 20 ms
dg/dt = -g / tau_syn                     tau_syn = 5 ms
spike: v = v_reset, g = 0, refractory 2.2 ms
presynaptic spike: g_post += 0.275 mV * signed synapse count, after 1.8 ms
```

Excitatory / inhibitory sign comes from the predicted neurotransmitter of the
presynaptic neuron (GABA and glutamate inhibitory). "Optogenetic" activation of a set
of neurons is Poisson spiking at a chosen rate, as in the reference.

One detail that matters: in the Brian2 reference, synaptic input arriving while a
neuron is refractory is dropped, not accumulated. Accumulating it makes the whole
network ~25% more excitable (MN9 at 112 Hz instead of 88 Hz). This model drops it too.

`scripts/reference_brian2.py` runs the original Brian2 model on the same experiment
(slow, CPU) to compare numbers.

### Speed

| Backend | Propagation | us / step | real time |
|---|---|---|---|
| torch | CSR sparse mat-vec over all 15 M edges | 376 | 0.27x |
| triton | event-driven, only out-edges of neurons that spiked | 80 | 1.2x |

Only a few dozen neurons spike per 0.1 ms step, so touching all edges every step is
wasted work. The Triton kernel launches one program per presynaptic neuron; programs
whose neuron is silent exit immediately. A second fused kernel does the LIF update,
Poisson forcing and the delay ring-buffer write. The whole step is replayed as one
CUDA graph. Triton on Windows comes from the `triton-windows` package.

## Layout

```
src/flysurvivors/connectome.py   load + cache the edge list, id <-> index mapping
src/flysurvivors/lif.py          LIFBrain: state, torch reference step, CUDA graph capture, run()
src/flysurvivors/kernels.py      Triton kernels: event-driven propagation + fused LIF update
src/flysurvivors/neurons.py      known root ids (sugar GRNs, MN9, later: eye + descending neurons)
scripts/                         download, validation, benchmark, reference
tests/                           tiny synthetic networks checking delay, refractory, summation
```

## Data and credits

- Connectome: FlyWire (Dorkenwald et al. 2024, Schlegel et al. 2024), version 783.
  FlyWire data is released for non-commercial use with attribution; see flywire.ai.
- Model and data files: Shiu et al. 2024, *A leaky integrate-and-fire computational
  model based on the connectome of the entire adult Drosophila brain reveals insights
  into sensorimotor processing* (Nature), code at github.com/philshiu/Drosophila_brain_model (MIT).
- Vampire Survivors is a game by poncle. This project sends inputs to a locally running
  copy and is not affiliated with it.

## License

MIT for the code in this repository. The connectome data keeps its own license.
