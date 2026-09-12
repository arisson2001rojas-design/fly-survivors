# fly-survivors

A whole-brain simulation of the fruit fly (*Drosophila melanogaster*), built from the
FlyWire connectome, that will play **Vampire Survivors**.

138,639 neurons and 15 million connections are simulated as leaky integrate-and-fire
units on the GPU. The goal: feed the game screen into the fly's eyes, read the
descending neurons that drive walking, and let the fly's own escape reflexes keep it
alive in a bullet-heaven game.

## Status

- [x] Connectome loader (FlyWire v783, via the files of Shiu et al. 2024)
- [x] LIF whole-brain simulator in PyTorch, CUDA-graph captured, all synapses
- [x] Validation: sugar-sensing neurons -> proboscis motor neuron MN9 fires
      (~112 Hz, ~400 active neurons, matching the reference model)
- [ ] Real-time speed (currently 0.27x with all synapses, 0.7x with >= 5 synapses per pair)
- [ ] Virtual eye: screen capture -> polar warp -> photoreceptor columns
- [ ] Motor readout: descending neurons -> virtual gamepad
- [ ] Vampire Survivors integration (dxcam + vgamepad), level-up menu handling
- [ ] Live visualization of the brain while it plays

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
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

`scripts/reference_brian2.py` runs the original Brian2 model on the same experiment
(slow, CPU) to compare numbers.

## Layout

```
src/flysurvivors/connectome.py   load + cache the edge list, id <-> index mapping
src/flysurvivors/lif.py          LIFBrain: GPU state, step(), CUDA graph capture, run()
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
