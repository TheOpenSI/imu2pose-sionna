## Human pose reconstruction from IMUs using 5G neural-receiver

### Dependency
Install `numpy`, `matplotlib`, `tensorflow`, `sionna`, `sionna-vispy`:

```
pip install -r requirements.txt
```

### Features
- [x] Quantize IMU data 
- [x] Transmit quantized IMU data over OFDM channel
- [x] Channel dataset generator with Ray Tracing
- [x] Neural-receiver (uplink)
- [ ] Downlink beamforming 
- [ ] Deep learning-based channel estimator (at base station)
- [x] Human body visualization with `smpl`
- [ ] Human body within 3D map 

### To-do
- [x] Fixing Perfect-CSI baseline
