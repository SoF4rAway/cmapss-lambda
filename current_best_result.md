================================================================================
                    ACADEMIC PERFORMANCE REPORT: RUL_1D_CNN (candidate_2)
================================================================================

1. PREDICTIVE ACCURACY (NASA C-MAPSS FD001 Test Set)
--------------------------------------------------------------------------------
- Root Mean Squared Error (RMSE):    14.8393
- NASA Asymmetric Score:            376.29
- Piecewise RUL Clipping:           125 Cycles

2. COMPUTATIONAL EFFICIENCY (Hardware-Aware Profile)
--------------------------------------------------------------------------------
- Average Inference Latency:        0.0350 ms
- Throughput:                       28555.01 samples/sec
- Model Size on Disk:               38.09 KB
- Deployment Format:                ONNX (Opset 14)
- CPU Target:                       x64 Single-Core (Spark Speed Layer Ready)

3. STATISTICAL ERROR ANALYSIS
--------------------------------------------------------------------------------
- Mean Error (Bias):                0.9723
- Standard Deviation:               14.8074
- Safety Margin:                    Negative (Aggressive)

4. BIG DATA & SPARK CONTEXT
--------------------------------------------------------------------------------
This model is specifically engineered for the Spark Speed Layer:
- **Zero Dynamic Branching**: The ONNX graph is static, ensuring consistent 
  latency and perfect execution tracing in distributed Spark executors.
- **Cache-Friendly**: The 38.1KB footprint allows the entire model 
  weights and intermediate activations to fit within L1/L2 caches, 
  minimizing memory bandwidth bottlenecks during high-frequency streaming.
- **Predictable CPU Usage**: The 0.04ms latency guarantees that a 
  single-core Spark executor can handle up to 28555 predictions 
  per second, easily satisfying sub-second inference requirements.

================================================================================