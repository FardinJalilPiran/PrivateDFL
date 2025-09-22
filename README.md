# PrivateDFL

The Private Decentralized Federated Learning (PrivateDFL) model is a framework for privacy-preserving, decentralized federated learning that combines hyperdimensional computing, differential privacy, and an explainable AI-guided noise accounting system. The model enables multiple clients to collaboratively train a machine learning model without a central server, while also providing formal privacy guarantees. 

## Key Features

- **Decentralized and Serverless**: Unlike traditional centralized federated learning, which uses a central server to aggregate model updates, PrivateDFL allows clients to train and exchange models in a sequence, eliminating the single point of failure.
- **Adaptive Differential Privacy**: The model uses an explainable AI-guided "noise accountant" to keep a running total of the cumulative noise added across all training rounds. Each client then adds only the incremental noise needed to meet the privacy requirements, rather than adding a full amount of noise each time. This prevents the excessive noise that can significantly degrade model performance in traditional differential privacy-federated learning methods.
- **Hyperdimensional computing (HDC)**: PrivateDFL uses HDC for its representational basis. HDC is a robust and interpretable machine learning paradigm that is well-suited for decentralized environments.
- **Efficient and Robust Performance**: PrivateDFL demonstrates better accuracy, lower latency, and less energy consumption compared to centralized deep learning baselines, particularly in non-independently and identically distributed (non-IID) data scenarios. 

## Citation
If you use PrivateDFL in your research, please cite our paper:

```bibtex
@article{jalil2025privacy,
  title={Privacy-Preserving Decentralized Federated Learning via Explainable Adaptive Differential Privacy},
  author={Piran, Fardin Jalil and Chen, Zhiling and Zhang, Yang and Zhou, Qianyu and Tang, Jiong and Imani, Farhad},
  journal={arXiv preprint arXiv:2509.10691},
  year={2025}
}
```
