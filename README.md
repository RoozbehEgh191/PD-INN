# PD-INN

This code combines the principle of Peridynamics (PD) with the concept of Physics-Informed Neural Networks (PINNs) to
predict the displacement field of a cracked plate.
The Ref. article for the PD-INN is (Eghbalpoor R. and Sheidaei A., 2024).
The following code simulates a plate subjected to a tensile load at the right edge. A pre-crack is considered within the domain.
The plate is discretized into a set of material points. The displacement field is predicted using a neural network and the loss function
is defined based on the principle of Peridynamics.
The code is tested on a single NVIDIA GeForce RTX 3060 Laptop GPU with Core(TM) i7-12650H. The code runtime was ~10min (dxdy=0.002).
The code is tested on a single NVidia A100 in the High Performance Computing facility at Iowa State University. The code runtime was ~9min (dxdy=0.001).
Reaserchers are encourages to develop the code and apply changes to simulate crack propagation by considering transfer learning technique and
partial training strategy as metioned in the paper.
