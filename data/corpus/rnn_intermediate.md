# Recurrent Neural Networks

## Hidden State and Recurrence
Recurrent neural networks maintain a hidden state that is updated at each time step as new input arrives. The same transition weights are applied repeatedly across the sequence, allowing the model to process variable-length inputs with a fixed parameter set. The hidden state acts as a compressed summary of past inputs, theoretically enabling the network to remember information from earlier time steps. This recurrence makes RNNs natural for speech, text, and time series where order matters.

## Sequence Processing
At each step, an RNN combines the current input with the previous hidden state to produce a new hidden state and optionally an output. Outputs can be produced at every step (sequence tagging) or only at the end (sequence classification). Because parameters are shared across time, the model generalizes across different sequence lengths. However, long sequences stress the model's ability to retain information in the hidden state without degradation.

## Backpropagation Through Time
Training unfolds the RNN over the sequence and applies backpropagation through time (BPTT), which is equivalent to backpropagation on an unrolled deep feedforward network with shared weights. Gradients flow backward through every time step. BPTT can be truncated to bounded history for efficiency, trading off the ability to credit early inputs. The unfolded graph makes clear that an RNN over many steps is effectively very deep, which exacerbates gradient scaling issues.

## Vanishing and Exploding Gradients in RNNs
Repeated application of the transition Jacobian across time steps can shrink or grow gradients exponentially. Vanishing gradients make it hard to learn long-range dependencies because early inputs have negligible influence on the loss gradient. Exploding gradients can destabilize training and are often mitigated by gradient clipping. These problems are typically more severe in RNNs than in feedforward nets of comparable depth because the same weights are multiplied across many sequential steps. LSTMs and GRUs were introduced specifically to improve gradient flow across long sequences.

## Elman Networks
Elman (1990) introduced simple recurrent networks for sequential prediction, laying groundwork for modern RNN research. While Elman networks are limited by vanishing gradients compared to gated architectures, they illustrate the core idea of recurrence and hidden state dynamics. Understanding Elman RNNs clarifies why more sophisticated recurrent units were needed for practical long-sequence modeling.
