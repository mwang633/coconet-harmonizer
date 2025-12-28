"""
Coconet Model Architecture

A convolutional neural network for Bach chorale harmonization.
Uses masked training to learn to fill in missing voices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class Coconet(nn.Module):
    """
    Coconet: Counterpoint by Convolution

    A CNN that learns to harmonize Bach chorales by training on
    masked reconstruction. Given partial scores, it learns to
    fill in the missing voices.

    Input shape: (batch, num_voices, time_steps, num_pitches)
    - num_voices: 4 (Soprano, Alto, Tenor, Bass)
    - time_steps: sequence length (typically 32 or 64)
    - num_pitches: MIDI pitch range (typically 128 or subset)

    The model takes a piano roll representation where each voice
    is a separate channel, and each time step shows which pitch
    is active (one-hot style, but can have no pitch = rest).
    """

    def __init__(
        self,
        num_pitches: int = 62,  # MIDI 36-97 covers SATB range
        num_voices: int = 4,
        num_layers: int = 32,
        num_filters: int = 128,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.num_pitches = num_pitches
        self.num_voices = num_voices
        self.num_layers = num_layers
        self.num_filters = num_filters

        # Input: concatenate piano roll + mask for each voice
        # Piano roll: (batch, voices, time, pitches)
        # Mask: (batch, voices, time, 1) - broadcast over pitches
        input_channels = num_voices * 2  # voices + masks

        # Initial projection
        self.input_conv = nn.Conv2d(
            input_channels, num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)  # Cycle: 1, 2, 4, 8, 1, 2, ...
            self.blocks.append(
                ResidualBlock(num_filters, kernel_size, dilation)
            )

        # Output projection - predict pitch distribution for each voice
        self.output_conv = nn.Conv2d(
            num_filters, num_voices * num_pitches,
            kernel_size=1
        )

    def forward(
        self,
        pianoroll: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pianoroll: (batch, voices, time, pitches) - the input music
            mask: (batch, voices, time) - 1 where voice is known, 0 where to predict

        Returns:
            logits: (batch, voices, time, pitches) - predicted pitch logits
        """
        batch, voices, time, pitches = pianoroll.shape

        # Expand mask to match pianoroll shape
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, -1, pitches)

        # Mask the input (zero out unknown positions)
        masked_input = pianoroll * mask_expanded

        # Concatenate input and mask info
        # Reshape for conv2d: treat (time, pitches) as spatial dims
        mask_channel = mask.unsqueeze(-1).expand(-1, -1, -1, pitches)
        x = torch.cat([masked_input, mask_channel], dim=1)  # (batch, voices*2, time, pitches)

        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        logits = self.output_conv(x)  # (batch, voices*pitches, time, pitches)

        # Reshape to (batch, voices, time, pitches)
        logits = logits.view(batch, voices, pitches, time, pitches)
        # We want the last dim to be the pitch prediction
        # Take diagonal: for each input pitch position, predict output pitch
        # Actually, we average over the input pitch dimension
        logits = logits.mean(dim=2)  # (batch, voices, time, pitches)

        return logits

    def sample(
        self,
        pianoroll: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        num_iterations: int = 100,
    ) -> torch.Tensor:
        """
        Gibbs sampling to generate harmonization.

        Iteratively samples missing voices using annealed Gibbs sampling.

        Args:
            pianoroll: Initial piano roll (can have empty voices)
            mask: Which voices are given (1) vs to generate (0)
            temperature: Sampling temperature (lower = more deterministic)
            num_iterations: Number of Gibbs sampling iterations

        Returns:
            Completed piano roll with all voices filled in
        """
        self.eval()
        device = next(self.parameters()).device
        pianoroll = pianoroll.to(device)
        mask = mask.to(device)

        batch, voices, time, pitches = pianoroll.shape
        result = pianoroll.clone()

        # Positions to sample (where mask is 0)
        sample_positions = (mask == 0).nonzero(as_tuple=False)

        if len(sample_positions) == 0:
            return result

        with torch.no_grad():
            for iteration in range(num_iterations):
                # Annealed temperature
                current_temp = temperature * (1 - iteration / num_iterations) + 0.1

                # Random order for Gibbs sampling
                perm = torch.randperm(len(sample_positions))

                for idx in perm:
                    pos = sample_positions[idx]
                    b, v, t = pos[0], pos[1], pos[2]

                    # Create temporary mask showing current position as unknown
                    temp_mask = mask.clone()
                    temp_mask[b, v, t] = 0

                    # Get predictions
                    logits = self.forward(result, temp_mask)

                    # Sample from the predicted distribution
                    probs = F.softmax(logits[b, v, t] / current_temp, dim=-1)
                    sampled_pitch = torch.multinomial(probs, 1).item()

                    # Update result
                    result[b, v, t] = 0
                    result[b, v, t, sampled_pitch] = 1

        return result


def create_model(
    num_pitches: int = 62,
    num_voices: int = 4,
    num_layers: int = 32,
    num_filters: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Coconet:
    """Create and return a Coconet model."""
    model = Coconet(
        num_pitches=num_pitches,
        num_voices=num_voices,
        num_layers=num_layers,
        num_filters=num_filters,
    )
    return model.to(device)
